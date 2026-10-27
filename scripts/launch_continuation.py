#!/usr/bin/env python3
"""Queue fresh-data, one-step continuations after the existing GPU study workers."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import yaml

from few_step_diffusion.config import config_fingerprint, load_config
from few_step_diffusion.continuation import continuation_config
from few_step_diffusion.runtime import atomic_json, experiment_lock, sha256
from launch_scale50k import process_matches, process_record

ROOT = Path(__file__).resolve().parents[1]
# Current user scope excludes all edge-loss experiments. Existing saved plans
# retain their provenance; active recovery uses run_selected_experiment.py.
GROUPS = {0: ["baseline"], 1: ["dmd"],
          2: ["swiftbrush_lora", "consistency"], 3: ["instaflow"]}


def now():
    return datetime.now(timezone.utc).isoformat()


def supervise(folder):
    manifest = json.loads((folder/"launch.json").read_text())
    state = {"pid": os.getpid(), "status": "running", "started_at": now(),
             "data": {"status": "queued"}, "workers": {}}
    mutex = threading.Lock()

    def update(section=None, **values):
        with mutex:
            target = state if section is None else state.setdefault(section, {})
            target.update(values)
            state["updated_at"] = now()
            atomic_json(folder/"status.json", state)

    def worker_update(gpu, **values):
        with mutex:
            state["workers"].setdefault(str(gpu), {}).update(values)
            state["updated_at"] = now()
            atomic_json(folder/"status.json", state)

    def command(args, logfile, gpu=None, phase=None):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print("START", phase, args, flush=True)
        with (folder/logfile).open("ab", buffering=0) as log:
            child = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            if gpu is None:
                update("data", status="downloading", child_pid=child.pid)
            else:
                worker_update(gpu, status="running", phase=phase, child_pid=child.pid)
            code = child.wait()
        if code:
            raise RuntimeError(f"{phase} failed with exit {code}; see {folder/logfile}")

    python = sys.executable
    base = [python, "-m", "few_step_diffusion.cli"]

    def prepare():
        try:
            command([python, str(Path(__file__).resolve()), "--prepare", str(folder)],
                    "prepare.log", phase="download_fresh_training_images")
            update("data", status="complete", child_pid=None, finished_at=now())
        except BaseException as exc:
            update("data", status="failed", error=str(exc), child_pid=None)
            raise

    def wait_for_gpu(gpu):
        prior = manifest.get("prior_study")
        if not prior:
            return
        while True:
            old = json.loads(Path(prior["status_file"]).read_text())
            worker = old.get("workers", {}).get(str(gpu), {})
            if worker.get("status") == "complete":
                return
            if worker.get("status") == "failed" or not process_matches(prior["process"]):
                raise RuntimeError(f"Prior GPU {gpu} worker did not finish; inspect {prior['status_file']}")
            worker_update(gpu, status="queued", phase="waiting_for_existing_gpu_queue",
                          prior_phase=worker.get("phase"), child_pid=None)
            time.sleep(20)

    def worker(gpu, methods, prepared):
        try:
            worker_update(gpu, status="queued", methods=methods, phase="waiting_for_existing_gpu_queue")
            wait_for_gpu(gpu)
            for method in methods:
                job = manifest["jobs"][method]
                parent, output = Path(job["parent_run"]), Path(job["output_dir"])
                completed = json.loads((parent/"run.json").read_text())
                config = continuation_config(completed["config"], parent, output,
                    manifest["train_offset"], manifest["fresh_training_images"], manifest["additional_updates"])
                plan = folder/"plans"/f"{method}_seed42.yaml"
                plan.parent.mkdir(parents=True, exist_ok=True)
                if plan.exists() and config_fingerprint(load_config(plan)) != config_fingerprint(config):
                    raise ValueError(f"Parent settings changed since this continuation was queued: {parent}")
                plan.write_text(yaml.safe_dump(config, sort_keys=False))
                if config["train"]["mode"] not in ("vsd", "dmd", "instaflow"):
                    worker_update(gpu, status="queued", phase="waiting_for_fresh_images", method=method)
                    prepared.result()
                while shutil.disk_usage(ROOT).free < (15 if method == "instaflow" else 8)*2**30:
                    worker_update(gpu, status="queued", phase="waiting_for_disk_space", method=method)
                    time.sleep(30)
                with experiment_lock(output):
                    if (output/"run.json").exists():
                        finished = json.loads((output/"run.json").read_text())
                        if (finished.get("status") != "complete" or
                                config_fingerprint(finished["config"]) != config_fingerprint(config)):
                            raise ValueError(f"Completed continuation config differs: {output}")
                    else:
                        args = base+["train", "--config", str(plan)]
                        if (output/"last.pt").exists():
                            args.append("--resume")
                        worker_update(gpu, method=method, run_dir=str(output))
                        command(args, f"gpu{gpu}.log", gpu, f"continue_one_step_{method}")
                    result_dir = Path(job["result_dir"])
                    result = result_dir/f"{method}_seed42_1steps.result.json"
                    if not result.exists():
                        command(base+["evaluate", "--config", str(plan), "--checkpoint",
                            str(output/"stage_1"), "--steps", "1", "--result-dir", str(result_dir)],
                            f"gpu{gpu}.log", gpu, f"evaluate_5000_{method}")
                    else:
                        saved = json.loads(result.read_text())
                        if (saved["config_hash"] != config_fingerprint(config) or
                                saved["model_identity"] != sha256(output/"stage_1/adapter.safetensors")):
                            raise ValueError(f"Saved continuation result differs: {result}")
                worker_update(gpu, last_completed_method=method, child_pid=None)
            worker_update(gpu, status="complete", phase="complete", finished_at=now(), child_pid=None)
        except BaseException as exc:
            worker_update(gpu, status="failed", phase="failed", error=str(exc), child_pid=None)
            raise

    update()
    errors = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        prepared = pool.submit(prepare)
        futures = [pool.submit(worker, int(gpu), methods, prepared)
                   for gpu, methods in manifest["groups"].items()]
        for future in as_completed([prepared, *futures]):
            try:
                future.result()
            except BaseException as exc:
                errors.append(str(exc))
                print("FAILED", str(exc), flush=True)
    if errors:
        update(status="failed", errors=errors)
        raise RuntimeError("Continuation tasks failed; see status.json")
    # The controlled cohort excludes SwiftBrush's intentionally separate CFG 4.5 recipe.
    command([python, "scripts/generate_qualitative.py", "--run-root", manifest["run_root"],
        "--output-dir", manifest["result_root"]+"/qualitative", "--gpu", "0", "--seed", "42",
        "--resume", "--methods", *[m for ms in GROUPS.values() for m in ms if m != "swiftbrush_lora"]],
        "report.log", 0, "generate_qualitative")
    command([python, "scripts/build_report.py", "--profile", "sd15", "--results-dir",
        manifest["result_root"], "--study-config", str(folder/"study.yaml"), "--require-results"],
        "report.log", 0, "compile_report")
    update(status="complete", phase="complete", finished_at=now())


def prepare(folder):
    from few_step_diffusion.sd_data import caption_records, manifest_hash, prepare_images
    config = load_config(folder/"study.yaml")
    parent_config = load_config("configs/sd15_50k.yaml")
    old, fresh, validation = (caption_records(parent_config), caption_records(config),
                              caption_records(config, "validation"))
    def ids(rows):
        return {row["image_id"] for row in rows}
    if ids(old) & ids(fresh) or (ids(old) | ids(fresh)) & ids(validation):
        raise ValueError("Fresh training and held-out validation manifests overlap")
    atomic_json(folder/"dataset_manifest.json", {"fresh_images": len(fresh),
        "parent_images": len(old), "combined_unique_images": len(ids(old) | ids(fresh)),
        "train_overlap": 0, "validation_overlap": 0, "fresh_prompt_hash": manifest_hash(fresh),
        "parent_prompt_hash": manifest_hash(old), "validation_prompt_hash": manifest_hash(validation),
        "fresh_records": fresh})
    prepare_images(config, fresh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=18000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--supervise", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--prepare", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.prepare:
        return prepare(args.prepare)
    if args.supervise:
        try:
            return supervise(args.supervise)
        except BaseException as exc:
            path = args.supervise/"status.json"
            status = json.loads(path.read_text()) if path.exists() else {"pid": os.getpid()}
            status.update(status="failed", error=str(exc), finished_at=now())
            atomic_json(path, status)
            raise
    if args.updates < 2:
        parser.error("Use at least two additional updates")
    run_root, result_root = "runs/sd15/fresh68287", "results/sd15/fresh68287"
    manifest = {"train_offset": 50000, "fresh_training_images": 68287,
        "combined_unique_training_images": 118287, "additional_updates": args.updates,
        "effective_batch_size": 16, "sampling_steps": 1, "nfe": 1, "student_lora_rank": 32,
        "sampling": "shuffled_epochs", "groups": GROUPS, "run_root": run_root,
        "result_root": result_root, "jobs": {}}
    prior_pointer = ROOT/"artifacts/launches/latest_sd15_scale50k.txt"
    if prior_pointer.exists():
        status_file = Path(prior_pointer.read_text().strip())/"status.json"
        status = json.loads(status_file.read_text())
        if status["status"] == "running":
            manifest["prior_study"] = {"status_file": str(status_file), "process": process_record(status["pid"])}
    for gpu, methods in GROUPS.items():
        for method in methods:
            parent = "runs/sd15/swiftbrush_lora_seed42" if method == "swiftbrush_lora" else f"runs/sd15/scale50k/{method}_seed42"
            manifest["jobs"][method] = {"gpu": gpu, "parent_run": str((ROOT/parent).resolve()),
                "output_dir": f"{run_root}/{method}_seed42", "result_dir":
                "results/sd15/fresh68287_swiftbrush_cfg4p5" if method == "swiftbrush_lora" else result_root}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    pointer = ROOT/"artifacts/launches/latest_sd15_continuation.txt"
    if pointer.exists():
        previous = Path(pointer.read_text().strip())/"process.json"
        if previous.exists() and process_matches(json.loads(previous.read_text())):
            raise RuntimeError("The continuation queue is already running")
        raise RuntimeError("A continuation plan already exists; resume that supervisor with --supervise")
    if shutil.disk_usage(ROOT).free < 25*2**30:
        raise RuntimeError("Fresh data and training need at least 25 GiB of available disk space")
    folder = ROOT/"artifacts/launches"/("sd15_continuation_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True)
    atomic_json(folder/"launch.json", manifest)
    (folder/"plans").mkdir()
    for method, job in manifest["jobs"].items():
        parent_config = Path(job["parent_run"])/"config.yaml"
        if parent_config.exists():
            config = continuation_config(load_config(parent_config), job["parent_run"],
                job["output_dir"], manifest["train_offset"], manifest["fresh_training_images"], args.updates)
            (folder/"plans"/f"{method}_seed42.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    study = load_config("configs/sd15_50k.yaml")
    study["data"].update(train_offset=50000, train_limit=68287, sampling="shuffled_epochs")
    study["train"]["updates_per_stage"] = args.updates
    study["eval"].update(steps=[1], comparison_steps=1, retain_generated_images=False)
    (folder/"study.yaml").write_text(yaml.safe_dump(study, sort_keys=False))
    with (folder/"output.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--supervise", str(folder)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    atomic_json(folder/"process.json", process_record(child.pid))
    pointer.write_text(str(folder)+"\n")
    run = ROOT/run_root
    run.mkdir(parents=True, exist_ok=True)
    for link, target in {"launch_status.json": "status.json", "launch_plan.json": "launch.json",
                         "launch.log": "output.log", "prepare.log": "prepare.log",
                         **{f"gpu{g}.log": f"gpu{g}.log" for g in GROUPS}}.items():
        (run/link).unlink(missing_ok=True)
        (run/link).symlink_to(folder/target)
    for method, job in manifest["jobs"].items():
        output = ROOT/job["output_dir"]
        output.mkdir(parents=True, exist_ok=True)
        for name, target in {"launch.log": folder/f"gpu{job['gpu']}.log",
                             "launch_plan.yaml": folder/"plans"/f"{method}_seed42.yaml"}.items():
            (output/name).unlink(missing_ok=True)
            (output/name).symlink_to(target)
    print(json.dumps({"pid": child.pid, "folder": str(folder), **manifest}, indent=2))


if __name__ == "__main__":
    main()
