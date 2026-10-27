#!/usr/bin/env python3
"""Prepare and launch the four controlled direct one-step DMD variants."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from few_step_diffusion.config import config_fingerprint, load_config, validate
from few_step_diffusion.continuation import continuation_config
from few_step_diffusion.dmd_diversity import apply_variant
from few_step_diffusion.runtime import atomic_json, experiment_lock, sha256
from launch_scale50k import process_matches, process_record

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["dmd_perceptual_diversity", "dmd_grouped", "dmd_perceptual", "dmd_diversity"]


def now():
    return datetime.now(timezone.utc).isoformat()


def free_gpu_gib(gpu):
    output = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free",
                                      "--format=csv,noheader,nounits"], text=True)
    return float(output.strip())/1024


def task(folder, action, method):
    manifest = json.loads((folder/"launch.json").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = manifest["gpu"]
    config = load_config(folder/"plans"/f"{method}.yaml")
    if action == "prepare":
        import torch
        from few_step_diffusion.diffusion import make_grid
        from few_step_diffusion.sd_data import caption_records
        from few_step_diffusion.sd_models import SDBundle
        from few_step_diffusion.teacher_pairs import TeacherPairs
        bundle = SDBundle(config, torch.device("cuda"))
        grid = make_grid(len(bundle.alphas), config["train"]["initial_steps"], bundle.device)
        with experiment_lock(manifest["pair_cache_dir"]):
            pairs = TeacherPairs(config, bundle, grid, manifest["pair_cache_dir"], caption_records(config))
        atomic_json(folder/"prepared_pairs.json", pairs.metadata)
        return
    if action == "diversity":
        from few_step_diffusion.diversity_evaluation import evaluate_diversity
        result = Path(manifest["result_dir"])/f"{method}_seed42_1steps.result.json"
        return evaluate_diversity(config, str(Path(config["output_dir"])/"stage_1"),
                                  Path(config["output_dir"])/"diversity_eval", result)
    if action == "teacher-diversity":
        from few_step_diffusion.diversity_evaluation import evaluate_diversity
        return evaluate_diversity(config, "teacher", folder/"teacher_diversity")
    raise ValueError(f"Unknown worker task: {action}")


def supervise(folder):
    manifest = json.loads((folder/"launch.json").read_text())
    state = {"pid": os.getpid(), "status": "running", "started_at": now(),
        "gpu": manifest["gpu"], "phase": "preparing_teacher_pairs",
        "jobs": {method: {"status": "queued", "phase": "waiting_for_completed_dmd_parent"} for method in ORDER}}

    def update(**values):
        state.update(values, updated_at=now())
        atomic_json(folder/"status.json", state)

    def job_update(method, **values):
        state["jobs"][method].update(values)
        update()

    def wait_for_memory(required, method=None):
        while free_gpu_gib(manifest["gpu"]) < required or shutil.disk_usage(ROOT).free < 8*2**30:
            update(phase="waiting_for_resources", required_free_gpu_gib=required)
            if method:
                job_update(method, status="queued", phase="waiting_for_resources")
            time.sleep(20)

    def command(args, phase, logfile, method=None):
        with (folder/logfile).open("ab", buffering=0) as log:
            child = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=manifest["gpu"],
                    PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
            update(phase=phase, child_pid=child.pid)
            if method:
                job_update(method, status="running", phase=phase, child_pid=child.pid)
            code = child.wait()
        update(child_pid=None)
        if code:
            raise RuntimeError(f"{phase} exited {code}; see {folder/logfile}")

    python = sys.executable
    native = [python, "-m", "few_step_diffusion.cli"]
    worker = [python, str(Path(__file__).resolve()), "--folder", str(folder)]
    update()
    wait_for_memory(8)
    command(worker+["--task", "prepare", "--method", ORDER[0]], "prepare_5000_grouped_teacher_pairs", "prepare.log")
    parent = Path(manifest["parent_run"])
    update(phase="waiting_for_completed_dmd_parent")
    while not (parent/"run.json").exists():
        prior = ROOT/"runs/sd15/scale50k/launch_status.json"
        if prior.exists():
            prior_state = json.loads(prior.read_text())
            if prior_state.get("workers", {}).get("3", {}).get("status") == "failed":
                raise RuntimeError("Parent DMD GPU worker failed; inspect the original launch status")
        time.sleep(20)
    completed = json.loads((parent/"run.json").read_text())
    if completed.get("status") != "complete" or config_fingerprint(completed["config"]) != manifest["parent_config_hash"]:
        raise ValueError("Completed parent differs from the planned DMD run")
    parent_digest = sha256(parent/"last.pt")
    if manifest.get("parent_checkpoint_sha256", parent_digest) != parent_digest:
        raise ValueError("Parent checkpoint changed since the study started")
    manifest["parent_checkpoint_sha256"] = parent_digest
    atomic_json(folder/"launch.json", manifest)
    for method in ORDER:
        plan = folder/"plans"/f"{method}.yaml"
        config = load_config(plan)
        config["continuation"]["parent_checkpoint_sha256"] = parent_digest
        plan.write_text(yaml.safe_dump(config, sort_keys=False))
    for method in ORDER:
        try:
            plan = folder/"plans"/f"{method}.yaml"
            config = load_config(plan)
            output = Path(config["output_dir"])
            wait_for_memory(manifest["min_free_gpu_gib"], method)
            with experiment_lock(output):
                if (output/"run.json").exists():
                    saved = json.loads((output/"run.json").read_text())
                    if saved["status"] != "complete" or config_fingerprint(saved["config"]) != config_fingerprint(config):
                        raise ValueError("Finished study run differs from the planned experiment")
                else:
                    args = native+["train", "--config", str(plan)]
                    if (output/"last.pt").exists():
                        args.append("--resume")
                    command(args, "training_one_step", f"{method}.log", method)
                result = Path(manifest["result_dir"])/f"{method}_seed42_1steps.result.json"
                if not result.exists():
                    command(native+["evaluate", "--config", str(plan), "--checkpoint", str(output/"stage_1"),
                        "--steps", "1", "--result-dir", manifest["result_dir"]],
                        "evaluating_5000_fid_is_clip", f"{method}.log", method)
                record = json.loads(result.read_text())
                if (record["config_hash"] != config_fingerprint(config) or
                        record["model_identity"] != sha256(output/"stage_1/adapter.safetensors")):
                    raise ValueError("Saved evaluation differs from the planned study model")
                record["evaluation_context"] = {"physical_gpu": manifest["gpu"], "concurrent_training": True,
                    "latency_comparable": False, "note": "Shared GPU: use image metrics, not timings, for method comparisons."}
                atomic_json(result, record)
                if "diversity_lpips_alex_mean" not in record:
                    command(worker+["--task", "diversity", "--method", method],
                            "evaluating_128_prompts_four_seeds", f"{method}.log", method)
            job_update(method, status="complete", phase="complete", child_pid=None, finished_at=now())
        except BaseException as exc:
            job_update(method, status="failed", phase="failed", error=str(exc), child_pid=None)
            raise
    wait_for_memory(8)
    command(worker+["--task", "teacher-diversity", "--method", ORDER[0]], "evaluate_teacher_diversity", "teacher.log")
    command([python, "scripts/build_dmd_report.py", "--results-dir", manifest["result_dir"],
             "--teacher-diversity", str(folder/"teacher_diversity/diversity.json")], "build_experiment_report", "report.log")
    update(status="complete", phase="complete", finished_at=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--updates", type=int, default=18000)
    parser.add_argument("--min-free-gpu-gib", type=float, default=26.)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--supervise", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--folder", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--task", choices=("prepare", "diversity", "teacher-diversity"), help=argparse.SUPPRESS)
    parser.add_argument("--method", default=ORDER[0], help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.task:
        return task(args.folder, args.task, args.method)
    if args.supervise:
        record = args.supervise/"process.json"
        if record.exists():
            previous = json.loads(record.read_text())
            if previous["pid"] != os.getpid() and process_matches(previous):
                raise RuntimeError("This DMD study supervisor is already running")
        atomic_json(record, process_record(os.getpid()))
        try:
            return supervise(args.supervise)
        except BaseException as exc:
            path = args.supervise/"status.json"
            status = json.loads(path.read_text()) if path.exists() else {}
            atomic_json(path, {**status, "status": "failed", "error": str(exc), "finished_at": now()})
            raise
    parent = ROOT/"runs/sd15/scale50k/dmd_seed42"
    parent_config = load_config(parent/"config.yaml")
    run_root, results = "runs/sd15/dmd_diversity_study", "results/sd15/dmd_diversity_study"
    cache = str((ROOT/run_root/"shared_teacher_pairs").resolve())
    manifest = {"gpu": args.gpu, "methods": ORDER, "parent_run": str(parent),
        "parent_config_hash": config_fingerprint(parent_config), "additional_updates": args.updates,
        "fresh_training_images": 68287, "train_offset": 50000, "effective_batch_size": 16,
        "student_lora_rank": 32, "sampling_steps": 1, "nfe": 1, "teacher_guidance_scale": 7.5,
        "teacher_pair_images": 5000, "teacher_pair_unique_prompts": 2500,
        "teacher_pair_seeds_per_prompt": 2, "pair_cache_dir": cache, "result_dir": results,
        "min_free_gpu_gib": args.min_free_gpu_gib}
    plans = {}
    for method in ORDER:
        config = continuation_config(parent_config, parent, f"{run_root}/{method}_seed42", updates=args.updates)
        plans[method] = apply_variant(config, method, cache)
        validate(plans[method])
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    pointer = ROOT/"artifacts/launches/latest_dmd_diversity.txt"
    if pointer.exists():
        raise RuntimeError("A DMD diversity plan already exists; inspect it or resume with --supervise")
    if shutil.disk_usage(ROOT).free < 10*2**30:
        raise RuntimeError("DMD study needs at least 10 GiB available disk space")
    folder = ROOT/"artifacts/launches"/("dmd_diversity_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    (folder/"plans").mkdir(parents=True)
    atomic_json(folder/"launch.json", manifest)
    for method, config in plans.items():
        (folder/"plans"/f"{method}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    with (folder/"output.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--supervise", str(folder)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    atomic_json(folder/"process.json", process_record(child.pid))
    pointer.write_text(str(folder)+"\n")
    root = ROOT/run_root
    root.mkdir(parents=True, exist_ok=True)
    for name, target in {"launch_status.json": "status.json", "launch_plan.json": "launch.json",
                         "launch.log": "output.log", "prepare.log": "prepare.log"}.items():
        (root/name).symlink_to(folder/target)
    for method in ORDER:
        output = root/f"{method}_seed42"
        output.mkdir(parents=True, exist_ok=True)
        (output/"launch.log").symlink_to(folder/f"{method}.log")
        (output/"launch_plan.yaml").symlink_to(folder/"plans"/f"{method}.yaml")
    print(json.dumps({"pid": child.pid, "folder": str(folder), **manifest}, indent=2))


if __name__ == "__main__":
    main()
