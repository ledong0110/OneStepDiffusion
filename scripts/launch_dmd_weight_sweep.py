#!/usr/bin/env python3
"""Detached, resumable diversity-weight sweep with full evaluation and report refresh."""

import argparse
from concurrent.futures import ThreadPoolExecutor
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

from few_step_diffusion.config import config_fingerprint, load_config, validate
from few_step_diffusion.continuation import continuation_config
from few_step_diffusion.dmd_diversity import apply_weight_sweep
from few_step_diffusion.runtime import atomic_json, experiment_lock, sha256
from launch_scale50k import process_matches, process_record

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/"configs/dmd_weight_sweep.json"
RUNS = ROOT/"runs/sd15/dmd_weight_sweep"
METHOD = "dmd_perceptual_diversity"
WEIGHTS = (.01, .3, .5)


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(gpus, max_parallel, min_free_gib):
    parent = ROOT/"runs/sd15/scale50k/dmd_seed42"
    cache = ROOT/"runs/sd15/dmd_diversity_study/shared_teacher_pairs"
    parent_config = load_config(parent/"config.yaml")
    saved = json.loads((parent/"run.json").read_text())
    if saved.get("status") != "complete" or saved["global_step"] != 18000:
        raise ValueError("Sweep requires the completed shared 18k DMD initialization")
    if config_fingerprint(saved["config"]) != config_fingerprint(parent_config):
        raise ValueError("Parent run/config mismatch")
    parent_hash = sha256(parent/"last.pt")
    pairs = json.loads((cache/"teacher_pairs.json").read_text())
    if sha256(cache/"teacher_pairs.safetensors") != pairs["weights_sha256"]:
        raise ValueError("Shared teacher-pair bank failed integrity verification")
    jobs = []
    for weight in WEIGHTS:
        key = "lambda_"+str(weight).replace(".", "p")
        output = RUNS/key
        config = continuation_config(parent_config, parent, str(output.relative_to(ROOT)))
        config = apply_weight_sweep(config, weight, cache)
        config["continuation"]["parent_checkpoint_sha256"] = parent_hash
        validate(config)
        plan = ROOT/"configs/generated/dmd_weight_sweep"/f"{key}.yaml"
        if plan.exists() and config_fingerprint(load_config(plan)) != config_fingerprint(config):
            raise ValueError(f"Refusing to replace a different coefficient plan: {plan}")
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(yaml.safe_dump(config, sort_keys=False))
        jobs.append({"key": key, "weight": weight, "config": str(plan.relative_to(ROOT)),
                     "run_dir": str(output.relative_to(ROOT)),
                     "result_dir": f"results/sd15/dmd_weight_sweep/{key}",
                     "config_hash": config_fingerprint(config)})
    manifest = {"policy": "dmd-diversity-weight-v1", "weights": list(WEIGHTS),
        "regression_weight": .25, "regression_type": "lpips_vgg", "seed": 42,
        "cumulative_training_updates": 36000, "loss_specific_updates": 18000,
        "parent_checkpoint_sha256": parent_hash, "teacher_pair_weights_sha256": pairs["weights_sha256"],
        "gpus": gpus, "max_parallel": max_parallel, "min_free_gpu_gib": min_free_gib, "jobs": jobs}
    if PLAN.exists():
        previous = json.loads(PLAN.read_text())
        if previous != manifest:
            raise ValueError("Existing coefficient plan differs; resume it rather than replacing it")
    atomic_json(PLAN, manifest)
    RUNS.mkdir(parents=True, exist_ok=True)
    return manifest


def gpu_free():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                      "--format=csv,noheader,nounits"], text=True)
    return {int(row.split(",")[0]): float(row.split(",")[1])/1024
            for row in output.strip().splitlines()}


def supervise():
    manifest = json.loads(PLAN.read_text())
    status_path = RUNS/"status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {"jobs": {}}
    state.update(pid=os.getpid(), status="running", started_at=now())
    lock = threading.RLock()
    reserved = set()

    def update(key=None, **values):
        with lock:
            if key:
                state["jobs"].setdefault(key, {}).update(values)
            else:
                state.update(values)
            state["updated_at"] = now()
            atomic_json(status_path, state)

    def acquire_gpu(key):
        while True:
            with lock:
                free = gpu_free()
                candidates = [g for g in manifest["gpus"] if g not in reserved and
                              free.get(g, 0) >= manifest["min_free_gpu_gib"]]
                if candidates and shutil.disk_usage(ROOT).free >= 8*2**30:
                    gpu = max(candidates, key=lambda g: free[g])
                    reserved.add(gpu)
                    update(key, status="running", gpu=gpu)
                    return gpu
                update(key, status="queued", phase="waiting_for_memory", free_gpu_gib=free)
            time.sleep(20)

    def refresh_report():
        with experiment_lock(ROOT/"report/build/assignment2_refresh"):
            with (RUNS/"report.log").open("ab", buffering=0) as log:
                code = subprocess.call([sys.executable, "scripts/build_assignment2_report.py", "--package"],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            update(report_refresh="complete" if code == 0 else "failed", report_exit_code=code)

    def worker(job):
        key = job["key"]
        output, plan = ROOT/job["run_dir"], ROOT/job["config"]
        output.mkdir(parents=True, exist_ok=True)
        config = load_config(plan)
        if config_fingerprint(config) != job["config_hash"]:
            raise ValueError("Sweep configuration changed after scheduling")
        # A restarted supervisor waits for an already-running child before resuming.
        previous = state["jobs"].get(key, {}).get("child")
        if previous and process_matches(previous):
            update(key, status="running", phase="waiting_for_existing_worker")
            while process_matches(previous):
                time.sleep(20)
        update(key, status="queued", phase="waiting_for_gpu", child=None)
        gpu = acquire_gpu(key)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

        def run(args, phase):
            with (output/"launch.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                update(key, phase=phase, child=process_record(child.pid))
                code = child.wait()
            update(key, child=None)
            if code:
                raise RuntimeError(f"{phase} exited {code}; see {output/'launch.log'}")

        try:
            with experiment_lock(output):
                native = [sys.executable, "-m", "few_step_diffusion.cli"]
                if (output/"run.json").exists():
                    completed = json.loads((output/"run.json").read_text())
                    if (completed.get("status") != "complete" or
                            config_fingerprint(completed["config"]) != job["config_hash"]):
                        raise ValueError("Finished sweep run differs from the scheduled recipe")
                else:
                    args = native+["train", "--config", str(plan)]
                    if (output/"last.pt").exists():
                        args.append("--resume")
                    run(args, "training")
                result = ROOT/job["result_dir"]/f"{METHOD}_seed42_1steps.result.json"
                if not result.exists():
                    run(native+["evaluate", "--config", str(plan), "--checkpoint", str(output/"stage_1"),
                        "--steps", "1", "--result-dir", str(result.parent)], "full_5000_image_evaluation")
                from build_active_report import load_result
                record = load_result(ROOT, result)
                if record["config_hash"] != job["config_hash"]:
                    raise ValueError("Sweep evaluation/config mismatch")
                if "diversity_lpips_alex_mean" not in record:
                    run([sys.executable, str(Path(__file__).resolve()), "--diversity", key],
                        "diversity_128_prompts_four_seeds")
                update(key, status="complete", phase="complete", finished_at=now())
            refresh_report()
        except Exception as exc:
            update(key, status="failed", phase="failed", error=str(exc), finished_at=now())
        finally:
            with lock:
                reserved.discard(gpu)

    for job in manifest["jobs"]:
        state["jobs"].setdefault(job["key"], {"status": "queued", "weight": job["weight"]})
    update()
    with ThreadPoolExecutor(max_workers=manifest["max_parallel"]) as pool:
        futures = [pool.submit(worker, job) for job in manifest["jobs"]]
        for future in futures:
            future.result()
    update(status="complete" if all(j["status"] == "complete" for j in state["jobs"].values())
           else "failed", finished_at=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=26)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--diversity", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.diversity:
        from few_step_diffusion.diversity_evaluation import evaluate_diversity
        job = next(j for j in json.loads(PLAN.read_text())["jobs"] if j["key"] == args.diversity)
        config = load_config(ROOT/job["config"])
        output = ROOT/job["run_dir"]
        return evaluate_diversity(config, str(output/"stage_1"), output/"diversity_eval",
            ROOT/job["result_dir"]/f"{METHOD}_seed42_1steps.result.json")
    if args.supervise:
        with experiment_lock(RUNS/"supervisor"):
            atomic_json(RUNS/"process.json", process_record(os.getpid()))
            return supervise()
    if (RUNS/"process.json").exists() and process_matches(json.loads((RUNS/"process.json").read_text())):
        print(f"Sweep already running: {RUNS/'status.json'}")
        return
    if args.max_parallel < 1 or args.min_free_gib < 26 or not args.gpus:
        raise ValueError("Use a positive worker count and at least 26 GiB free memory per new run")
    manifest = prepare(args.gpus, args.max_parallel, args.min_free_gib)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    with (RUNS/"supervisor.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--supervise"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    atomic_json(RUNS/"process.json", process_record(child.pid))
    print(f"Sweep supervisor PID {child.pid}; status: {RUNS/'status.json'}")


if __name__ == "__main__":
    main()
