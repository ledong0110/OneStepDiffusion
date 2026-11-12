#!/usr/bin/env python3
"""Prepare teacher targets and run fresh 36k-update proposal/control jobs detached."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from few_step_diffusion.config import config_fingerprint, load_config
from few_step_diffusion.runtime import atomic_json, experiment_lock, sha256
from few_step_diffusion.checkpoint_storage import prepare_output_storage
from launch_scale50k import process_matches, process_record
from launch_dmd_conflict import started_process_record
from launch_recent_baselines import gpu_free, now, read

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/"runs/sd15/independent36k"
PLAN = ROOT/"configs/independent_response_36k.json"
METHODS = ("response_scratch", "dmd_lpips_scratch")
FILES = ("src/few_step_diffusion/independent_response.py", "src/few_step_diffusion/sd_training.py",
         "src/few_step_diffusion/dmd_response.py", "src/few_step_diffusion/teacher_pairs.py",
         "src/few_step_diffusion/dmd_diversity.py", "src/few_step_diffusion/training.py",
         "src/few_step_diffusion/checkpoint_storage.py",
         "scripts/launch_independent_job.py",
         "scripts/launch_independent_response.py")


def plan():
    jobs = []
    for method in METHODS:
        path = f"configs/sd15_{method}_36k.yaml"
        cfg = load_config(ROOT/path)
        jobs.append({"method": method, "config": path, "run_dir": cfg["output_dir"],
                     "config_hash": config_fingerprint(cfg)})
    value = {"policy": "independent-sd15-response-36k-v1", "gpus": [0, 1, 2], "jobs": jobs,
             "student_updates": 36000, "initial_distillation_updates": 0,
             "source_hashes": {p: sha256(ROOT/p) for p in FILES},
             "phase_limits": {"prepare": {"gpu_gib": 9, "cap_gib": 8, "disk_gib": 2},
                 "preflight": {"gpu_gib": 16, "cap_gib": 15, "disk_gib": 6},
                 "train": {"gpu_gib": 16, "cap_gib": 15, "disk_gib": 6},
                 "evaluate": {"gpu_gib": 16, "cap_gib": 15, "disk_gib": 3}}}
    if PLAN.exists() and read(PLAN) != value:
        raise ValueError("Independent training plan changed; refusing to replace its identity")
    atomic_json(PLAN, value)
    return value


def verify(value):
    for path, digest in value["source_hashes"].items():
        if sha256(ROOT/path) != digest:
            raise ValueError(f"Independent training implementation changed: {path}")
    for job in value["jobs"]:
        if config_fingerprint(load_config(ROOT/job["config"])) != job["config_hash"]:
            raise ValueError("Independent training config changed")


def available_gpu(value, phase):
    limit = value["phase_limits"][phase]
    memory = gpu_free()
    disk = shutil.disk_usage(ROOT).free/2**30
    candidates = [g for g in value["gpus"] if memory.get(g, 0) >= limit["gpu_gib"]]
    return (candidates[0] if candidates and disk >= limit["disk_gib"] else None), memory, disk


def prepare(config):
    from few_step_diffusion.diffusion import make_grid
    from few_step_diffusion.independent_response import PhasedCaptions, PhasedPairs
    from few_step_diffusion.runtime import device_for, seed_everything
    from few_step_diffusion.sd_models import SDBundle
    seed_everything(config["seed"])
    data = PhasedCaptions(config)
    bundle = SDBundle(config, device_for(config))
    pairs = PhasedPairs(config, bundle, make_grid(len(bundle.alphas), config["train"]["initial_steps"], bundle.device), data)
    for step in (0, 18000):
        pairs.select(step)
        print(f"Teacher-only bank phase {pairs.phase+1}: {pairs.metadata['weights_sha256']}", flush=True)
    atomic_json(FOLDER/"teacher_targets_ready.json", {"status": "complete",
        "banks": [pairs.bank_metadata[i] for i in range(2)], "student_updates": 0})


def evaluate(config, output):
    from few_step_diffusion.evaluation import evaluate as evaluate_model
    from few_step_diffusion.independent_response import validate_result
    method = config["name"]
    pending = ROOT/"results/sd15/independent36k/_pending"/method
    path = pending/f"{method}_seed42_1steps.result.json"
    if not path.exists():
        evaluate_model(config, str(output/"stage_1"), 1, str(pending), False)
    result = read(path)
    metadata = read(output/"stage_1/metadata.json")
    for key in ("independent_training", "initialization", "independent_initialization_sha256",
                "training_data_phases", "teacher_pair_phases", "cumulative_teacher_pair_count", "cumulative_teacher_pair_nfe"):
        result[key] = metadata[key]
    if result["model_identity"] != sha256(output/"stage_1/adapter.safetensors"):
        raise ValueError("Independent evaluation uses another checkpoint")
    validate_result(result)
    if result["num_samples"] != 5000 or result["num_real"] != 5000:
        raise ValueError("Full independent evaluation requires all 5,000 samples")
    atomic_json(ROOT/"results/sd15/independent36k"/path.name, result)


def supervise(value):
    state_path = FOLDER/"status.json"
    state = read(state_path, {"jobs": {}})
    state.update(status="starting", supervisor=process_record(os.getpid()))
    for job in value["jobs"]:
        state["jobs"].get(job["method"], {}).pop("student_updates", None)
        state["jobs"].setdefault(job["method"], {"status": "queued", "phase": "pending",
                                               "initial_student_updates": 0})

    def update(method=None, **values):
        target = state if method is None else state["jobs"].setdefault(method, {})
        target.update(values)
        state["updated_at"] = now()
        atomic_json(state_path, state)

    def command(job, phase):
        method = job["method"]
        while True:
            verify(value)
            gpu, memory, disk = available_gpu(value, phase)
            if gpu is not None:
                break
            update(method, status="queued", phase=phase, waiting_for="GPU memory or disk capacity",
                   free_gpu_gib=memory, free_disk_gib=round(disk, 2), required=value["phase_limits"][phase])
            update(status="waiting_for_resources")
            time.sleep(30)
        output = ROOT/job["run_dir"]
        argv = [sys.executable, "scripts/launch_independent_response.py", phase, "--method", method]
        with (output/f"{phase}.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                    FSD_MAX_GPU_MEMORY_GIB=str(value["phase_limits"][phase]["cap_gib"]),
                    OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
            record = started_process_record(child, argv)
            update(method, status="running", phase=phase, gpu=gpu, child=record, waiting_for=None)
            update(status="running")
            code = child.wait()
        update(method, child=None, exit_code=code)
        if code:
            raise RuntimeError(f"{method} {phase} exited {code}; inspect {output/phase}.log")

    for job in value["jobs"]:
        method = job["method"]
        output = ROOT/job["run_dir"]
        output.mkdir(parents=True, exist_ok=True)
        prepare_output_storage(output)
        previous = state["jobs"].get(method, {}).get("child")
        if previous and process_matches(previous):
            update(method, status="running", phase="adopting_existing_child")
            update(status="running")
            while process_matches(previous):
                time.sleep(20)
        try:
            with experiment_lock(output):
                if not (FOLDER/"teacher_targets_ready.json").exists():
                    command(job, "prepare")
                if not (output/"preflight.json").exists():
                    command(job, "preflight")
                preflight = read(output/"preflight.json")
                if preflight.get("status") != "passed" or preflight.get("global_step") != 2:
                    raise ValueError("Independent full-model preflight must complete two actual updates")
                if not (output/"run.json").exists():
                    command(job, "train")
                completed = read(output/"run.json")
                if completed.get("status") != "complete" or completed.get("global_step") != 36000:
                    raise ValueError("Independent training is incomplete")
                result = ROOT/f"results/sd15/independent36k/{method}_seed42_1steps.result.json"
                if not result.exists():
                    command(job, "evaluate")
                from few_step_diffusion.independent_response import validate_result
                validate_result(read(result))
            update(method, status="evaluated", phase="report", result_file=str(result.relative_to(ROOT)))
            with (output/"report.log").open("a") as log:
                subprocess.run([sys.executable, "scripts/build_assignment2_report.py", "--group-id", "11", "--package"],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            update(method, status="complete", phase="complete", finished_at=now())
        except Exception as exc:
            update(method, status="failed", error=str(exc))
            update(status="needs_attention")
            raise
    update(status="complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "supervise", "status", "prepare", "preflight", "train", "evaluate"))
    parser.add_argument("--method", choices=METHODS, default=METHODS[0])
    args = parser.parse_args()
    os.chdir(ROOT)
    FOLDER.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        state = read(FOLDER/"status.json", {"status": "not_launched"})
        state["supervisor_alive"] = bool(state.get("supervisor") and process_matches(state["supervisor"]))
        for method, item in state.get("jobs", {}).items():
            parallel = read(FOLDER/f"{method}_seed42/parallel.json")
            if parallel:
                item.update(parallel)
                item["parallel_worker_alive"] = any(process_matches(parallel[k])
                    for k in ("worker", "child") if parallel.get(k))
            log = FOLDER/f"{method}_seed42/train.jsonl"
            latest = next((json.loads(line) for line in reversed(log.read_text().splitlines()) if line.strip()), {}) if log.exists() else {}
            item["last_logged_student_updates"] = latest.get("global_step", 0)
            item["target_student_updates"] = 36000
        print(json.dumps(state, indent=2))
        return
    if args.action == "launch":
        with (FOLDER/".launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            record = read(FOLDER/"process.json")
            if record and process_matches(record):
                print(f"Independent queue already active: PID {record['pid']}")
                return
            plan()
            argv = [sys.executable, "scripts/launch_independent_response.py", "supervise"]
            with (FOLDER/"supervisor.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
            record = started_process_record(child, argv)
            if record is None:
                raise RuntimeError("Independent supervisor failed at startup")
            atomic_json(FOLDER/"process.json", record)
            print(f"Independent queue PID {child.pid}; {FOLDER/'status.json'}")
        return
    value = read(PLAN)
    verify(value)
    if args.action == "supervise":
        with experiment_lock(FOLDER/"supervisor"):
            supervise(value)
        return
    job = next(j for j in value["jobs"] if j["method"] == args.method)
    config, output = load_config(ROOT/job["config"]), ROOT/job["run_dir"]
    if args.action == "prepare":
        prepare(config)
    elif args.action == "evaluate":
        evaluate(config, output)
    else:
        from few_step_diffusion.training import train
        if args.action == "preflight":
            import torch
            resume = (output/"last.pt").exists()
            saved = torch.load(output/"last.pt", map_location="cpu", weights_only=False) if resume else None
            step = saved["global_step"] if saved else 0
            if step >= 2:
                raise ValueError("Existing checkpoint has no matching preflight completion record")
            train(config, resume=resume, stop_after=2-step)
        else:
            train(config, resume=True)


if __name__ == "__main__":
    main()
