#!/usr/bin/env python3
"""Durable resource-aware DMD2/f-distill queue. Never stops other GPU processes."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_scale50k import process_matches, process_record
from few_step_diffusion.config import config_fingerprint, load_config
from few_step_diffusion.runtime import atomic_json, sha256
from few_step_diffusion.checkpoint_storage import checkpoint_free_gib, prepare_output_storage

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/"runs/sd15/recent36k"
PLAN = ROOT/"configs/recent_baselines_36k.json"
METHODS = ("dmd2", "f_distill")
SOURCE_FILES = ("src/few_step_diffusion/recent_objectives.py", "src/few_step_diffusion/sd_recent_training.py",
                "src/few_step_diffusion/sd_models.py", "src/few_step_diffusion/sd_evaluation.py",
                "src/few_step_diffusion/sd_data.py", "src/few_step_diffusion/checkpoint_storage.py",
                "src/few_step_diffusion/recent_performance.py",
                "src/few_step_diffusion/vae_cache.py",
                "scripts/accelerate_recent_baselines.py",
                "scripts/cache_recent_vae.py",
                "scripts/launch_recent_baselines.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def make_plan():
    from few_step_diffusion.recent_objectives import SOURCE_COMMIT
    jobs = []
    for method in METHODS:
        path = f"configs/sd15_{method}_36k.yaml"
        config = load_config(ROOT/path)
        jobs.append({"method": method, "config": path, "config_hash": config_fingerprint(config),
                     "run_dir": config["output_dir"], "result_dir": "results/sd15/recent36k"})
    plan = {"policy": "recent36k-resource-gated-v1", "source_commit": SOURCE_COMMIT,
            "gpus": [0, 1, 2], "max_parallel": 2, "minimum_free_gpu_gib": 64,
            "training_memory_cap_gib": 48, "minimum_free_disk_gib": 70,
            "student_updates": 36000, "auxiliary_updates": 324000, "jobs": jobs,
            "source_hashes": {name: sha256(ROOT/name) for name in SOURCE_FILES}}
    if PLAN.exists() and read(PLAN) != plan:
        raise ValueError("Existing queue plan differs from code/configs; refusing an unreviewed replacement")
    atomic_json(PLAN, plan)
    return plan


def verify_plan(plan):
    for path, digest in plan["source_hashes"].items():
        if sha256(ROOT/path) != digest:
            raise ValueError(f"Queued implementation changed: {path}")
    for job in plan["jobs"]:
        if config_fingerprint(load_config(ROOT/job["config"])) != job["config_hash"]:
            raise ValueError(f"Queued config changed: {job['config']}")


def gpu_free():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                      "--format=csv,noheader,nounits"], text=True)
    return {int(line.split(",")[0]): float(line.split(",")[1])/1024 for line in output.splitlines()}


def eligible_gpus(plan, available, occupied, disk_free):
    if disk_free < plan["minimum_free_disk_gib"]:
        return []
    return [g for g in plan["gpus"] if g not in occupied and
            available.get(g, 0) >= plan["minimum_free_gpu_gib"]]


def live_worker(state):
    # If the supervisor/worker died but its child survived, wait for that exact
    # process incarnation too. Never spawn a duplicate or act on a reused PID.
    return any(process_matches(state[k]) for k in ("worker", "child") if state.get(k))


def preflight(job):
    """Two actual updates, checkpoint/export/preview, then resume the same run."""
    import torch
    import yaml
    from few_step_diffusion.runtime import device_for, provenance, seed_everything
    from few_step_diffusion.sd_recent_training import train_recent
    from few_step_diffusion.tracking import TrainingTracker
    config = load_config(ROOT/job["config"])
    output = ROOT/job["run_dir"]
    seed_everything(config["seed"])
    device = device_for(config)
    checkpoint = output/"last.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint.exists() else None
    if saved and config_fingerprint(saved["config"]) != job["config_hash"]:
        raise ValueError("Preflight checkpoint has a different config")
    if saved and saved["global_step"] >= 2:
        raise ValueError("Missing preflight approval for a nontrivial existing run")
    (output/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    env = provenance(config, device)
    atomic_json(output/"environment.json", env)
    with TrainingTracker(config, env, output, resume=saved is not None) as tracker:
        train_recent(config, saved, device, output, checkpoint, env, tracker,
                     stop_after=2-(saved["global_step"] if saved else 0))


def worker(plan, method, gpu):
    job = next(j for j in plan["jobs"] if j["method"] == method)
    output = ROOT/job["run_dir"]
    output.mkdir(parents=True, exist_ok=True)
    prepare_output_storage(output)
    with (output/".queue-worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"method": method, "gpu": gpu, "worker": process_record(os.getpid()),
                 "status": "starting", "started_at": now()}

        def update(**values):
            state.update(values, updated_at=now())
            atomic_json(output/"queue.json", state)

        def command(args, phase, memory_cap=48):
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                FSD_MAX_GPU_MEMORY_GIB=str(memory_cap), PYTHONUNBUFFERED="1",
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
            execution = read(output/"execution.json") if phase in ("preflight", "train") else {}
            if execution:
                if execution.get("config_hash") != job["config_hash"]:
                    raise ValueError("Execution settings belong to another experiment")
                environment.update(FSD_MAX_GPU_MEMORY_GIB=str(execution.get("training_memory_gib", memory_cap)),
                    FSD_RECENT_OPTIMIZATIONS=",".join(execution["optimizations"]),
                    FSD_RECENT_MICRO_FACTOR=str(execution["micro_factor"]))
                if execution.get("posterior_cache"):
                    environment["FSD_RECENT_VAE_CACHE"] = execution["posterior_cache"]
                if "vae_helper" in execution["optimizations"]:
                    helper = execution["helper_gpu"]
                    if helper == gpu:
                        raise ValueError("The VAE helper must differ from the training GPU")
                    environment.update(CUDA_VISIBLE_DEVICES=f"{gpu},{helper}",
                        FSD_VAE_HELPER_DEVICE="cuda:1",
                        FSD_VAE_HELPER_MEMORY_GIB=str(execution["helper_memory_gib"]))
            with (output/f"{phase}.log").open("ab", buffering=0) as log:
                child = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    env=environment)
                update(status="running", phase=phase, child=process_record(child.pid))
                code = child.wait()
                update(child=None, exit_code=code)
                if code:
                    raise RuntimeError(f"{phase} exited {code}; see {output/phase}.log")

        try:
            verify_plan(plan)
            complete = read(output/"run.json")
            if complete.get("status") != "complete":
                command(["scripts/launch_recent_baselines.py", "prepare", "--method", method], "prepare")
                if not (output/"preflight.json").exists():
                    command(["scripts/launch_recent_baselines.py", "preflight", "--method", method], "preflight")
                checked = read(output/"preflight.json")
                if checked.get("status") != "passed" or checked.get("global_step") != 2:
                    raise ValueError("GPU checkpoint/resume preflight did not pass")
                command(["-m", "few_step_diffusion.cli", "train", "--config", job["config"], "--resume"], "train")
            result = ROOT/job["result_dir"]/f"{method}_seed42_1steps.result.json"
            if not result.exists():
                command(["-m", "few_step_diffusion.cli", "evaluate", "--config", job["config"],
                         "--checkpoint", str(output/"stage_1"), "--steps", "1",
                         "--result-dir", job["result_dir"]], "evaluate", memory_cap=16)
            from few_step_diffusion.recent_reporting import validate_result
            validate_result(ROOT, result, job)
            update(status="evaluated", phase="report", result_file=str(result.relative_to(ROOT)))
            command(["scripts/build_assignment2_report.py", "--group-id", "11", "--package"], "report")
            update(status="complete", phase="complete", completed_at=now())
        except Exception as exc:
            update(status="failed", error=str(exc))
            raise


def supervise(plan, retry_failed=False):
    FOLDER.mkdir(parents=True, exist_ok=True)
    with (FOLDER/".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children, retried = {}, set()
        while True:
            retry_request = FOLDER/"retry_failed.json"
            if retry_request.exists():
                retry_failed = True
                retried.clear()
                retry_request.unlink()
            for key, process in list(children.items()):
                if process.poll() is not None:
                    del children[key]
            states = {j["method"]: read(ROOT/j["run_dir"]/"queue.json",
                      {"method": j["method"], "status": "queued"}) for j in plan["jobs"]}
            active = {m: s for m, s in states.items() if live_worker(s)}
            available = gpu_free()
            disk = checkpoint_free_gib(ROOT)
            occupied = {s["gpu"] for s in active.values()}
            candidates = eligible_gpus(plan, available, occupied, disk)
            verify_plan(plan)
            for job in plan["jobs"]:
                method, previous = job["method"], states[job["method"]]
                failed = previous.get("status") == "failed"
                if (method in active or previous.get("status") == "complete" or
                        (failed and (not retry_failed or method in retried))):
                    continue
                if not candidates or len(active) >= plan["max_parallel"]:
                    break
                # Refresh disk before every launch; reserve 35 GiB per active
                # job for atomic full-scorer checkpoints and final evaluation.
                if disk-35*len(active) < plan["minimum_free_disk_gib"]:
                    break
                gpu = candidates.pop(0)
                out = ROOT/job["run_dir"]
                out.mkdir(parents=True, exist_ok=True)
                with (out/"launch.log").open("ab", buffering=0) as log:
                    child = subprocess.Popen([sys.executable, "scripts/launch_recent_baselines.py", "worker",
                        "--method", method, "--gpu", str(gpu)], cwd=ROOT,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record = {"worker": process_record(child.pid), "gpu": gpu, "status": "starting"}
                atomic_json(out/"queue.json", record)
                active[method] = record
                children[method] = child
                retried.add(method)
            states = {j["method"]: read(ROOT/j["run_dir"]/"queue.json",
                      {"method": j["method"], "status": "queued"}) for j in plan["jobs"]}
            finished = all(s.get("status") == "complete" for s in states.values())
            status = "complete" if finished else "running" if active else "waiting_for_resources"
            if not active and any(s.get("status") == "failed" for s in states.values()):
                status = "needs_attention"
            atomic_json(FOLDER/"status.json", {"status": status, "updated_at": now(),
                "supervisor": process_record(os.getpid()), "jobs": states,
                "free_gpu_gib": available, "free_disk_gib": round(disk, 2),
                "required_free_gpu_gib": plan["minimum_free_gpu_gib"],
                "required_free_disk_gib": plan["minimum_free_disk_gib"],
                "waiting_reasons": (["Insufficient disk space"] if disk < plan["minimum_free_disk_gib"] else [])+
                    (["No eligible GPU has enough free memory"] if not any(
                        available.get(g, 0) >= plan["minimum_free_gpu_gib"] for g in plan["gpus"]) else []),
                "student_updates_per_method": 36000, "auxiliary_updates_per_method": 324000})
            if finished:
                return
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status", "supervise", "worker", "preflight", "prepare"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "status":
        state = read(FOLDER/"status.json", {"status": "not_launched"})
        state["supervisor_alive"] = bool(state.get("supervisor") and process_matches(state["supervisor"]))
        cache_process = read(FOLDER/"posterior_cache_process.json")
        cache_status = read(FOLDER/"posterior_cache_status.json")
        cache_alive = bool(cache_process.get("worker") and process_matches(cache_process["worker"]))
        if cache_status:
            state["posterior_cache"] = {**cache_status, "worker_alive": cache_alive}
        if cache_alive and cache_status.get("status") == "building":
            state["status"] = "preparing_posterior_cache"
            for method, job in state.get("jobs", {}).items():
                if job.get("status") != "complete" and not live_worker(job):
                    job.update(status="waiting_for_cache", phase="cache_prepare",
                               resume_step=cache_process.get("resume_steps", {}).get(method))
        print(json.dumps(state, indent=2))
        return
    if args.action == "launch":
        FOLDER.mkdir(parents=True, exist_ok=True)
        with (FOLDER/".launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prior = read(FOLDER/"launch.json")
            if prior.get("supervisor") and process_matches(prior["supervisor"]):
                if args.retry_failed:
                    atomic_json(FOLDER/"retry_failed.json", {"requested_at": now()})
                    print("Requested one retry of failed jobs by the existing supervisor")
                print(f"Queue already running: PID {prior['supervisor']['pid']}")
                return
            plan = make_plan()
            with (FOLDER/"queue.log").open("ab", buffering=0) as log:
                child = subprocess.Popen([sys.executable, "scripts/launch_recent_baselines.py", "supervise",
                    *(["--retry-failed"] if args.retry_failed else [])], cwd=ROOT, start_new_session=True,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(FOLDER/"launch.json", {"supervisor": process_record(child.pid), "started_at": now()})
            print(f"Detached queue PID {child.pid}; status: {FOLDER/'status.json'}")
        return
    plan = read(PLAN)
    verify_plan(plan)
    if args.action == "supervise":
        supervise(plan, args.retry_failed)
    elif args.action == "worker":
        worker(plan, args.method, args.gpu)
    elif args.action == "prepare":
        from few_step_diffusion.runtime import experiment_lock
        from few_step_diffusion.sd_data import prepare_sd
        config = load_config(ROOT/next(j for j in plan["jobs"] if j["method"] == args.method)["config"])
        with experiment_lock(ROOT/config["data"]["root"]/"recent_preparation"):
            prepare_sd(config, metrics=True)
    else:
        preflight(next(j for j in plan["jobs"] if j["method"] == args.method))


if __name__ == "__main__":
    main()
