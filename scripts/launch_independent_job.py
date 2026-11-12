#!/usr/bin/env python3
"""Launch an independent 36k job alongside the serial queue on a selected GPU."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from few_step_diffusion.checkpoint_storage import checkpoint_free_gib, prepare_output_storage
from few_step_diffusion.runtime import atomic_json, experiment_lock
import launch_independent_response as queue
from launch_dmd_conflict import started_process_record
from launch_scale50k import process_matches, process_record

ROOT = Path(__file__).resolve().parents[1]


def enough_memory(free_gib, cap_gib):
    return free_gib >= cap_gib+1


def live(state):
    return any(process_matches(state[k]) for k in ("worker", "child") if state.get(k))


def worker(value, job, gpu, memory_cap):
    output = ROOT/job["run_dir"]
    state = {"method": job["method"], "gpu": gpu, "worker": process_record(os.getpid()),
             "status": "starting", "started_at": queue.now(), "memory_cap_gib": memory_cap,
             "target_student_updates": 36000, "config_hash": job["config_hash"]}

    def update(**values):
        state.update(values, updated_at=queue.now())
        atomic_json(output/"parallel.json", state)

    def command(phase):
        cap = min(memory_cap, 8) if phase == "evaluate" else memory_cap
        while True:
            queue.verify(value)
            memory = queue.gpu_free()
            if (enough_memory(memory.get(gpu, 0), cap) and
                    checkpoint_free_gib(ROOT) >= 6 and shutil.disk_usage(ROOT).free/2**30 >= .5):
                break
            update(status="queued", phase=phase, free_gpu_gib=memory,
                   waiting_for="GPU memory or checkpoint capacity")
            time.sleep(30)
        argv = [sys.executable, "scripts/launch_independent_response.py", phase,
                "--method", job["method"]]
        with (output/f"{phase}.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                    FSD_MAX_GPU_MEMORY_GIB=str(cap), PYTHONUNBUFFERED="1",
                    OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
            record = started_process_record(child, argv)
            update(status="running", phase=phase, child=record, waiting_for=None)
            code = child.wait()
        update(child=None, exit_code=code)
        if code:
            raise RuntimeError(f"{phase} exited {code}; inspect {output/phase}.log")

    try:
        # The original serial supervisor uses this same lock. After this worker
        # finishes, it sees the completion files and cannot retrain the job.
        with experiment_lock(output):
            queue.verify(value)
            prepare_output_storage(output)
            if queue.read(queue.FOLDER/"teacher_targets_ready.json").get("status") != "complete":
                raise ValueError("Prepare the shared teacher targets before parallel launch")
            if not (output/"preflight.json").exists():
                command("preflight")
            checked = queue.read(output/"preflight.json")
            if checked.get("status") != "passed" or checked.get("global_step") != 2:
                raise ValueError("Two-update preflight did not pass")
            complete = queue.read(output/"run.json")
            if complete.get("status") != "complete":
                command("train")
            complete = queue.read(output/"run.json")
            if complete.get("status") != "complete" or complete.get("global_step") != 36000:
                raise ValueError("Training did not complete 36,000 student updates")
            result = ROOT/f"results/sd15/independent36k/{job['method']}_seed42_1steps.result.json"
            if not result.exists():
                command("evaluate")
            from few_step_diffusion.independent_response import validate_result
            validate_result(queue.read(result))
            update(status="evaluated", phase="report", result_file=str(result.relative_to(ROOT)))
            with (output/"report.log").open("a") as log:
                subprocess.run([sys.executable, "scripts/build_assignment2_report.py", "--group-id", "11", "--package"],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            update(status="complete", phase="complete", finished_at=queue.now())
    except Exception as exc:
        update(status="failed", error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "worker", "status"))
    parser.add_argument("--method", choices=queue.METHODS, default="dmd_lpips_scratch")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--memory-cap-gib", type=float, default=13.5)
    args = parser.parse_args()
    if not 0 < args.memory_cap_gib <= 15:
        parser.error("memory cap must be positive and at most 15 GiB")
    os.chdir(ROOT)
    value = queue.read(queue.PLAN)
    queue.verify(value)
    if args.gpu not in value["gpus"]:
        parser.error("GPU is outside the independent experiment plan")
    job = next(j for j in value["jobs"] if j["method"] == args.method)
    output = ROOT/job["run_dir"]
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        import json
        state = queue.read(output/"parallel.json")
        state["alive"] = live(state)
        print(json.dumps(state, indent=2))
    elif args.action == "worker":
        worker(value, job, args.gpu, args.memory_cap_gib)
    else:
        with (output/".parallel-launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if live(queue.read(output/"parallel.json")) or live(queue.read(output/"parallel_launch.json")):
                print("Parallel job already active; refusing a duplicate")
                return
            shared = queue.read(queue.FOLDER/"status.json").get("jobs", {}).get(args.method, {})
            if live(shared):
                print("Job already owned by the serial supervisor")
                return
            argv = [sys.executable, "scripts/launch_independent_job.py", "worker", "--method", args.method,
                    "--gpu", str(args.gpu), "--memory-cap-gib", str(args.memory_cap_gib)]
            with (output/"parallel_launch.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, PYTHONUNBUFFERED="1"))
            record = started_process_record(child, argv)
            if not record:
                raise RuntimeError("Parallel worker exited during startup")
            atomic_json(output/"parallel_launch.json", {"worker": record, "gpu": args.gpu,
                "requested_at": queue.now(), "memory_cap_gib": args.memory_cap_gib})
            print(f"Detached {args.method} worker PID {child.pid} on GPU {args.gpu}")


if __name__ == "__main__":
    main()
