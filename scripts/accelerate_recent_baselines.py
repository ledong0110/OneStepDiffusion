#!/usr/bin/env python3
"""Resume recent baselines with faster execution at a fully saved checkpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import torch

from few_step_diffusion.config import config_fingerprint
from few_step_diffusion.runtime import atomic_json, sha256
from launch_dmd_conflict import started_process_record
from launch_scale50k import process_matches, process_record
import launch_recent_baselines as queue

ROOT = Path(__file__).resolve().parents[1]
STATUS = queue.FOLDER/"speedup.json"
FLAGS = ["fused_adamw", "no_checkpoint", "vae_helper"]


def execution_for_job(args, method):
    """Separate the current training GPU from each job's resume destination."""
    placements = getattr(args, "placements", {})
    if set(placements)-set(queue.METHODS):
        raise ValueError("Placement names must identify a recent baseline")
    values = {"train_gpu": args.train_gpu, "helper_gpu": args.helper_gpu,
              "micro_factor": args.micro_factor, "training_memory_gib": args.training_memory_gib,
              "optimizations": list(FLAGS)}
    override = placements.get(method, {})
    if set(override)-set(values):
        raise ValueError("Unknown execution placement setting")
    values.update(override)
    if (values["micro_factor"] not in (1, 2, 4) or
            not 22 <= values["training_memory_gib"] <= 48 or
            set(values["optimizations"])-set(FLAGS)):
        raise ValueError("Invalid execution placement settings")
    if (not isinstance(values["train_gpu"], int) or values["train_gpu"] < 0 or
            not isinstance(values["helper_gpu"], int) or values["helper_gpu"] < 0):
        raise ValueError("GPU indices must be nonnegative integers")
    if "vae_helper" in values["optimizations"] and values["train_gpu"] == values["helper_gpu"]:
        raise ValueError("The helper GPU must differ from the resume destination")
    if "no_checkpoint" in values["optimizations"] and values["micro_factor"] == 4 and values["training_memory_gib"] < 30:
        raise ValueError("The measured microbatch-8 recipe needs at least 30 GiB")
    return SimpleNamespace(**values)


def gate_step(command, output):
    """The preview subprocess starts only after the complete checkpoint save."""
    if command[1:4] != ["-m", "few_step_diffusion.cli", "evaluate"] or "--checkpoint" not in command:
        return None
    index = command.index("--checkpoint")+1
    if index >= len(command):
        return None
    path = Path(command[index])
    if not path.is_absolute():
        path = ROOT/path
    if path.parent.resolve() != (output/"checkpoints").resolve() or not path.name.startswith("step_"):
        return None
    try:
        return int(path.name.removeprefix("step_"))
    except ValueError:
        return None


def resource_gate(free_training, owned_training, free_helper, training_cap=32, helper_cap=2.5):
    # The new process initializes a fresh CUDA context before device_for checks
    # its own one-GiB allocation margin. Reserve another GiB for that context.
    return free_training+owned_training >= training_cap+2 and free_helper >= helper_cap+.5


def capacity(record, args):
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    devices = {}
    for line in lines:
        index, uuid, free = (value.strip() for value in line.split(","))
        devices[int(index)] = (uuid, float(free)/1024)
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    owned = 0.
    for line in rows:
        parts = [value.strip() for value in line.split(",")]
        if len(parts) == 3 and parts[0] == str(record["pid"]) and parts[1] == devices[args.train_gpu][0]:
            try:
                owned += float(parts[2])/1024
            except ValueError:
                pass
    free, helper = devices[args.train_gpu][1], devices[args.helper_gpu][1]
    uses_helper = "vae_helper" in getattr(args, "optimizations", FLAGS)
    ready = (resource_gate(free, owned, helper, args.training_memory_gib) if uses_helper else
             free+owned >= args.training_memory_gib+2)
    return {"free_training_gpu_gib": round(free, 2), "owned_training_gpu_gib": round(owned, 2),
        "free_after_exit_gib": round(free+owned, 2), "required_after_exit_gib": args.training_memory_gib+2,
        "free_helper_gpu_gib": round(helper, 2),
        "ready": ready}


def choose_execution(record, preferred, source_gpu, allow_fallback):
    """Prefer the requested destination; optionally optimize on the source first."""
    budget = capacity(record, preferred)
    if not budget["ready"] and allow_fallback and preferred.train_gpu != source_gpu:
        fallback = SimpleNamespace(**vars(preferred))
        fallback.train_gpu = source_gpu
        fallback_budget = capacity(record, fallback)
        if fallback_budget["ready"]:
            return fallback, fallback_budget
    return preferred, budget


def preview_child(record, output, minimum):
    if not process_matches(record):
        return None
    path = Path(f"/proc/{record['pid']}/task/{record['pid']}/children")
    try:
        children = path.read_text().split()
    except FileNotFoundError:
        return None
    for pid in children:
        try:
            child = process_record(int(pid))
        except (FileNotFoundError, ProcessLookupError):
            continue
        step = gate_step(child["command"], output)
        if step is not None and step >= minimum and process_matches(child):
            return child, step
    return None


def signal_owned(record, value):
    if not process_matches(record):
        raise RuntimeError("Recorded process is no longer the same live process")
    os.kill(record["pid"], value)


def stopped(record):
    return Path(f"/proc/{record['pid']}/stat").read_text().rsplit(") ", 1)[1].split()[0] in ("T", "t")


def supervise(plan, args):
    state = {"status": "waiting_for_saved_checkpoints", "worker": process_record(os.getpid()),
             "started_at": queue.now(), "jobs": {}}
    mutex = threading.Lock()
    transition = threading.Lock()

    def update(method, **values):
        with mutex:
            state["jobs"].setdefault(method, {}).update(values)
            statuses = [job.get("status") for job in state["jobs"].values()]
            if statuses and all(value == "waiting_for_gpu_memory" for value in statuses):
                state["status"] = "waiting_for_gpu_memory"
            elif any(value in ("resuming", "verified_checkpoint") for value in statuses):
                state["status"] = "running"
            else:
                state["status"] = "waiting_for_saved_checkpoints"
            state["updated_at"] = queue.now()
            atomic_json(STATUS, state)

    def migrate(job, allow_fallback=True):
        method = job["method"]
        preferred = execution_for_job(args, method)
        execution_settings = preferred
        output = ROOT/job["run_dir"]
        original = queue.read(output/"queue.json")
        child, owner = original.get("child"), original.get("worker")
        if not child or not owner or not process_matches(child) or not process_matches(owner):
            raise ValueError(f"{method} has no live training process to adopt")
        if original.get("gpu") != args.train_gpu or original.get("phase") != "train":
            raise ValueError(f"{method} is not training on the requested GPU")
        current_execution = queue.read(output/"execution.json")
        if (original["gpu"] == preferred.train_gpu and
                current_execution.get("helper_gpu") == preferred.helper_gpu and
                current_execution.get("micro_factor") == preferred.micro_factor and
                set(current_execution.get("optimizations", [])) == set(preferred.optimizations)):
            update(method, status="already_optimized", child=child, training_gpu=original["gpu"])
            return
        latest = sorted((output/"checkpoints").glob("step_*/metadata.json"))[-1]
        minimum = queue.read(latest)["global_step"]+1000
        update(method, status="waiting", minimum_checkpoint=minimum, original_child=child,
               source_gpu=args.train_gpu, helper_gpu=execution_settings.helper_gpu,
               training_gpu=execution_settings.train_gpu,
               preferred_training_gpu=preferred.train_gpu,
               optimizations=execution_settings.optimizations,
               micro_factor=execution_settings.micro_factor)
        next_check = 0.
        budget = {"ready": False}
        while True:
            if time.monotonic() >= next_check:
                queue.verify_plan(plan)
                execution_settings, budget = choose_execution(child, preferred, args.train_gpu,
                    allow_fallback and getattr(args, "fallback_to_source", False))
                update(method, status="waiting_for_checkpoint" if budget["ready"] else "waiting_for_gpu_memory",
                       capacity=budget, training_gpu=execution_settings.train_gpu,
                       deferred_destination_gpu=preferred.train_gpu if execution_settings.train_gpu != preferred.train_gpu else None)
                next_check = time.monotonic()+10
            gate = preview_child(child, output, minimum)
            if gate and budget["ready"] and transition.acquire(blocking=False):
                try:
                    budget = capacity(child, execution_settings)
                except Exception:
                    transition.release()
                    raise
                if budget["ready"] and process_matches(gate[0]):
                    break
                transition.release()
            if not process_matches(child):
                raise RuntimeError(f"{method} stopped before its next checkpoint")
            time.sleep(.5)
        preview, step = gate
        paused, terminated = [], False
        try:
            # Freeze the CPU parents while the already-launched preview finishes.
            # They are between updates; the train checkpoint is fully committed.
            for record in (child, owner):
                signal_owned(record, signal.SIGSTOP)
                paused.append(record)
            for _ in range(30):
                if all(stopped(record) for record in paused):
                    break
                time.sleep(.1)
            if not all(stopped(record) for record in paused):
                raise RuntimeError("Could not pause the checkpoint boundary")
            claimed = dict(original, worker=process_record(os.getpid()), status="reconfiguring",
                           phase="checkpoint_safe_resume", updated_at=queue.now())
            atomic_json(output/"queue.json", claimed)
            checkpoint = output/"last.pt"
            with checkpoint.open("rb") as handle:
                saved = torch.load(handle, map_location="cpu", weights_only=False)
            if (saved["global_step"] != step or saved["stage_step"] != step or saved["stage_index"] != 0 or
                    saved["auxiliary_updates"] != step*9 or
                    config_fingerprint(saved["config"]) != job["config_hash"]):
                raise ValueError("Saved counters/config do not match the stopped training boundary")
            for optimizer, expected in zip(saved["optimizers"], (step, step*9, step*9), strict=True):
                if not optimizer["state"] or any(float(entry["step"]) != expected for entry in optimizer["state"].values()):
                    raise ValueError("Saved optimizer counters do not match the declared update budget")
            del saved
            update(method, status="verified_checkpoint", checkpoint_step=step)
            deadline = time.monotonic()+120
            while process_matches(preview) and time.monotonic() < deadline:
                time.sleep(.2)
            if process_matches(preview):
                raise RuntimeError("Checkpoint preview is still running; leave the original training intact")
            result = output/"checkpoint_evaluations"/f"step_{step:06d}"/f"{method}_seed42_1steps.result.json"
            adapter = output/"checkpoints"/f"step_{step:06d}"/"adapter.safetensors"
            if not result.exists() or queue.read(result).get("model_identity") != sha256(adapter):
                raise ValueError("Checkpoint preview did not finish with the saved adapter")
            budget = capacity(child, execution_settings)
            if not budget["ready"]:
                raise RuntimeError("GPU capacity changed; keep the original training process")
            audit = {"status": "verified", "checkpoint_step": step, "checkpoint_sha256": sha256(checkpoint),
                "original_worker": owner, "original_child": child, "config_hash": job["config_hash"],
                "updated_at": queue.now(), "checkpoint_path": str(checkpoint.resolve()),
                "source_gpu": args.train_gpu, "destination_gpu": execution_settings.train_gpu,
                "reason": "GPU placement and execution change at a complete checkpoint; no saved updates discarded"}
            atomic_json(output/"speedup_resume.json", audit)
            # The old worker must not overwrite queue state when its child exits.
            for record in (owner, child):
                signal_owned(record, signal.SIGTERM)
                if process_matches(record):
                    os.kill(record["pid"], signal.SIGCONT)
            for _ in range(100):
                if not any(process_matches(record) for record in (owner, child)):
                    break
                time.sleep(.1)
            if any(process_matches(record) for record in (owner, child)):
                raise RuntimeError("Old process has not exited; refuse a duplicate")
            terminated = True
            execution = {"policy": "recent-execution-v1", "config_hash": job["config_hash"],
                "optimizations": execution_settings.optimizations, "micro_factor": execution_settings.micro_factor,
                "training_gpu": execution_settings.train_gpu, "helper_gpu": execution_settings.helper_gpu,
                "training_memory_gib": execution_settings.training_memory_gib,
                "helper_memory_gib": 2.5, "effective_batch_size": 16,
                "auxiliary_updates_per_student": 9, "resume_step": step}
            if current_execution.get("posterior_cache"):
                execution["posterior_cache"] = current_execution["posterior_cache"]
            atomic_json(output/"execution.json", execution)
            argv = [sys.executable, "scripts/launch_recent_baselines.py", "worker", "--method", method,
                    "--gpu", str(execution_settings.train_gpu)]
            with (output/"launch.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, PYTHONUNBUFFERED="1"))
            record = started_process_record(process, argv)
            if record is None:
                raise RuntimeError("Optimized worker exited during startup")
            update(method, status="resuming", checkpoint_step=step, replacement_worker=record,
                   execution=execution)
            for _ in range(300):
                current = queue.read(output/"queue.json")
                if current.get("worker") == record and current.get("phase") == "train" and current.get("child"):
                    log = output/"train.log"
                    with log.open("rb") as handle:
                        handle.seek(max(0, log.stat().st_size-16000))
                        counts = re.findall(r"(?<!\d)(\d+)/36000\b", handle.read().decode(errors="replace"))
                    if process_matches(current["child"]) and any(int(count) > step for count in counts):
                        update(method, status="resumed", child=current["child"],
                               deferred_destination_gpu=preferred.train_gpu if execution_settings.train_gpu != preferred.train_gpu else None)
                        break
                if not process_matches(record) or current.get("status") == "failed":
                    raise RuntimeError("Optimized worker failed; verified checkpoint is preserved")
                time.sleep(1)
            else:
                raise RuntimeError("Optimized worker did not reach training; inspect launch.log")
        except Exception:
            if not terminated:
                for record in paused:
                    if process_matches(record):
                        os.kill(record["pid"], signal.SIGCONT)
                if process_matches(owner) and process_matches(child):
                    atomic_json(output/"queue.json", original)
            raise
        finally:
            transition.release()
        if execution_settings.train_gpu != preferred.train_gpu:
            # Faster execution can fit the source before the desired destination
            # is free. Preserve the destination request after this first resume.
            migrate(job, allow_fallback=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(migrate, job): job["method"] for job in plan["jobs"]}
        errors = []
        for future, method in futures.items():
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
                update(method, status="failed", error=str(exc))
        state["status"] = "needs_attention" if errors else "complete"
        state["updated_at"] = queue.now()
        atomic_json(STATUS, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "supervise", "status"))
    parser.add_argument("--train-gpu", type=int, default=3)
    parser.add_argument("--helper-gpu", type=int, default=2)
    parser.add_argument("--micro-factor", type=int, choices=(2, 4), default=4)
    parser.add_argument("--training-memory-gib", type=float, default=32)
    parser.add_argument("--placement-file", type=Path,
                        help="JSON per-method destination/flags overrides; the source GPU is --train-gpu")
    parser.add_argument("--fallback-to-source", action="store_true",
                        help="Optimize on the source GPU first when the destination is full, then keep waiting to move")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "status":
        import json
        state = queue.read(STATUS)
        state["worker_alive"] = bool(state.get("worker") and process_matches(state["worker"]))
        print(json.dumps(state, indent=2))
        return
    plan = queue.read(queue.PLAN)
    queue.verify_plan(plan)
    if args.train_gpu == args.helper_gpu:
        parser.error("The helper GPU must differ from the training GPU")
    if not 30 <= args.training_memory_gib <= 48:
        parser.error("The measured microbatch-8 recipe requires a cap between 30 and 48 GiB")
    args.placements = queue.read(args.placement_file) if args.placement_file else {}
    for method in queue.METHODS:
        execution_for_job(args, method)
    if args.action == "supervise":
        supervise(plan, args)
        return
    with (queue.FOLDER/".speedup-launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prior = queue.read(queue.FOLDER/"speedup_launch.json")
        if prior.get("worker") and process_matches(prior["worker"]):
            print("Checkpoint-boundary speedup worker is already active")
            return
        argv = [sys.executable, "scripts/accelerate_recent_baselines.py", "supervise",
                "--train-gpu", str(args.train_gpu), "--helper-gpu", str(args.helper_gpu),
                "--micro-factor", str(args.micro_factor), "--training-memory-gib", str(args.training_memory_gib),
                *(["--placement-file", str(args.placement_file.resolve())] if args.placement_file else []),
                *(["--fallback-to-source"] if args.fallback_to_source else [])]
        with (queue.FOLDER/"speedup.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        record = started_process_record(process, argv)
        if not record:
            raise RuntimeError("Checkpoint-boundary worker exited during startup")
        atomic_json(queue.FOLDER/"speedup_launch.json", {"worker": record, "requested_at": queue.now(),
            "train_gpu": args.train_gpu, "helper_gpu": args.helper_gpu, "micro_factor": args.micro_factor,
            "placements": args.placements, "fallback_to_source": args.fallback_to_source})
        print(f"Detached checkpoint-boundary speedup worker PID {process.pid}")


if __name__ == "__main__":
    main()
