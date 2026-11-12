#!/usr/bin/env python3
"""Retain the requested DMD2 GPU0 handoff after cache-first recovery."""
import os
from pathlib import Path
import subprocess
import sys
import time

from few_step_diffusion.runtime import atomic_json, experiment_lock
import launch_recent_baselines as queue
from launch_scale50k import process_matches, process_record

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    folder = queue.FOLDER
    with experiment_lock(folder/"move_after_cache"):
        atomic_json(folder/"move_after_cache_process.json", process_record(os.getpid()))
        while True:
            cache = queue.read(folder/"posterior_cache_status.json")
            ready = cache.get("status") == "complete"
            for method in queue.METHODS:
                run = folder/f"{method}_seed42"
                state = queue.read(run/"queue.json")
                execution = queue.read(run/"execution.json")
                ready = (ready and state.get("phase") == "train" and
                         state.get("child") and process_matches(state["child"]) and
                         bool(execution.get("posterior_cache")))
            atomic_json(folder/"move_after_cache_status.json", {
                "status": "launching_handoff" if ready else "waiting_for_cached_training",
                "preferred_gpu": 0, "method": "dmd2", "updated_at": queue.now()})
            if ready:
                placements = {m: {"train_gpu": 0 if m == "dmd2" else 3,
                    "helper_gpu": 3 if m == "dmd2" else 0, "micro_factor": 1,
                    "training_memory_gib": 24,
                    "optimizations": ["fused_adamw", "no_checkpoint", "vae_helper"]}
                    for m in queue.METHODS}
                path = folder/"gpu0_placement.json"
                atomic_json(path, placements)
                subprocess.run([sys.executable, "scripts/accelerate_recent_baselines.py", "launch",
                    "--train-gpu", "3", "--helper-gpu", "0", "--placement-file", str(path)], check=True)
                atomic_json(folder/"move_after_cache_status.json", {"status": "handoff_queued",
                    "method": "dmd2", "preferred_gpu": 0, "updated_at": queue.now()})
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
