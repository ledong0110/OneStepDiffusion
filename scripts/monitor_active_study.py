#!/usr/bin/env python3
"""Track only the current one-step proposal, three controls and five baselines."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from few_step_diffusion.runtime import atomic_json

ALLOWED = {"baseline", "dmd", "consistency", "instaflow", "swiftbrush_lora",
           "dmd_grouped", "dmd_perceptual", "dmd_diversity", "dmd_perceptual_diversity"}


def snapshot(job):
    method, run = job["method"], Path(job["run_dir"])
    if method not in ALLOWED:
        raise ValueError("Edge methods are outside the active study")
    state = {"method": method, "gpu": job.get("gpu"), "status": "queued",
             "additional_updates": 0, "cumulative_updates": 18000, "target_updates": 36000}
    if job.get("owner_status"):
        owner = json.loads(Path(job["owner_status"]).read_text())
        worker = owner.get("workers", owner.get("jobs", {})).get(job["owner_key"], {})
        if worker.get("method", method) == method:
            state.update(status=worker.get("status", "queued"), phase=worker.get("phase"),
                         child_pid=worker.get("child_pid"))
            if worker.get("error"):
                state["error"] = worker["error"]
            if method == "instaflow" and worker.get("status") == "running" and not (run/"train.jsonl").exists():
                state["phase"] = "preparing_fresh_reflow_pairs"
    log = run/"train.jsonl"
    if log.exists():
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size-8192))
            rows = handle.read().decode().splitlines()
        for row in reversed(rows):
            try:
                last = json.loads(row)
                state.update(additional_updates=last["global_step"],
                             cumulative_updates=18000+last["global_step"])
                if state["status"] == "queued":
                    state["status"] = "training"
                break
            except (ValueError, KeyError):
                continue
    selected = run/"selected_status.json"
    if selected.exists():
        worker = json.loads(selected.read_text())
        state.update(status=worker["status"], phase=worker.get("phase"), pid=worker.get("pid"),
                     child_pid=worker.get("child_pid"))
        if worker.get("error"):
            state["error"] = worker["error"]
    completed = run/"run.json"
    if completed.exists():
        trained = json.loads(completed.read_text())
        if trained["status"] == "complete":
            state.update(status="evaluating", additional_updates=trained["global_step"],
                         cumulative_updates=trained.get("cumulative_training_updates", 18000+trained["global_step"]))
    result = Path(job["result_dir"])/f"{method}_seed42_1steps.result.json"
    if result.exists():
        metrics = json.loads(result.read_text())
        if metrics.get("status") != "complete" or metrics.get("cumulative_training_updates") != 36000:
            raise ValueError(f"Active-study result has an unexpected completion/budget: {result}")
        state.update(status="complete", additional_updates=metrics["training_updates"],
            cumulative_updates=metrics["cumulative_training_updates"], result=str(result),
            fid=metrics["fid"], clip_score_mean=metrics.get("clip_score_mean"))
        if method.startswith("dmd_") and "diversity_lpips_alex_mean" not in metrics:
            state.update(status="evaluating", phase="diversity_evaluation")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    plan = json.loads((args.folder/"launch.json").read_text())
    while True:
        states = {job["method"]: snapshot(job) for job in plan["jobs"]}
        complete = all(s["status"] == "complete" for s in states.values())
        failed = [m for m, s in states.items() if s["status"] == "failed"]
        state = {"status": "complete" if complete else "needs_attention" if failed else "running",
            "scope": "new proposal, proposal ablations and non-edge baselines only",
            "updated_at": datetime.now(timezone.utc).isoformat(), "jobs": states,
            "excluded_methods": plan["excluded_methods"], "completed": sum(s["status"] == "complete" for s in states.values()),
            "total": len(states), "failed_methods": failed}
        atomic_json(args.folder/"status.json", state)
        with (args.folder/"summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["method", "gpu", "status", "additional_updates",
                "cumulative_updates", "target_updates", "fid", "clip_score_mean"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(states.values())
        if args.once or complete:
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
