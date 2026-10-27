#!/usr/bin/env python3
"""Launch the scale50k InstaFlow adaptation alongside the existing GPU worker."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from launch_scale50k import ROOT, process_matches, process_record


def now():
    return datetime.now(timezone.utc).isoformat()


def supervise(folder):
    manifest = json.loads((folder/"launch.json").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = manifest["gpu"]
    from few_step_diffusion.runtime import atomic_json
    run = ROOT/manifest["run_dir"]
    state = {"pid": os.getpid(), "status": "running", "started_at": now(), **manifest}
    with (folder/"worker.log").open("ab", buffering=0) as log:
        child = subprocess.Popen(manifest["command"], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
        state["child_pid"] = child.pid
        try:
            while True:
                if (run/"run.json").exists():
                    phase = "evaluating_5000_images"
                elif (run/"teacher_pairs.json").exists():
                    phase = "training"
                else:
                    phase = "preparing_50000_teacher_pairs"
                state.update(phase=phase, updated_at=now())
                atomic_json(folder/"status.json", state)
                try:
                    code = child.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if code:
                raise RuntimeError(f"InstaFlow runner exited {code}; see {folder/'worker.log'}")
            state.update(status="complete", phase="complete", finished_at=now(), child_pid=None)
            atomic_json(folder/"status.json", state)
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            atomic_json(folder/"status.json", state)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="3")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    if args.worker:
        supervise(args.worker)
        return
    diagnostic = ROOT/"runs/verification/flow_vp_scale50k/diagnostic.json"
    if json.loads(diagnostic.read_text()).get("status") != "passed":
        raise RuntimeError("The flow learning diagnostic must pass before launching")
    manifest = {"method": "instaflow", "gpu": args.gpu, "seed": 42,
                "run_dir": "runs/sd15/scale50k/instaflow_seed42",
                "result_dir": "results/sd15/scale50k", "teacher_pairs": 50000,
                "generator_updates": 18000, "effective_batch_size": 16, "student_lora_rank": 32,
                "sampling_steps": 1, "nfe": 1, "flow_diagnostic": str(diagnostic),
                "command": [sys.executable, "scripts/run_experiments.py", "--preset", "scale50k",
                            "--profile", "sd15", "--gpu", args.gpu, "--seeds", "42", "--methods",
                            "instaflow", "--skip-teacher", "--resume"]}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    pointer = ROOT/"artifacts/launches/latest_sd15_instaflow.txt"
    if pointer.is_file():
        status = Path(pointer.read_text().strip())/"status.json"
        if status.is_file():
            previous = json.loads(status.read_text())
            if previous.get("status") in ("starting", "running"):
                try:
                    record = process_record(previous["pid"])
                except FileNotFoundError:
                    record = None
                if record and process_matches(record) and any("launch_instaflow.py" in s for s in record["command"]):
                    raise RuntimeError("An InstaFlow launcher is already active; inspect its status")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from few_step_diffusion.runtime import atomic_json
    folder = ROOT/"artifacts/launches"/("sd15_instaflow_gpu"+args.gpu+"_"+
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True)
    atomic_json(folder/"launch.json", manifest)
    with (folder/"output.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(folder)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    atomic_json(folder/"status.json", {"pid": child.pid, "status": "starting", **manifest})
    pointer.write_text(str(folder)+"\n")
    run = ROOT/manifest["run_dir"]
    run.mkdir(parents=True, exist_ok=True)
    for link, target in ((run/"launch_status.json", folder/"status.json"),
                         (run/"launch.log", folder/"worker.log"),
                         (run.parent/"instaflow_launch_status.json", folder/"status.json")):
        link.unlink(missing_ok=True)
        link.symlink_to(target)
    index = run.parent/"parallel_workers.json"
    entries = json.loads(index.read_text()) if index.exists() else {}
    entries["instaflow_seed42"] = {"gpu": args.gpu, "status_file": str(run/"launch_status.json")}
    atomic_json(index, entries)
    print(json.dumps({"pid": child.pid, "folder": str(folder), **manifest}, indent=2))


if __name__ == "__main__":
    main()
