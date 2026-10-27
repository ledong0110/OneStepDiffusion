#!/usr/bin/env python3
"""Launch detached, resumable 50k-caption training across four GPUs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {0: ["baseline", "edge_constant", "edge_weight_low"],
          1: ["proposed", "edge_weight_high"],
          2: ["swiftbrush_vsd", "consistency"],
          3: ["dmd", "instaflow"]}


def now():
    return datetime.now(timezone.utc).isoformat()


def process_record(pid):
    """Bind adoption to a process incarnation, argv and workspace, not only a PID."""
    proc = Path(f"/proc/{pid}")
    fields = (proc/"stat").read_text().rsplit(") ", 1)[1].split()
    return {"pid": pid, "start_ticks": fields[19],
            "command": [s.decode() for s in (proc/"cmdline").read_bytes().split(b"\0") if s],
            "cwd": str((proc/"cwd").resolve())}


def process_matches(record):
    try:
        current = process_record(record["pid"])
        fields = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(") ", 1)[1].split()
        return current == record and fields[0] not in ("Z", "X")
    except (FileNotFoundError, ProcessLookupError):
        return False


def supervise(folder):
    from few_step_diffusion.runtime import atomic_json
    manifest = json.loads((folder/"launch.json").read_text())
    state = {"pid": os.getpid(), "status": "running", "started_at": now(),
             "preset": "scale50k", "data": {"status": "waiting"},
             "flow_diagnostic": {"status": "waiting"}, "workers": {}}
    if "relaunch" in manifest:
        state["relaunch"] = manifest["relaunch"]
    lock = threading.Lock()

    def update(component=None, **values):
        with lock:
            target = state if component is None else state.setdefault(component, {})
            target.update(values)
            state["updated_at"] = now()
            atomic_json(folder/"status.json", state)

    def worker_update(gpu, **values):
        with lock:
            state["workers"].setdefault(str(gpu), {}).update(values)
            state["updated_at"] = now()
            atomic_json(folder/"status.json", state)

    python = sys.executable

    def command(args, log_name, gpu=None, phase=None):
        print("START", phase, args, flush=True)
        with (folder/log_name).open("ab", buffering=0) as log:
            child = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
            if gpu is not None:
                worker_update(gpu, status="running", phase=phase, child_pid=child.pid, command=args)
            code = child.wait()
        if code:
            raise RuntimeError(f"{phase or log_name} exited with code {code}; see {folder/log_name}")

    def prepare():
        try:
            update("data", status="waiting_for_existing_download", pid=manifest.get("prepare_pid"))
            pid = manifest.get("prepare_pid")
            while pid and Path(f"/proc/{pid}/cmdline").exists():
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
                if b"few_step_diffusion.cli\x00prepare" not in cmdline:
                    break
                time.sleep(10)
            update("data", status="verifying_cache")
            command([python, "-m", "few_step_diffusion.cli", "prepare", "--config",
                     "configs/sd15_50k.yaml", "--metrics"], "prepare.log", phase="prepare_50k")
            update("data", status="complete", finished_at=now())
        except BaseException as exc:
            update("data", status="failed", error=str(exc))
            raise

    def flow_gate():
        path = ROOT/manifest["flow_diagnostic"]
        try:
            deadline = time.monotonic()+900
            while not path.is_file():
                if time.monotonic() >= deadline:
                    raise RuntimeError("Flow learning diagnostic did not finish within 15 minutes")
                time.sleep(5)
            result = json.loads(path.read_text())
            update("flow_diagnostic", **result)
            if result["status"] != "passed":
                raise RuntimeError("InstaFlow learning diagnostic failed; large flow training withheld")
        except BaseException as exc:
            update("flow_diagnostic", status="failed", error=str(exc))
            raise

    def worker(gpu, methods, prepared, flow):
        log_name = f"gpu{gpu}.log"
        base = [python, "scripts/run_experiments.py", "--preset", "scale50k", "--profile", "sd15",
                "--gpu", str(gpu), "--resume", "--seeds", *map(str, manifest["seeds"])]
        try:
            worker_update(gpu, status="starting", methods=methods)
            adopted = manifest.get("adopted_workers", {}).get(str(gpu), [])
            if adopted:
                worker_update(gpu, status="running", phase="adopting_existing_worker",
                              adopted_processes=adopted, child_pid=adopted[0]["pid"])
                while any(process_matches(record) for record in adopted):
                    time.sleep(5)
                # Re-run --resume after adoption to verify completed artifacts,
                # finish any missing evaluation, and recover interrupted updates.
            if gpu == 0:
                worker_update(gpu, status="waiting", phase="waiting_for_training_images")
                prepared.result()
                command(base+["--teacher-only", "--eval-only"], log_name, gpu, "evaluate_teacher")
            for method in methods:
                source = manifest.get("source_recipes", {}).get(method)
                if source:
                    if manifest["seeds"] != [42]:
                        raise ValueError("The source-recipe handoff currently uses seed 42")
                    command([python, "scripts/run_swiftbrush_source.py", "--config", source["config"],
                             "--gpu", str(gpu), "--result-dir", source["result_dir"], "--resume"],
                            log_name, gpu, f"train_and_evaluate_{method}")
                    continue
                if method not in ("swiftbrush_vsd", "dmd", "instaflow"):
                    worker_update(gpu, status="waiting", phase="waiting_for_training_images", method=method)
                    prepared.result()
                if method == "instaflow":
                    worker_update(gpu, status="waiting", phase="waiting_for_flow_diagnostic", method=method)
                    flow.result()
                command(base+["--methods", method, "--skip-teacher"], log_name, gpu, f"train_and_evaluate_{method}")
            # All gallery images use training seed 42 unless the launcher was given another first seed.
            gallery_methods = [m for m in methods if m not in manifest.get("source_recipes", {})]
            gallery = [python, "scripts/generate_qualitative.py", "--run-root", "runs/sd15/scale50k",
                       "--output-dir", "results/sd15/scale50k/qualitative", "--seed", str(manifest["seeds"][0]),
                       "--gpu", str(gpu), "--resume", "--methods", *gallery_methods]
            if gpu != 0:
                gallery.append("--skip-teacher")
            if gallery_methods:
                command(gallery, log_name, gpu, "generate_qualitative")
            worker_update(gpu, status="complete", phase="complete", child_pid=None, finished_at=now())
        except BaseException as exc:
            worker_update(gpu, status="failed", error=str(exc), child_pid=None)
            raise

    update()
    errors = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        prepared, flow = pool.submit(prepare), pool.submit(flow_gate)
        futures = [pool.submit(worker, int(gpu), methods, prepared, flow)
                   for gpu, methods in manifest["groups"].items()]
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException as exc:
                errors.append(str(exc))
                print("WORKER FAILED", str(exc), flush=True)
    if errors:
        update(status="failed", errors=errors)
        raise RuntimeError("One or more study workers failed")
    update(phase="compile_report")
    try:
        command([python, "scripts/build_report.py", "--profile", "sd15", "--results-dir",
                 "results/sd15/scale50k", "--study-config", "configs/sd15_50k.yaml",
                 "--sample-seed", str(manifest["seeds"][0]), "--require-results"],
                "report.log", phase="compile_report")
        update(status="complete", phase="complete", finished_at=now())
    except BaseException as exc:
        update(status="failed", error=str(exc))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--supervise", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--prepare-pid", type=int)
    p.add_argument("--flow-diagnostic", default="runs/verification/flow_vp_scale50k/diagnostic.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    if args.supervise:
        supervise(args.supervise)
        return
    if args.prepare_pid is None:
        previous_prepare = ROOT/"artifacts/launches/latest_sd15_scale50k_prepare.txt"
        if previous_prepare.is_file():
            pid_path = Path(previous_prepare.read_text().strip())/"pid"
            if pid_path.is_file():
                candidate = int(pid_path.read_text())
                if Path(f"/proc/{candidate}/cmdline").exists():
                    args.prepare_pid = candidate
    manifest = {"preset": "scale50k", "training_captions": 50000, "evaluation_pairs": 5000,
                "generator_updates_per_method": 18000, "effective_batch": 16,
                "student_steps": 1, "student_nfe": 1, "seeds": args.seeds, "groups": GROUPS,
                "prepare_pid": args.prepare_pid, "flow_diagnostic": args.flow_diagnostic}
    prepare_pointer = ROOT/"artifacts/launches/latest_sd15_scale50k_prepare.txt"
    if prepare_pointer.is_file():
        manifest["download_log"] = str(Path(prepare_pointer.read_text().strip())/"output.log")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    base = ROOT/"artifacts/launches"
    pointer = base/"latest_sd15_scale50k.txt"
    if pointer.exists():
        old = Path(pointer.read_text().strip())/"status.json"
        if old.exists():
            previous = json.loads(old.read_text())
            proc = Path(f"/proc/{previous.get('pid')}/cmdline")
            if previous.get("status") == "running" and proc.exists() and b"--supervise" in proc.read_bytes():
                raise RuntimeError("The 50k study is already running; inspect its status before another launch")
    folder = base/("sd15_scale50k_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True)
    (folder/"launch.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with (folder/"output.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--supervise", str(folder)],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    (folder/"supervisor.pid").write_text(str(child.pid)+"\n")
    pointer.write_text(str(folder)+"\n")
    run = ROOT/"runs/sd15/scale50k"
    run.mkdir(parents=True, exist_ok=True)
    links = {"launch_status.json": "status.json", "launch.log": "output.log",
             "launch_plan.json": "launch.json", "prepare.log": "prepare.log"}
    links.update({f"gpu{gpu}.log": f"gpu{gpu}.log" for gpu in GROUPS})
    for name, target in links.items():
        link = run/name
        link.unlink(missing_ok=True)
        link.symlink_to(folder/target)
    if manifest.get("download_log"):
        link = run/"download.log"
        link.unlink(missing_ok=True)
        link.symlink_to(manifest["download_log"])
    print(json.dumps({"supervisor_pid": child.pid, "folder": str(folder), **manifest}, indent=2))


if __name__ == "__main__":
    main()
