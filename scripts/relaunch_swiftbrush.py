#!/usr/bin/env python3
"""Replace the old SwiftBrush worker without interrupting the other GPU jobs."""

from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from launch_scale50k import ROOT, process_matches, process_record


def stop_owned(record, sig):
    if not process_matches(record) or record["cwd"] != str(ROOT):
        raise RuntimeError(f"Process identity changed before stop: {record['pid']}")
    os.kill(record["pid"], sig)
    deadline = time.monotonic()+30
    while process_matches(record) and time.monotonic() < deadline:
        time.sleep(.2)
    if process_matches(record):
        os.kill(record["pid"], signal.SIGKILL)
        deadline = time.monotonic()+10
        while process_matches(record) and time.monotonic() < deadline:
            time.sleep(.2)
        if process_matches(record):
            raise RuntimeError(f"Process did not stop: {record['pid']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    from few_step_diffusion.config import load_config
    from few_step_diffusion.runtime import atomic_json, sha256
    pointer = ROOT/"artifacts/launches/latest_sd15_scale50k.txt"
    previous_folder = Path(pointer.read_text().strip())
    previous = json.loads((previous_folder/"status.json").read_text())
    manifest = json.loads((previous_folder/"launch.json").read_text())
    supervisor = process_record(previous["pid"])
    if "--supervise" not in supervisor["command"] or previous["status"] != "running":
        raise RuntimeError("The original 50k study supervisor is not active")
    config = load_config("configs/sd15_swiftbrush_lora.yaml")
    if Path(config["output_dir"]).exists():
        raise FileExistsError("The new SwiftBrush run folder already exists; inspect before relaunch")
    adopted, stopped = {}, []
    for gpu in ("0", "1", "2", "3"):
        worker = previous["workers"][gpu]
        runner = process_record(worker["child_pid"])
        if ("scripts/run_experiments.py" not in runner["command"] or
                runner["command"][runner["command"].index("--gpu")+1] != gpu or
                runner["cwd"] != str(ROOT)):
            raise RuntimeError(f"Unexpected existing worker on GPU {gpu}")
        children = Path(f"/proc/{runner['pid']}/task/{runner['pid']}/children").read_text().split()
        processes = [runner]+[process_record(int(pid)) for pid in children]
        if gpu == "2":
            if "swiftbrush_vsd" not in runner["command"] or len(processes) != 2:
                raise RuntimeError("GPU 2 is no longer the expected SwiftBrush worker")
            if "few_step_diffusion.cli" not in processes[1]["command"] or "train" not in processes[1]["command"]:
                raise RuntimeError("GPU 2's child is no longer training")
            stopped = processes
        else:
            adopted[gpu] = processes
    original = ROOT/"runs/sd15/scale50k/swiftbrush_vsd_seed42"
    source = {"config": "configs/sd15_swiftbrush_lora.yaml",
              "output_dir": config["output_dir"], "result_dir": "results/sd15/swiftbrush_lora"}
    manifest.update(adopted_workers=adopted, source_recipes={"swiftbrush_lora": source}, prepare_pid=None)
    manifest["groups"]["2"] = ["swiftbrush_lora", "consistency"]
    manifest["relaunch"] = {"reason": "User requested rank-32 G with official SwiftBrush settings and CFG 4.5",
        "old_folder": str(previous_folder), "old_swiftbrush_run": str(original),
        "old_supervisor": supervisor, "stopped_worker_processes": stopped,
        "shared_comparison_note": "CFG 4.5/FP32 SwiftBrush results are kept separate from CFG 7.5/bf16 study"}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    folder = ROOT/"artifacts/launches"/("sd15_swiftbrush_relaunch_"+
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True)
    atomic_json(folder/"launch.json", manifest)
    # Only the controller is replaced. Its other GPU runners/children remain
    # detached and are adopted by the new controller using process identities.
    stop_owned(supervisor, signal.SIGTERM)
    stop_owned(stopped[1], signal.SIGINT)
    if process_matches(stopped[0]):
        stop_owned(stopped[0], signal.SIGTERM)
    import torch
    saved = torch.load(original/"last.pt", map_location="cpu", weights_only=False)
    atomic_json(original/"interrupted.json", {"status": "stopped_for_relaunch",
        "saved_global_step": saved["global_step"], "checkpoint_sha256": sha256(original/"last.pt"),
        "successor_run": config["output_dir"], "successor_controller": str(folder)})
    del saved
    for gpu in adopted:
        (folder/f"gpu{gpu}.log").symlink_to(previous_folder/f"gpu{gpu}.log")
    with (folder/"output.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, str(ROOT/"scripts/launch_scale50k.py"),
            "--supervise", str(folder)], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
    (folder/"supervisor.pid").write_text(str(child.pid)+"\n")
    previous.update(status="superseded", successor_pid=child.pid, successor_folder=str(folder),
                    note="GPU 0/1/3 runners were preserved; only old SwiftBrush GPU 2 training stopped")
    atomic_json(previous_folder/"status.json", previous)
    pointer.write_text(str(folder)+"\n")
    links = {"launch_status.json": "status.json", "launch.log": "output.log",
             "launch_plan.json": "launch.json", "prepare.log": "prepare.log"}
    links.update({f"gpu{gpu}.log": f"gpu{gpu}.log" for gpu in range(4)})
    root = ROOT/"runs/sd15/scale50k"
    for name, target in links.items():
        link = root/name
        link.unlink(missing_ok=True)
        link.symlink_to(folder/target)
    print(json.dumps({"supervisor_pid": child.pid, "folder": str(folder),
        "new_swiftbrush_run": config["output_dir"], "new_results": source["result_dir"],
        "preserved_workers": adopted, "consistency_queued_on_gpu": 2}, indent=2))


if __name__ == "__main__":
    main()
