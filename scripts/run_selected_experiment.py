#!/usr/bin/env python3
"""Run one existing non-edge plan on a selected GPU, with resumable evaluation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"baseline", "dmd", "consistency", "instaflow", "swiftbrush_lora",
           "dmd_grouped", "dmd_perceptual", "dmd_diversity", "dmd_perceptual_diversity"}


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--diversity", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from few_step_diffusion.config import config_fingerprint, load_config
    from few_step_diffusion.runtime import atomic_json, experiment_lock, sha256
    from launch_scale50k import process_matches, process_record
    config = load_config(args.config)
    if config["name"] not in ALLOWED or config["train"]["edge_weight"]:
        raise ValueError("Active training includes only the new proposal, its ablations and non-edge baselines")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    process_path = output/"selected_process.json"
    if not args.worker and process_path.exists() and process_matches(json.loads(process_path.read_text())):
        raise RuntimeError("This selected experiment is already active")
    if args.background:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", args.config,
                   "--gpu", args.gpu, "--result-dir", args.result_dir]
        if args.diversity:
            command.append("--diversity")
        log_path = output/"selected_launch.log"
        with log_path.open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
        atomic_json(process_path, process_record(child.pid))
        print(json.dumps({"pid": child.pid, "method": config["name"], "gpu": args.gpu,
                          "log": str(log_path), "status": str(output/"selected_status.json")}))
        return
    state = {"pid": os.getpid(), "status": "running", "phase": "waiting_for_experiment_lock",
        "method": config["name"], "gpu": args.gpu, "config": args.config,
        "output_dir": str(output), "result_dir": args.result_dir, "started_at": now()}
    atomic_json(process_path, process_record(os.getpid()))

    def update(**values):
        state.update(values, updated_at=now())
        atomic_json(output/"selected_status.json", state)

    def command(argv, phase):
        child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
            env=dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
        update(phase=phase, child_pid=child.pid)
        code = child.wait()
        update(child_pid=None)
        if code:
            raise RuntimeError(f"{phase} exited with code {code}")

    update()
    try:
        with experiment_lock(output):
            run = output/"run.json"
            if run.exists():
                done = json.loads(run.read_text())
                if done["status"] != "complete" or config_fingerprint(done["config"]) != config_fingerprint(config):
                    raise ValueError("Completed training differs from this plan")
            else:
                argv = [sys.executable, "-m", "few_step_diffusion.cli", "train", "--config", args.config]
                if (output/"last.pt").exists():
                    argv.append("--resume")
                command(argv, "training_one_step")
            result = Path(args.result_dir)/f"{config['name']}_seed{config['seed']}_1steps.result.json"
            if not result.exists():
                command([sys.executable, "-m", "few_step_diffusion.cli", "evaluate", "--config", args.config,
                    "--checkpoint", str(output/"stage_1"), "--steps", "1", "--result-dir", args.result_dir],
                    "evaluating_5000_fid_is_clip")
            measured = json.loads(result.read_text())
            if (measured["config_hash"] != config_fingerprint(config) or
                    measured["model_identity"] != sha256(output/"stage_1/adapter.safetensors")):
                raise ValueError("Saved evaluation differs from this plan/checkpoint")
            measured["evaluation_context"] = {"physical_gpu": args.gpu, "concurrent_training": True,
                "latency_comparable": False, "note": "Shared hardware; GPU timings are not controlled comparisons."}
            atomic_json(result, measured)
            if args.diversity and "diversity_lpips_alex_mean" not in measured:
                update(phase="evaluating_128_prompts_four_seeds")
                from few_step_diffusion.diversity_evaluation import evaluate_diversity
                evaluate_diversity(config, str(output/"stage_1"), output/"diversity_eval", result)
        update(status="complete", phase="complete", finished_at=now(), child_pid=None)
    except BaseException as exc:
        update(status="failed", phase="failed", error=str(exc), child_pid=None)
        raise


if __name__ == "__main__":
    main()
