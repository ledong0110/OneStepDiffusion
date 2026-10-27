#!/usr/bin/env python3
"""Run a separate SwiftBrush training recipe, optionally detached."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/sd15_swiftbrush_source.yaml")
    p.add_argument("--gpu", default="3")
    p.add_argument("--output")
    p.add_argument("--result-dir")
    p.add_argument("--updates", type=int)
    p.add_argument("--accumulation", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--background", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    os.chdir(ROOT)
    # Set device visibility before any torch/CUDA initialization.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from few_step_diffusion.config import load_config, validate
    from few_step_diffusion.runtime import atomic_json
    import yaml
    config = load_config(args.config)
    args.result_dir = args.result_dir or f"results/sd15/{config['name']}"
    if config["train"].get("vsd_recipe") != "swiftbrush-source-v1":
        raise ValueError("This launcher only accepts the SwiftBrush source recipe")
    if args.output:
        config["output_dir"] = args.output
    if args.updates is not None:
        config["train"]["updates_per_stage"] = args.updates
    if args.accumulation is not None:
        config["train"]["grad_accumulation"] = args.accumulation
    validate(config)
    output = Path(config["output_dir"]).resolve()
    status_file = output/"launch_status.json"
    identity = {"gpu": args.gpu, "output_dir": str(output), "result_dir": args.result_dir,
                "config": config}
    if args.dry_run:
        print(json.dumps(identity, indent=2))
        return
    if not args.worker and status_file.is_file():
        previous = json.loads(status_file.read_text())
        proc = Path(f"/proc/{previous.get('pid')}/cmdline")
        if proc.is_file() and b"run_swiftbrush_source.py" in proc.read_bytes():
            raise RuntimeError("This source-recipe run is already active; inspect launch_status.json")
    if not args.resume and ((output/"last.pt").exists() or (output/"run.json").exists()):
        raise ValueError("Run exists; use --resume or a new --output")
    output.mkdir(parents=True, exist_ok=True)
    if args.background:
        plan = output/"launch_plan.yaml"
        plan.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, str(Path(__file__).resolve()), "--config", str(plan),
                   "--gpu", args.gpu, "--result-dir", args.result_dir, "--worker"]
        if args.resume:
            command.append("--resume")
        if args.skip_eval:
            command.append("--skip-eval")
        with (output/"launch.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONUNBUFFERED="1"))
        atomic_json(output/"launch_pid.json", {"pid": child.pid, "command": command})
        print(json.dumps({"pid": child.pid, "log": str(output/"launch.log"), **identity}, indent=2))
        return
    state = {"pid": os.getpid(), "status": "running", "phase": "training", **identity}
    atomic_json(status_file, state)
    try:
        from few_step_diffusion.training import train
        train(config, resume=args.resume and (output/"last.pt").exists())
        if not args.skip_eval:
            state["phase"] = "evaluating"
            atomic_json(status_file, state)
            from few_step_diffusion.sd_evaluation import evaluate_sd
            result = Path(args.result_dir)/f"{config['name']}_seed{config['seed']}_1steps.result.json"
            if result.exists():
                if not args.resume:
                    raise FileExistsError(f"Result already exists: {result}")
                from few_step_diffusion.config import config_fingerprint
                from few_step_diffusion.runtime import sha256
                record = json.loads(result.read_text())
                weights = ("adapter.safetensors" if config["train"].get("student_parameterization") == "lora"
                           else "diffusion_pytorch_model.safetensors")
                if (record.get("config_hash") != config_fingerprint(config) or
                        record.get("model_identity") != sha256(output/"stage_1"/weights)):
                    raise ValueError("Saved evaluation differs from this run")
            else:
                evaluate_sd(config, str(output/"stage_1"), 1, args.result_dir)
        state.update(status="complete", phase="complete")
        atomic_json(status_file, state)
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed",
                     error=str(exc))
        atomic_json(status_file, state)
        raise


if __name__ == "__main__":
    main()
