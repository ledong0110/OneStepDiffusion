#!/usr/bin/env python3
"""Sequential, resumable experiment runner; --shard-index/--shard-count split team jobs."""

import argparse
from contextlib import nullcontext
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from few_step_diffusion.config import (config_fingerprint, experiment_config, load_config, merge,
                                      validate)
from few_step_diffusion.runtime import experiment_lock

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("smoke", "pilot", "fair_pilot", "full", "scale50k"), default="pilot")
    p.add_argument("--profile", choices=("sd15", "celebahq", "cifar10"), default="sd15")
    p.add_argument("--methods", nargs="+", choices=("baseline", "proposed", "edge_constant", "swiftbrush_vsd", "consistency", "dmd", "instaflow", "edge_weight_low", "edge_weight_high"))
    p.add_argument("--ablations", action="store_true", help="Add matched edge-weight 0.03 and 0.30 runs")
    p.add_argument("--skip-teacher", action="store_true", help="For a worker sharing results with another teacher-evaluation worker")
    p.add_argument("--teacher-only", action="store_true", help="Evaluate teacher references only")
    p.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"))
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-entity")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--gpu", help="Physical GPU index, passed through CUDA_VISIBLE_DEVICES")
    p.add_argument("--updates", type=int, help="Optimizer updates per progressive stage")
    p.add_argument("--num-samples", type=int)
    p.add_argument("--teacher-steps", nargs="+", type=int,
                   help="Override teacher evaluations without changing saved training configs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--eval-only", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    args = p.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        p.error("Require 0 <= shard-index < shard-count")
    os.chdir(ROOT)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    profile = "cifar10" if args.preset == "smoke" else args.profile
    default_methods = (["baseline", "proposed", "edge_constant", "swiftbrush_vsd"]
                       if args.preset in ("fair_pilot", "full", "scale50k") else ["baseline", "proposed", "swiftbrush_vsd"])
    if profile == "sd15":
        default_methods += ["consistency", "dmd", "instaflow"]
    methods = list(args.methods or default_methods)
    if args.ablations:
        methods += [method for method in ("edge_weight_low", "edge_weight_high") if method not in methods]
    if profile != "sd15" and (args.preset in ("fair_pilot", "scale50k") or
            any(method in ("consistency", "dmd", "instaflow", "edge_weight_low", "edge_weight_high") for method in methods)):
        p.error("Matched one-step comparisons and these ablations use --profile sd15")
    seeds = args.seeds or ([42, 43, 44] if args.preset == "full" else [42])
    jobs = ([] if args.skip_teacher else [("teacher", 0)]) + [(method, seed) for method in methods for seed in seeds]
    if args.teacher_only:
        if args.skip_teacher or args.train_only:
            p.error("--teacher-only cannot be combined with --skip-teacher or --train-only")
        jobs = [("teacher", 0)]
    jobs = jobs[args.shard_index::args.shard_count]
    profile_path = Path(profile) if profile != "cifar10" else Path()
    results_dir = Path(args.results_dir) / profile_path / args.preset
    run_root = Path("runs") / profile_path / args.preset
    profile_file = f"configs/{profile}.yaml" if profile != "cifar10" else "configs/base.yaml"
    profile_config = load_config(profile_file)
    profile_overlay = yaml.safe_load(Path(profile_file).read_text()) if profile != "cifar10" else {}
    profile_overlay = {k: v for k, v in profile_overlay.items() if k not in ("extends", "output_dir")}
    if args.preset == "scale50k":
        scale = yaml.safe_load(Path("configs/sd15_50k.yaml").read_text())
        profile_overlay = merge(profile_overlay, {k: v for k, v in scale.items() if k not in ("extends", "output_dir")})
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    def run(command):
        print("$ " + shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)

    for method, seed in jobs:
        if method == "teacher" and args.train_only:
            continue
        config = load_config("configs/baseline.yaml" if method == "teacher" else f"configs/{method}.yaml")
        config = merge(config, profile_overlay)
        one_step = config["train"]["mode"] in ("vsd", "consistency", "dmd", "instaflow")
        if args.preset == "smoke":
            config = merge(config, {"device": "cpu", "precision": "fp32", "model": {"source": "tiny"},
                "data": {"fake": True, "batch_size": 2}, "wandb": {"mode": "disabled"},
                "train": {"initial_steps": 8, "stages": [4, 2, 1], "updates_per_stage": 2,
                          "warmup_steps": 0, "save_every": 1, "log_every": 1, "preview_every": 2},
                "eval": {"steps": [8, 4, 2, 1], "num_samples": 8, "num_real": 8,
                         "batch_size": 2, "benchmark_repeats": 2, "benchmark_warmup": 1}})
        elif args.preset == "pilot":
            pilot_updates, pilot_samples = (100, 128) if profile != "cifar10" else (500, 1000)
            config = merge(config, {"train": {"stages": [32, 16, 8, 4], "updates_per_stage": pilot_updates},
                "eval": {"steps": [64, 32, 16, 8, 4], "num_samples": pilot_samples, "num_real": pilot_samples,
                         "reference_split": "validation" if profile != "cifar10" else "train"}})
            if config["train"]["mode"] == "dmd":
                config["train"]["dmd_pair_count"] = 64
            if config["train"]["mode"] == "instaflow":
                config["train"]["instaflow_pair_count"] = 64
        elif args.preset == "fair_pilot":
            config = merge(config, {"train": {"stages": [32, 16, 8, 4, 2, 1], "updates_per_stage": 100},
                "eval": {"steps": [64, 1], "num_samples": 128, "num_real": 128,
                         "reference_split": "validation", "comparison_steps": 1}})
            if config["train"]["mode"] == "dmd":
                config["train"]["dmd_pair_count"] = 64
            if config["train"]["mode"] == "instaflow":
                config["train"]["instaflow_pair_count"] = 64
        elif args.preset == "scale50k":
            if config["train"]["mode"] == "dmd":
                config["train"]["dmd_pair_count"] = 5000
            if config["train"]["mode"] == "instaflow":
                config["train"]["instaflow_pair_count"] = 50000
                config["train"]["instaflow_prediction_type"] = "epsilon-converted-velocity"
        if args.updates is not None:
            config["train"]["updates_per_stage"] = args.updates
        if one_step:
            # Match cumulative generator updates to the complete PD curriculum.
            # Auxiliary scorer updates are additional and reported separately.
            stage_count = {"smoke": 3, "pilot": 4, "fair_pilot": 6, "scale50k": 6,
                           "full": len(profile_config["train"]["stages"])}[args.preset]
            per_stage = (config["train"]["updates_per_stage"] if args.updates is not None or args.preset != "full"
                         else profile_config["train"]["updates_per_stage"])
            config["train"]["updates_per_stage"] = per_stage * stage_count
            config["train"]["stages"] = [1]
        if args.wandb_mode is not None:
            config["wandb"]["mode"] = args.wandb_mode
        if args.wandb_project is not None:
            config["wandb"]["project"] = args.wandb_project
        if args.wandb_entity is not None:
            config["wandb"]["entity"] = args.wandb_entity
        config["wandb"]["group"] = config["wandb"]["group"] or (
            f"{profile}/{args.preset}" if profile != "cifar10" else args.preset)
        if args.no_progress:
            config["progress_bar"] = False
        if args.num_samples is not None:
            config["eval"]["num_samples"] = args.num_samples
        if args.gpu is not None:
            config["device"] = "cuda"
        config["seed"] = seed
        config["output_dir"] = str(run_root / f"{method}_seed{seed}")
        validate(config)
        plan = run_root / "plans" / f"{method}_seed{seed}.yaml"
        print(f"\nJob: {method}, seed={seed}, {profile}/{args.preset}", flush=True)
        # A separately launched method can share the study queue. The later
        # queue entry waits here, then verifies/reuses its finished artifacts.
        with nullcontext() if args.dry_run else experiment_lock(config["output_dir"]):
            if not args.dry_run:
                plan.parent.mkdir(parents=True, exist_ok=True)
                plan.write_text(yaml.safe_dump(config, sort_keys=False))
            base_command = [sys.executable, "-m", "few_step_diffusion.cli"]
            if method != "teacher" and not args.eval_only:
                output = Path(config["output_dir"])
                completed = output / "run.json"
                if completed.exists():
                    saved = json.loads(completed.read_text())
                    if experiment_config(saved["config"]) != experiment_config(config):
                        raise ValueError(f"Existing run settings differ: {output}; use another preset/output")
                    print(f"Reuse completed training: {output}", flush=True)
                else:
                    command = base_command + ["train", "--config", str(plan)]
                    if args.resume and (output / "last.pt").exists():
                        command.append("--resume")
                    run(command)
            if args.train_only:
                continue
            step_counts = ((args.teacher_steps or config["eval"]["steps"]) if method == "teacher" else
                           [config["eval"]["comparison_steps"]] if "comparison_steps" in config["eval"] else
                           config["train"]["stages"])
            for steps in step_counts:
                result = results_dir / f"{method}_seed{seed}_{steps}steps.result.json"
                if result.exists():
                    saved = json.loads(result.read_text())
                    if saved.get("config_hash") not in (None, config_fingerprint(config)):
                        raise ValueError(f"Existing result has different configuration: {result}")
                    for key, expected in (("num_samples", config["eval"]["num_samples"]),
                                          ("num_real", config["eval"]["num_real"]),
                                          ("eval_seed", config["eval"]["seed"]),
                                          ("precision", config["precision"])):
                        if saved.get(key) != expected:
                            raise ValueError(f"Existing result has different {key}: {result}")
                    print(f"Reuse result: {result}", flush=True)
                    continue
                checkpoint = "teacher" if method == "teacher" else f"{config['output_dir']}/stage_{steps}"
                command = base_command + ["evaluate", "--config", str(plan), "--checkpoint", checkpoint,
                                          "--steps", str(steps), "--result-dir", str(results_dir)]
                if args.preset == "smoke":
                    command.append("--skip-metrics")
                run(command)
    print(f"\nSweep complete. Result folder: {results_dir}", flush=True)


if __name__ == "__main__":
    main()
