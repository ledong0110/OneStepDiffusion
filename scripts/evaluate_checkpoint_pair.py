#!/usr/bin/env python3
"""Evaluate saved baseline/proposed SD checkpoints with a matched COCO protocol."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from launch_scale50k import ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def supervise(folder):
    manifest = json.loads((folder/"launch.json").read_text())
    os.environ["CUDA_VISIBLE_DEVICES"] = manifest["gpu"]
    from few_step_diffusion.config import config_fingerprint, load_config
    from few_step_diffusion.reporting import collect_results
    from few_step_diffusion.runtime import atomic_json, experiment_lock, sha256
    state = {"pid": os.getpid(), "status": "running", "started_at": now(),
             "gpu": manifest["gpu"], "steps": manifest["steps"], "workers": {}}
    mutex = threading.Lock()

    def update(method, **values):
        with mutex:
            state["workers"].setdefault(method, {}).update(values)
            state["updated_at"] = now()
            atomic_json(folder/"status.json", state)

    def evaluate(job):
        method = job["method"]
        config = load_config(job["config"])
        result = Path(manifest["result_dir"])/f"{method}_seed42_{manifest['steps']}steps.result.json"
        try:
            with experiment_lock(folder/f"{method}_lock"):
                if result.exists():
                    saved = json.loads(result.read_text())
                    if (saved.get("config_hash") != config_fingerprint(config) or
                            saved.get("model_identity") != job["model_sha256"] or
                            saved.get("status") != "complete"):
                        raise ValueError(f"Existing evaluation differs: {result}")
                else:
                    command = [sys.executable, "-m", "few_step_diffusion.cli", "evaluate", "--config",
                        job["config"], "--checkpoint", job["checkpoint"], "--steps", str(manifest["steps"]),
                        "--result-dir", manifest["result_dir"]]
                    with (folder/f"{method}.log").open("ab", buffering=0) as log:
                        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
                        update(method, status="running", pid=child.pid, command=command)
                        code = child.wait()
                    if code:
                        raise RuntimeError(f"{method} evaluation exited {code}; see {folder/f'{method}.log'}")
                saved = json.loads(result.read_text())
                if sha256(Path(job["checkpoint"])/"adapter.safetensors") != job["model_sha256"]:
                    raise ValueError("Checkpoint changed during evaluation")
                saved["evaluation_context"] = {
                    "physical_gpu": manifest["gpu"], "concurrent_training": True,
                    "latency_comparable": False,
                    "note": "FID comparison uses matched images; GPU timing was measured under shared load."}
                atomic_json(result, saved)
                update(method, status="complete", result=str(result), fid=saved["fid"],
                       clip_score_mean=saved["clip_score_mean"], finished_at=now())
        except BaseException as exc:
            update(method, status="failed", error=str(exc))
            raise

    errors = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(evaluate, job) for job in manifest["jobs"]]):
            try:
                future.result()
            except BaseException as exc:
                errors.append(str(exc))
    if errors:
        state.update(status="failed", errors=errors)
        atomic_json(folder/"status.json", state)
        raise RuntimeError("Checkpoint evaluation failed")
    records, _ = collect_results(manifest["result_dir"], ROOT/"schemas/result.schema.json")
    by_method = {r["method"]: r for r in records}
    baseline, proposed = by_method["baseline"]["fid"], by_method["proposed"]["fid"]
    with (Path(manifest["result_dir"])/"comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "sampling_steps", "nfe", "training_updates", "num_samples", "num_real", "fid", "clip_score_mean"])
        for method in ("baseline", "proposed"):
            r = by_method[method]
            writer.writerow([r[k] for k in ("method", "sampling_steps", "nfe", "training_updates", "num_samples", "num_real", "fid", "clip_score_mean")])
    state.update(status="complete", finished_at=now(), baseline_fid=baseline, proposed_fid=proposed,
                 proposed_minus_baseline=proposed-baseline)
    atomic_json(folder/"status.json", state)
    print(json.dumps(state, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--gpu", default="3")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--run-root", default="runs/sd15/scale50k")
    p.add_argument("--result-dir")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)
    if args.worker:
        supervise(args.worker)
        return
    import yaml
    from few_step_diffusion.config import load_config, validate
    from few_step_diffusion.runtime import atomic_json, sha256
    result_dir = args.result_dir or f"results/sd15/scale50k_{args.steps}steps"
    jobs, checkpoints, configs = [], [], []
    for method in ("baseline", "proposed"):
        run = Path(args.run_root)/f"{method}_seed42"
        checkpoint = run/f"stage_{args.steps}"
        metadata = json.loads((checkpoint/"metadata.json").read_text())
        config = load_config(run/"config.yaml")
        config["eval"].update(steps=[args.steps], comparison_steps=args.steps,
            num_samples=args.samples, num_real=args.samples, batch_size=args.batch_size)
        validate(config)
        checkpoints.append(metadata)
        configs.append(config)
        jobs.append({"method": method, "checkpoint": str(checkpoint),
                     "model_sha256": sha256(checkpoint/"adapter.safetensors")})
    for key in ("global_step", "effective_batch_size", "trainable_parameters", "teacher_identity",
                "training_prompt_hash", "training_prompt_count", "timesteps", "lora_rank", "teacher_guidance_scale"):
        if checkpoints[0][key] != checkpoints[1][key]:
            raise ValueError(f"Checkpoints differ in {key}")
    for key in ("seed", "steps", "comparison_steps", "num_samples", "num_real", "reference_split", "batch_size"):
        if configs[0]["eval"][key] != configs[1]["eval"][key]:
            raise ValueError(f"Evaluation differs in {key}")
    if configs[0]["precision"] != configs[1]["precision"]:
        raise ValueError("Evaluation precision differs")
    manifest = {"gpu": args.gpu, "steps": args.steps, "samples": args.samples,
                "training_updates": checkpoints[0]["global_step"], "result_dir": result_dir, "jobs": jobs}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    pointer = ROOT/f"artifacts/launches/latest_sd15_step{args.steps}_eval.txt"
    if pointer.is_file():
        status = Path(pointer.read_text().strip())/"status.json"
        if status.is_file():
            old = json.loads(status.read_text())
            proc = Path(f"/proc/{old.get('pid')}/cmdline")
            if old.get("status") in ("running", "starting") and proc.exists() and b"evaluate_checkpoint_pair.py" in proc.read_bytes():
                raise RuntimeError("This checkpoint-pair evaluation is already active")
    folder = ROOT/"artifacts/launches"/(f"sd15_step{args.steps}_eval_"+
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    folder.mkdir(parents=True)
    for job, config in zip(jobs, configs, strict=True):
        plan = folder/f"{job['method']}.yaml"
        plan.write_text(yaml.safe_dump(config, sort_keys=False))
        job["config"] = str(plan)
    atomic_json(folder/"launch.json", manifest)
    with (folder/"output.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(folder)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    atomic_json(folder/"status.json", {"pid": child.pid, "status": "starting"})
    pointer.write_text(str(folder)+"\n")
    for name, target in ((f"step{args.steps}_eval_status.json", "status.json"),
                         (f"step{args.steps}_baseline_eval.log", "baseline.log"),
                         (f"step{args.steps}_proposed_eval.log", "proposed.log")):
        link = Path(args.run_root)/name
        link.unlink(missing_ok=True)
        link.symlink_to(folder/target)
    print(json.dumps({"pid": child.pid, "folder": str(folder), **manifest}, indent=2))


if __name__ == "__main__":
    main()
