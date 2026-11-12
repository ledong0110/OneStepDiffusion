#!/usr/bin/env python3
"""Detached proposal experiments with a frozen DMD reference and full evaluation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import yaml

from few_step_diffusion.config import config_fingerprint, load_config
from few_step_diffusion.dmd_conflict import MODES, POLICY, refinement_config
from few_step_diffusion import dmd_noise_agreement as noise_agreement
from few_step_diffusion import dmd_response as response_alignment
from few_step_diffusion.runtime import atomic_json, experiment_lock, sha256
from launch_scale50k import process_matches, process_record

ROOT = Path(__file__).resolve().parents[1]
STUDY = "dmd_conflict"
STUDIES = {"dmd_conflict": {"modes": ("project", "norm_match"), "memory_cap_gib": None},
           "dmd_block_conflict": {"modes": ("block_project", "block_norm_match"), "memory_cap_gib": 15},
           "dmd_noise_agreement": {"modes": tuple(noise_agreement.MODES), "memory_cap_gib": 15},
           "dmd_pair_agreement": {"modes": tuple(noise_agreement.PAIR_MODES), "memory_cap_gib": 15},
           "dmd_response_strength": {"modes": ("angular_10", "angular_30", "angular_50"), "memory_cap_gib": 15},
           "dmd_response_final": {"modes": (), "memory_cap_gib": 15}}
PLAN = ROOT/"configs/dmd_conflict.json"
RUNS = ROOT/"runs/sd15/dmd_conflict"



def now():
    return datetime.now(timezone.utc).isoformat()


def started_process_record(child, expected_command):
    """Do not pin the transient empty argv between fork and exec."""
    for _ in range(100):
        if child.poll() is not None:
            return None
        try:
            record = process_record(child.pid)
        except (FileNotFoundError, ProcessLookupError):
            record = None
        if record and record["command"] == expected_command:
            return record
        time.sleep(.01)
    raise RuntimeError(f"Could not verify launched process {child.pid}")


def preserve_milestone(output, expected_config_hash, steps=(250, 500)):
    """Archive a complete rolling checkpoint using the same opened inode.

    No gradients or RNG draws are performed. A different snapshot at an already
    archived step is an error, never silently overwritten.
    """
    import torch
    source = output/"last.pt"
    if not source.exists():
        return None
    with source.open("rb") as handle:
        state = torch.load(handle, map_location="cpu", weights_only=False)
        if config_fingerprint(state["config"]) != expected_config_hash:
            raise ValueError("Checkpoint archive configuration differs from the scheduled run")
        step = state["global_step"]
        if step not in steps:
            return None
        if state["stage_index"] != 0 or state["stage_step"] != step:
            raise ValueError("Invalid one-step checkpoint counter for archive")
        folder = output/"checkpoint_previews"/f"step_{step:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temporary:
            copied = Path(temporary)/"checkpoint.pt"
            handle.seek(0)
            with copied.open("wb") as destination:
                shutil.copyfileobj(handle, destination)
            digest = sha256(copied)
            archived = folder/"checkpoint.pt"
            if archived.exists() and sha256(archived) != digest:
                raise ValueError("Refusing to replace a different checkpoint at the same update")
            if not archived.exists():
                copied.replace(archived)
        atomic_json(folder/"archive.json", {"step": step, "sha256": digest,
            "purpose": "Preserved training checkpoint; not an evaluated result"})
        return archived


def response_final_design(root, fraction, reason):
    """Pin completed pilot evidence without automatically choosing a winner."""
    if fraction not in (.1, .3, .5) or not isinstance(reason, str) or not reason.strip():
        raise ValueError("Final response training requires a selected strength and an evidence-based rationale")
    study = "dmd_response_strength"
    plan_path = root/"configs"/f"{study}.json"
    if not plan_path.exists():
        raise ValueError("Final response training requires the completed strength pilot")
    refresh_results(root, study)
    plan = json.loads(plan_path.read_text())
    comparison = json.loads((root/"runs/sd15"/study/"comparison.json").read_text())
    records = {row["method"]: row for row in comparison["results"]}
    from build_assignment2_report import completed_preview
    from analyze_teacher_responses import completed_source
    evidence = []
    for job in plan["jobs"]:
        result = records.get(job["method"])
        if not result or result.get("diversity_lpips_alex_mean") is None:
            raise ValueError("Final response training requires full quality and diversity results for every pilot arm")
        completed_source(root, study, job["key"])
        preview = completed_preview(root, job, result, comparison["fixed_reference"], step=plan["additional_updates"])
        if preview is None:
            raise ValueError("Final response training requires native samples from every evaluated pilot checkpoint")
        baseline, sample = preview["records"]
        if any(baseline.get(key) != sample.get(key) for key in
               ("sample_prompts", "sample_image_ids", "eval_seed", "evaluation_prompt_hash")):
            raise ValueError("Final response selection needs matching prompts and noise across native samples")
        path = root/job["result_dir"]/f"{job['method']}_seed{plan['seed']}_1steps.result.json"
        evidence.append({"method": job["method"], "result_file": str(path.relative_to(root)),
            "result_sha256": sha256(path), "model_identity": result["model_identity"],
            "checkpoint_sha256": preview["checkpoint_sha256"], "fid": result["fid"],
            "clip_score_mean": result["clip_score_mean"], "diversity_lpips_alex_mean": result["diversity_lpips_alex_mean"]})
    selected = response_alignment.MODES[f"angular_{round(fraction*100)}"]
    if selected not in records or len(evidence) != 3:
        raise ValueError("Final response selection requires all three predeclared pilot strengths")
    return {"policy": "response-final-design-from-completed-pilot-v1", "pilot_study": study,
        "pilot_plan_sha256": sha256(plan_path), "selected_pilot_method": selected,
        "gradient_fraction": fraction, "reason": reason.strip(), "pilot_evidence": evidence,
        "selection_scope": "exploratory-validation-selection; not an independent test set"}


def prepare(gpus, max_parallel, min_free_gib, updates=500, *, response_fraction=None, selection_reason=None):
    final_design = (response_final_design(ROOT, response_fraction, selection_reason)
                    if STUDY == "dmd_response_final" else None)
    if STUDY != "dmd_response_final" and (response_fraction is not None or selection_reason is not None):
        raise ValueError("Response selection options apply only to the separate final study")
    parent = ROOT/"runs/sd15/dmd_diversity_study/dmd_perceptual_seed42"
    parent_config = load_config(parent/"config.yaml")
    saved = json.loads((parent/"run.json").read_text())
    metadata = json.loads((parent/"stage_1/metadata.json").read_text())
    if saved.get("status") != "complete" or metadata["cumulative_training_updates"] != 36000:
        raise ValueError("Experiment requires the completed 36k DMD+LPIPS control")
    if config_fingerprint(saved["config"]) != config_fingerprint(parent_config):
        raise ValueError("Parent run/config mismatch")
    parent_hash = sha256(parent/"last.pt")
    cache = Path(parent_config["train"]["dmd_pair_cache_dir"])
    pairs = json.loads((cache/"teacher_pairs.json").read_text())
    if sha256(cache/"teacher_pairs.safetensors") != pairs["weights_sha256"]:
        raise ValueError("Shared teacher-pair bank failed integrity verification")
    jobs = []
    use_noise = STUDY in ("dmd_noise_agreement", "dmd_pair_agreement")
    use_response = STUDY in ("dmd_response_strength", "dmd_response_final")
    independent_pairs = STUDY == "dmd_pair_agreement"
    factory = (response_alignment.refinement_config if use_response else
               noise_agreement.refinement_config if use_noise else refinement_config)
    methods = (response_alignment.MODES if use_response else noise_agreement.PAIR_MODES if independent_pairs else
               noise_agreement.MODES if use_noise else MODES)
    modes = (tuple(f"{mode}_{round(response_fraction*100)}" for mode in ("angular", "magnitude", "vector"))
             if final_design else STUDIES[STUDY]["modes"])
    for mode in modes:
        output = RUNS/mode
        options = {"noise_pairs": 2} if independent_pairs else {}
        if use_response:
            calibration = ROOT/"artifacts/dmd_teacher_response/gradient_calibration.json"
            options.update(calibration_file=str(calibration.relative_to(ROOT)), calibration_sha256=sha256(calibration))
        config = factory(parent_config, parent, str(output.relative_to(ROOT)), mode, parent_hash,
                         updates=updates, **options)
        plan = ROOT/"configs/generated"/STUDY/f"{mode}.yaml"
        if plan.exists() and config_fingerprint(load_config(plan)) != config_fingerprint(config):
            raise ValueError(f"Refusing to replace a different experiment plan: {plan}")
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(yaml.safe_dump(config, sort_keys=False))
        jobs.append({"key": mode, "method": methods[mode], "config": str(plan.relative_to(ROOT)),
                     "run_dir": str(output.relative_to(ROOT)),
                     "result_dir": f"results/sd15/{STUDY}",
                     "config_hash": config_fingerprint(config)})
    reference = ROOT/"results/sd15/dmd_diversity_study/dmd_perceptual_seed42_1steps.result.json"
    policy = (response_alignment.POLICY if use_response else noise_agreement.PAIR_POLICY if independent_pairs else
              noise_agreement.POLICY if use_noise else POLICY)
    manifest = {"policy": policy, "seed": 42, "additional_updates": updates,
        "cumulative_training_updates": 36000+updates, "parent_training_updates": 36000,
        "parent_checkpoint_sha256": parent_hash, "teacher_pair_weights_sha256": pairs["weights_sha256"],
        "gpus": gpus, "max_parallel": max_parallel, "min_free_gpu_gib": min_free_gib, "jobs": jobs,
        "reference_baseline": {"result_file": str(reference.relative_to(ROOT)),
            "result_sha256": sha256(reference),
            "model_identity": sha256(parent/"stage_1/adapter.safetensors")}}
    if STUDIES[STUDY]["memory_cap_gib"] is not None:
        manifest["max_gpu_memory_gib"] = STUDIES[STUDY]["memory_cap_gib"]
    if final_design:
        manifest["final_design"] = final_design
    if PLAN.exists() and json.loads(PLAN.read_text()) != manifest:
        raise ValueError("Existing experiment plan differs; resume it rather than replacing it")
    atomic_json(PLAN, manifest)
    RUNS.mkdir(parents=True, exist_ok=True)
    return manifest


def refresh_results(root=ROOT, study="dmd_conflict"):
    from few_step_diffusion.reporting import collect_results
    manifest = json.loads((root/"configs"/f"{study}.json").read_text())
    folder = root/"results/sd15"/study
    files = [folder/f"{job['method']}_seed42_1steps.result.json" for job in manifest["jobs"]]
    records, _ = collect_results(folder, root/"schemas/result.schema.json",
                                 result_files=[p for p in files if p.exists()])
    reference_spec = manifest["reference_baseline"]
    reference_path = root/reference_spec["result_file"]
    if sha256(reference_path) != reference_spec["result_sha256"]:
        raise ValueError("Frozen DMD baseline result changed")
    reference_records, _ = collect_results(reference_path.parent, root/"schemas/result.schema.json",
                                           result_files=[reference_path])
    reference = reference_records[0]
    if (reference["method"] != "dmd_perceptual" or reference["num_samples"] != 5000 or
            reference["num_real"] != 5000 or reference["cumulative_training_updates"] != 36000 or
            config_fingerprint(reference["resolved_config"]) != reference["config_hash"] or
            reference["model_identity"] != reference_spec["model_identity"]):
        raise ValueError("Frozen baseline is not the completed 36k DMD+LPIPS result")
    labels = {"project": "DMD + conflict projection",
              "norm_match": "Proposal ablation: norm-matched LPIPS",
              "block_project": "DMD + module projection", "block_norm_match": "Module-norm ablation",
              "agreement": "Spatial noise agreement (candidate)",
              "paired_mean": "Ablation: paired mean only",
              "shifted_agreement": "Ablation: spatially shifted agreement"}
    tex_labels = {"agreement": "Spatial agreement", "paired_mean": "Paired mean",
                  "shifted_agreement": "Shifted map"}
    for key, (mode, fraction) in response_alignment.SETTINGS.items():
        labels[key] = f"{mode.capitalize()} response ({fraction:.0%} initial gradient)"
        tex_labels[key] = f"{mode.capitalize()} {fraction:.0%}".replace("%", r"\%")
    if manifest["policy"] == noise_agreement.PAIR_POLICY:
        labels.update(agreement="Independent-pair agreement (candidate)",
                      paired_mean="Ablation: four-probe mean", shifted_agreement="Ablation: shifted pair agreement")
        tex_labels.update(agreement="Pair agreement", paired_mean="Four-probe mean", shifted_agreement="Shifted pairs")
    by_method = {r["method"]: r for r in records}
    rows = [f"| DMD + LPIPS (fixed reference) | 36,000 | {reference['fid']:.2f} | "
            f"{reference['inception_score_mean']:.2f} | {reference['clip_score_mean']:.2f} |"]
    tex_rows = [f"DMD + LPIPS (fixed) & 36,000 & {reference['fid']:.2f} & "
                f"{reference['inception_score_mean']:.2f} & {reference['clip_score_mean']:.2f} "+chr(92)*2]
    for job in manifest["jobs"]:
        record = by_method.get(job["method"])
        if record:
            config = load_config(root/job["config"])
            comparison_keys = ("evaluation_prompt_hash", "reference_manifest_hash", "eval_seed",
                "num_samples", "num_real", "metric_implementation", "clip_model_identity",
                "sampling_steps", "nfe", "precision", "lora_rank", "teacher_guidance_scale",
                "teacher_identity", "teacher_pair_weights_sha256")
            if any(record.get(k) != reference.get(k) for k in comparison_keys):
                raise ValueError("Proposal evaluation differs from the frozen DMD reference protocol")
            if "diversity_lpips_alex_mean" in record:
                validate_diversity(record, reference)
            if (record["config_hash"] != job["config_hash"] or record["num_samples"] != 5000 or
                    config_fingerprint(record["resolved_config"]) != job["config_hash"] or
                    record["num_real"] != 5000 or record["cumulative_training_updates"] != manifest.get("cumulative_training_updates", 36500) or
                    record["model_identity"] != sha256(root/job["run_dir"]/"stage_1/adapter.safetensors") or
                    record["refinement_parent_checkpoint_sha256"] != manifest["parent_checkpoint_sha256"] or
                    record["teacher_pair_weights_sha256"] != manifest["teacher_pair_weights_sha256"] or
                    record["eval_seed"] != config["eval"]["seed"]):
                raise ValueError("Completed evaluation differs from the matched experiment")
            values = f"{record['fid']:.2f} | {record['inception_score_mean']:.2f} | {record['clip_score_mean']:.2f}"
            tex_values = f"{record['fid']:.2f} & {record['inception_score_mean']:.2f} & {record['clip_score_mean']:.2f}"
        else:
            values, tex_values = "pending | pending | pending", "-- & -- & --"
        total = f"{manifest.get('cumulative_training_updates', 36500):,}"
        rows.append(f"| {labels[job['key']]} | {total} | {values} |")
        tex_rows.append(tex_labels.get(job["key"], labels[job["key"]])+f" & {total} & "+tex_values+" "+chr(92)*2)
    title = ("Selected response method and component ablations" if study == "dmd_response_final" else
             "Direction-only response strength pilot" if manifest["policy"] == response_alignment.POLICY else
             "Independent-pair agreement experiments" if manifest["policy"] == noise_agreement.PAIR_POLICY else
             "Spatial noise-agreement experiments" if manifest["policy"] == noise_agreement.POLICY else
             "Earlier module-projection experiments" if study == "dmd_block_conflict" else
             "Earlier global-projection experiments")
    header = f"# {title}\n\n"
    if manifest["policy"] == noise_agreement.POLICY and (root/"configs/dmd_pair_agreement.json").exists():
        header += ("This completed two-probe pilot did not improve FID over the fixed DMD reference. "
                   "The follow-up is [independent-pair agreement](DMD_PAIR_AGREEMENT_RESULTS.md).\n\n")
    if manifest["policy"] == POLICY and (root/"configs/dmd_noise_agreement.json").exists():
        header += ("These are completed exploratory projection experiments, not the current proposal. "
                   "The subsequent spatial-noise pilot is recorded in [the spatial noise-agreement study]"
                   "(DMD_NOISE_AGREEMENT_RESULTS.md).\n\n")
    updates = manifest.get("additional_updates", 500)
    total = manifest.get("cumulative_training_updates", 36500)
    header += (
        "The existing best DMD+LPIPS result is retained as the fixed 36,000-update reference; "
        "additional baseline training was cancelled at the user's request. The proposal and its "
        f"variants start from that same checkpoint and receive {updates:,} additional updates ({total:,} total), "
        "rank 32, effective batch 16, teacher CFG 7.5 and one-step inference. "
        "Evaluation uses all 5,000 COCO validation images with the same prompts and noise. "
        f"The {updates:,}-update budget difference means a gain over the frozen reference alone cannot "
        "isolate the effect of the new update rule. ")
    if manifest["policy"] == response_alignment.POLICY:
        if study == "dmd_response_final":
            selected = manifest["final_design"]
            header += (
                "The selected angular response loss is compared with magnitude-only and full-vector "
                "response objectives. Each starts from the fixed 36k parent, not from a tuned pilot export. "
                f"All added losses use the same {selected['gradient_fraction']:.0%} median initial "
                "latent-gradient ratio, calibrated on training captions; the ratios are not maintained "
                "dynamically. Original DMD and its LPIPS term remain unchanged. "
                f"Exploratory selection rationale: {selected['reason']}\n\n")
        else:
            header += (
                "The added term aligns the direction of finite latent changes between two independent noises "
                "for the same caption. Original DMD and its LPIPS regression remain unchanged. "
                "Strengths are calibrated on 32 training captions to median initial latent-gradient ratios "
                "of 10%, 30% and 50% relative to DMD, not maintained dynamically during training. "
                "Cached teacher pairs and existing regression forwards are reused. Related response-matching "
                "work is credited; this is an exploratory extension, not a claim to invent response alignment.\n\n")
    elif manifest["policy"] in noise_agreement.POLICIES:
        description = ("All three variants use two independent antithetic pairs (four score probes). "
                       "Agreement is computed between the two pair means, after antithetic averaging. "
                       if manifest["policy"] == noise_agreement.PAIR_POLICY else
                       "All three variants use two opposite-noise score probes per generated latent. ")
        header += (description+
            "The candidate applies a spatial agreement map and preserves the paired mean's latent-gradient norm; "
            "the controls remove or spatially shift that map. LPIPS remains inherited from DMD. "
            "Training score-probe compute exceeds the fixed reference; inference remains one step. "
            "This is an exploratory, biased update rule, not a proven improvement or novelty claim.\n\n")
    else:
        header += "Projection is an asymmetric PCGrad-style application; quality evaluation is exploratory.\n\n"
    text = header+"| Method | Total updates | FID ↓ | IS ↑ | CLIP ↑ |\n|---|---:|---:|---:|---:|\n"+"\n".join(rows)+"\n"
    (root/f"{study.upper()}_RESULTS.md").write_text(text)
    generated = root/"report/generated"/f"{study}_results.tex"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text("\n".join([r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Method & Updates & FID $\downarrow$ & IS $\uparrow$ & CLIP $\uparrow$ \\",
        r"\midrule", *tex_rows, r"\bottomrule", r"\end{tabular}", ""]))
    atomic_json(root/"runs/sd15"/study/"comparison.json", {"policy": manifest["policy"], "completed": len(records),
        "fixed_reference": {k: reference[k] for k in ("method", "fid", "inception_score_mean",
            "clip_score_mean", "cumulative_training_updates", "model_identity", "diversity_lpips_alex_mean")},
        "best_fid_method_so_far": min([reference, *records], key=lambda r: r["fid"])["method"],
        "planned": len(manifest["jobs"]), "results": [{k: r.get(k) for k in
            ("method", "fid", "inception_score_mean", "clip_score_mean", "cumulative_training_updates",
             "model_identity", "diversity_lpips_alex_mean")}
            for r in records]})


def validate_diversity(record, reference):
    keys = ("diversity_protocol", "diversity_prompt_hash", "diversity_prompt_count",
            "diversity_seed_list", "diversity_samples_per_prompt", "diversity_lpips_identity",
            "diversity_generation_nfe")
    if any(record.get(key) is None or record[key] != reference.get(key) for key in keys):
        raise ValueError("Proposal diversity evaluation differs from the fixed reference protocol")
    scores = record.get("diversity_pairwise_scores_per_prompt", [])
    if (len(scores) != 128 or record["diversity_prompt_count"] != 128 or
            record["diversity_samples_per_prompt"] != 4 or
            not all(math.isfinite(value) and value >= 0 for value in scores) or
            not math.isclose(statistics.mean(scores), record["diversity_lpips_alex_mean"], abs_tol=1e-9) or
            not math.isclose(statistics.stdev(scores), record.get("diversity_lpips_alex_std", -1), abs_tol=1e-9)):
        raise ValueError("Diversity summary does not match all 128 per-prompt measurements")


def gpu_free():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                      "--format=csv,noheader,nounits"], text=True)
    return {int(row.split(",")[0]): float(row.split(",")[1])/1024
            for row in output.strip().splitlines()}


def phase_launch_args(args, phase, gpu):
    """The native preview sets CUDA visibility from its own explicit GPU option."""
    command = list(args)
    if phase in ("matched_qualitative_samples", "teacher_response_diagnostic"):
        command[command.index("--gpu")+1] = str(gpu)
    return command


def phase_memory_limits(manifest, phase):
    """Inference-only diagnostics need less memory than generator training."""
    caps = {"diversity_128_prompts_four_seeds": 10., "matched_qualitative_samples": 4.,
            "teacher_response_diagnostic": 8.}
    if phase in caps:
        return caps[phase], caps[phase]+1
    return manifest.get("max_gpu_memory_gib"), manifest["min_free_gpu_gib"]


def supervise():
    manifest = json.loads(PLAN.read_text())
    status_path = RUNS/"status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {"jobs": {}}
    state.update(pid=os.getpid(), status="running", started_at=now(), finished_at=None)
    lock = threading.RLock()
    reserved = {entry["gpu"] for job in manifest["jobs"]
                if (entry := state["jobs"].get(job["key"], {})).get("child")
                and "gpu" in entry and process_matches(entry["child"])}

    def update(key=None, **values):
        with lock:
            if key:
                state["jobs"].setdefault(key, {}).update(values)
            else:
                state.update(values)
            state["updated_at"] = now()
            atomic_json(status_path, state)

    def acquire_gpu(key, minimum):
        while True:
            with lock:
                free = gpu_free()
                candidates = [g for g in manifest["gpus"] if g not in reserved and
                              free.get(g, 0) >= minimum]
                if candidates and shutil.disk_usage(ROOT).free >= 8*2**30:
                    gpu = max(candidates, key=lambda g: free[g])
                    reserved.add(gpu)
                    update(key, status="running", gpu=gpu)
                    return gpu
                update(key, status="queued", phase="waiting_for_memory", free_gpu_gib=free,
                       required_free_gpu_gib=minimum)
            time.sleep(20)

    def refresh_report():
        with experiment_lock(RUNS/"result_refresh"):
            refresh_results(ROOT, STUDY)
            with (RUNS/"report.log").open("ab", buffering=0) as log:
                code = subprocess.call([sys.executable, "scripts/build_assignment2_report.py", "--package"],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            update(report_refresh="complete" if code == 0 else "failed", report_exit_code=code)
            if code:
                raise RuntimeError("Assignment report refresh failed; see report.log")

    def worker(job):
        key = job["key"]
        output, plan = ROOT/job["run_dir"], ROOT/job["config"]
        output.mkdir(parents=True, exist_ok=True)
        config = load_config(plan)
        if config_fingerprint(config) != job["config_hash"]:
            raise ValueError("Experiment configuration changed after scheduling")
        seen_checkpoint = None

        def archive_if_changed():
            nonlocal seen_checkpoint
            checkpoint = output/"last.pt"
            if checkpoint.exists():
                info = checkpoint.stat()
                identity = (info.st_ino, info.st_mtime_ns, info.st_size)
                if identity != seen_checkpoint:
                    budget = manifest["additional_updates"]
                    preserve_milestone(output, job["config_hash"], steps=(budget//2, budget))
                    seen_checkpoint = identity

        # A restarted supervisor waits for an already-running child before resuming.
        previous = state["jobs"].get(key, {}).get("child")
        if previous and process_matches(previous):
            previous_gpu = state["jobs"][key].get("gpu")
            with lock:
                if previous_gpu is not None:
                    reserved.add(previous_gpu)
            update(key, status="running", phase="training" if "train" in previous["command"]
                   else "waiting_for_existing_worker", adopted_worker=True, error=None, finished_at=None)
            while process_matches(previous):
                archive_if_changed()
                time.sleep(20)
            with lock:
                reserved.discard(previous_gpu)
        update(key, status="queued", phase="waiting_for_gpu", child=None, error=None, finished_at=None)
        # Acquire lazily for the next unfinished phase. Finished experiments do
        # not need a GPU merely to validate their files and refresh the report.
        gpu = None
        env = dict(os.environ, PYTHONUNBUFFERED="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

        def run(args, phase):
            nonlocal gpu
            # External workers can claim memory between training and evaluation.
            # Reacquire capacity instead of assuming a study-local reservation
            # protects GPU memory across process boundaries.
            cap, minimum = phase_memory_limits(manifest, phase)
            if gpu is None or gpu_free().get(gpu, 0) < minimum:
                with lock:
                    reserved.discard(gpu)
                update(key, status="queued", phase="waiting_for_phase_memory", next_phase=phase)
                gpu = acquire_gpu(key, minimum)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            if cap is None:
                env.pop("FSD_MAX_GPU_MEMORY_GIB", None)
            else:
                env["FSD_MAX_GPU_MEMORY_GIB"] = str(cap)
            command = phase_launch_args(args, phase, gpu)
            with (output/"launch.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                update(key, phase=phase, next_phase=None, gpu_memory_cap_gib=cap,
                       phase_min_free_gpu_gib=minimum, child=started_process_record(child, command))
                while True:
                    try:
                        code = child.wait(timeout=20)
                        break
                    except subprocess.TimeoutExpired:
                        archive_if_changed()
                archive_if_changed()
            update(key, child=None)
            if code:
                raise RuntimeError(f"{phase} exited {code}; see {output/'launch.log'}")

        try:
            with experiment_lock(output):
                native = [sys.executable, "-m", "few_step_diffusion.cli"]
                if (output/"run.json").exists():
                    completed = json.loads((output/"run.json").read_text())
                    if (completed.get("status") != "complete" or
                            config_fingerprint(completed["config"]) != job["config_hash"]):
                        raise ValueError("Finished experiment run differs from the scheduled recipe")
                else:
                    args = native+["train", "--config", str(plan)]
                    if (output/"last.pt").exists():
                        args.append("--resume")
                    run(args, "training")
                result = ROOT/job["result_dir"]/f"{job['method']}_seed42_1steps.result.json"
                if not result.exists():
                    run(native+["evaluate", "--config", str(plan), "--checkpoint", str(output/"stage_1"),
                        "--steps", "1", "--result-dir", str(result.parent)], "full_5000_image_evaluation")
                from few_step_diffusion.reporting import collect_results
                records, _ = collect_results(result.parent, ROOT/"schemas/result.schema.json", result_files=[result])
                record = records[0]
                if record["config_hash"] != job["config_hash"]:
                    raise ValueError("Experiment evaluation/config mismatch")
                if "diversity_lpips_alex_mean" not in record:
                    run([sys.executable, str(Path(__file__).resolve()), "--study", STUDY, "--diversity", key],
                        "diversity_128_prompts_four_seeds")
                budget = manifest["additional_updates"]
                preview = output/"checkpoint_previews"/f"step_{budget:06d}"/"samples.json"
                if not preview.exists():
                    preserve_milestone(output, job["config_hash"], steps=(budget,))
                    run([sys.executable, "scripts/preview_dmd_refinement.py", "--mode", key,
                         "--study", STUDY, "--step", str(budget), "--gpu", str(gpu)], "matched_qualitative_samples")
                if STUDY == "dmd_response_final" and not (output/"response_diagnostic.json").exists():
                    run([sys.executable, "scripts/analyze_teacher_responses.py", "--study", STUDY,
                         "--variant", key, "--gpu", str(gpu)], "teacher_response_diagnostic")
                update(key, status="complete", phase="complete", finished_at=now())
            refresh_report()
        except Exception as exc:
            update(key, status="failed", phase="failed", error=str(exc), finished_at=now())
        finally:
            with lock:
                reserved.discard(gpu)

    for job in manifest["jobs"]:
        state["jobs"].setdefault(job["key"], {"status": "queued", "method": job["method"]})
    update()
    with ThreadPoolExecutor(max_workers=manifest["max_parallel"]) as pool:
        futures = [pool.submit(worker, job) for job in manifest["jobs"]]
        for future in futures:
            future.result()
    update(status="complete" if all(state["jobs"][j["key"]]["status"] == "complete" for j in manifest["jobs"])
           else "failed", finished_at=now())


def main():
    global STUDY, PLAN, RUNS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=tuple(STUDIES), default="dmd_conflict")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=26)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--response-fraction", type=float, choices=(.1, .3, .5))
    parser.add_argument("--selection-reason")
    parser.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--diversity", help=argparse.SUPPRESS)
    args = parser.parse_args()
    STUDY = args.study
    PLAN = ROOT/"configs"/f"{STUDY}.json"
    RUNS = ROOT/"runs/sd15"/STUDY
    os.chdir(ROOT)
    if args.diversity:
        from few_step_diffusion.diversity_evaluation import evaluate_diversity
        job = next(j for j in json.loads(PLAN.read_text())["jobs"] if j["key"] == args.diversity)
        config = load_config(ROOT/job["config"])
        output = ROOT/job["run_dir"]
        return evaluate_diversity(config, str(output/"stage_1"), output/"diversity_eval",
            ROOT/job["result_dir"]/f"{job['method']}_seed42_1steps.result.json")
    if args.supervise:
        with experiment_lock(RUNS/"supervisor"):
            atomic_json(RUNS/"process.json", process_record(os.getpid()))
            return supervise()
    if (RUNS/"process.json").exists() and process_matches(json.loads((RUNS/"process.json").read_text())):
        print(f"Experiment already running: {RUNS/'status.json'}")
        return
    minimum = STUDIES[STUDY]["memory_cap_gib"]+1 if STUDIES[STUDY]["memory_cap_gib"] else 26
    if args.max_parallel < 1 or args.min_free_gib < minimum or not args.gpus:
        raise ValueError(f"Use a positive worker count and at least {minimum} GiB free memory per new run")
    if STUDY == "dmd_response_final" and args.updates is None:
        raise ValueError("Choose an explicit final training budget; the pilot budget is not assumed")
    manifest = prepare(args.gpus, args.max_parallel, args.min_free_gib,
        500 if args.updates is None else args.updates,
        response_fraction=args.response_fraction, selection_reason=args.selection_reason)
    refresh_results(ROOT, STUDY)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    with (RUNS/"supervisor.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--study", STUDY, "--supervise"],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    command = [sys.executable, str(Path(__file__).resolve()), "--study", STUDY, "--supervise"]
    record = started_process_record(child, command)
    if record is None:
        raise RuntimeError("Experiment supervisor exited at startup; inspect supervisor.log")
    atomic_json(RUNS/"process.json", record)
    print(f"Experiment supervisor PID {child.pid}; status: {RUNS/'status.json'}")


if __name__ == "__main__":
    main()
