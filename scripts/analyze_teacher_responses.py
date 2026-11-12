#!/usr/bin/env python3
"""Frozen diagnostic of finite, same-caption noise displacements.

The cached noises are independent, not nearby perturbations. These are secants,
not Jacobian estimates, and their alignment is not an image-quality metric.
Response matching has prior art (e.g. GAD, arXiv:2606.01651); this script is an
empirical diagnostic, not a new training method or a novelty claim.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def completed_source(root, study, variant):
    """Resolve a finished, evaluated export instead of reading a live checkpoint."""
    from few_step_diffusion.config import config_fingerprint, load_config
    from few_step_diffusion.reporting import collect_results
    from few_step_diffusion.runtime import sha256
    plan = json.loads((root/"configs"/f"{study}.json").read_text())
    jobs = [job for job in plan["jobs"] if job["key"] == variant]
    if len(jobs) != 1:
        raise ValueError("Diagnostic variant is absent or ambiguous in the pinned study")
    job = jobs[0]
    run = root/job["run_dir"]
    if not (run/"run.json").exists():
        raise ValueError("Response diagnostic waits for completed training and full evaluation")
    finished = json.loads((run/"run.json").read_text())
    config = load_config(root/job["config"])
    if (finished.get("status") != "complete" or
            finished.get("global_step") != plan["additional_updates"] or
            config_fingerprint(config) != job["config_hash"] or
            config_fingerprint(finished["config"]) != job["config_hash"]):
        raise ValueError("Response diagnostic needs the completed scheduled training run")
    result_path = root/job["result_dir"]/f"{job['method']}_seed{plan['seed']}_1steps.result.json"
    if not result_path.exists():
        raise ValueError("Response diagnostic waits for completed training and full evaluation")
    records, _ = collect_results(result_path.parent, root/"schemas/result.schema.json", result_files=[result_path])
    result = records[0]
    if (result["method"] != job["method"] or result["config_hash"] != job["config_hash"] or
            result["num_samples"] != 5000 or result["num_real"] != 5000 or
            result["cumulative_training_updates"] != plan["cumulative_training_updates"] or
            result["model_identity"] != sha256(run/"stage_1/adapter.safetensors") or
            result["refinement_parent_checkpoint_sha256"] != plan["parent_checkpoint_sha256"]):
        raise ValueError("Response diagnostic model differs from the completed full evaluation")
    return run, config, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--start-prompt", type=int, default=128)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--study", choices=("dmd_response_strength", "dmd_response_final"))
    parser.add_argument("--variant", help="Completed study arm to diagnose instead of the fixed reference")
    parser.add_argument("--calibrate-gradients", action="store_true",
                        help="Measure response-loss gradient sizes against frozen DMD score gradients")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.start_prompt < 0 or args.count < 1 or args.start_prompt+args.count > 2500:
        parser.error("Select a nonempty range within the 2,500 cached training captions")
    if bool(args.study) != bool(args.variant) or (args.study and args.calibrate_gradients):
        parser.error("Provide study and variant together; gradient calibration uses only the fixed reference")
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from tqdm.auto import tqdm
    from few_step_diffusion.config import load_config, config_fingerprint
    from few_step_diffusion.runtime import atomic_json, sha256
    from few_step_diffusion.sd_data import caption_records, manifest_hash
    from few_step_diffusion.sd_models import (
        SDBundle, adapter_state, generate_sd_latents, restore_export,
    )

    torch.set_num_threads(2)
    parent = ROOT/"runs/sd15/dmd_diversity_study/dmd_perceptual_seed42"
    config = load_config(parent/"config.yaml")
    plan = json.loads((ROOT/"configs/dmd_pair_agreement.json").read_text())
    source = parent/"stage_1/adapter.safetensors"
    identity = sha256(source)
    if identity != plan["reference_baseline"]["model_identity"]:
        raise ValueError("Frozen reference adapter identity changed")
    checkpoint_sha = sha256(parent/"last.pt")
    if checkpoint_sha != plan["parent_checkpoint_sha256"]:
        raise ValueError("Frozen reference training state changed")
    reference_identity = identity
    source_run = parent
    if args.study:
        source_run, config, candidate_plan = completed_source(ROOT, args.study, args.variant)
        if (candidate_plan["parent_checkpoint_sha256"] != checkpoint_sha or
                candidate_plan["reference_baseline"]["model_identity"] != reference_identity):
            raise ValueError("Response diagnostic study has a different fixed reference")
        source = source_run/"stage_1/adapter.safetensors"
        identity = sha256(source)
    source_checkpoint_sha = sha256(source_run/"last.pt")
    output = args.output or (source_run/"response_diagnostic.json" if args.study else
                             ROOT/"artifacts/dmd_teacher_response/frozen_reference.json")
    if output.exists():
        saved = json.loads(output.read_text())
        if (saved.get("policy") != "frozen-teacher-finite-response-diagnostic-v1" or
                saved.get("model_identity") != identity or saved.get("config_hash") != config_fingerprint(config) or
                saved.get("parent_checkpoint_sha256") != checkpoint_sha or
                saved.get("teacher_pair_weights_sha256") != plan["teacher_pair_weights_sha256"] or
                saved.get("start_prompt") != args.start_prompt or saved.get("prompt_count") != args.count or
                bool(saved.get("gradient_calibration")) != args.calibrate_gradients or
                saved.get("optimizer_updates") != 0 or saved.get("weights_unchanged") is not True):
            raise ValueError("Refusing to overwrite a diagnostic from a different model or protocol")
        print(f"Verified existing response diagnostic: {output}")
        return
    rows = caption_records(config, "train", config["data"]["train_limit"])
    selected = rows[args.start_prompt:args.start_prompt+args.count]
    validation = caption_records(config, "validation", 5000)
    if {r["image_id"] for r in selected} & {r["image_id"] for r in validation}:
        raise ValueError("Diagnostic captions must be training-only")
    bank = Path(config["train"]["dmd_pair_cache_dir"])
    bank_meta = json.loads((bank/"teacher_pairs.json").read_text())
    bank_sha = sha256(bank/"teacher_pairs.safetensors")
    repeated = [r for r in rows[:2500] for _ in range(2)]
    if (bank_meta["weights_sha256"] != bank_sha or
            bank_meta["protocol"]["prompt_hash"] != manifest_hash(repeated) or
            bank_meta["protocol"]["seeds_per_prompt"] != 2 or
            bank_meta["protocol"]["grouping"] != "adjacent-images-share-caption"):
        raise ValueError("Teacher-pair provenance does not match the training caption pool")
    tensors = load_file(str(bank/"teacher_pairs.safetensors"))
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    memory_gib = 12 if args.calibrate_gradients else 8
    if free < (memory_gib+1)*2**30:
        raise RuntimeError(f"Diagnostic needs at least {memory_gib+1} GiB free GPU memory")
    torch.cuda.set_per_process_memory_fraction(memory_gib*2**30/total, 0)
    bundle = SDBundle(config, device)
    metadata = restore_export(bundle, source_run/"stage_1")
    grid = torch.tensor(metadata["timesteps"], device=device)
    if grid.tolist() != [999]:
        raise ValueError("The diagnostic requires the same one-step endpoint")
    before = {k: v.cpu() for k, v in adapter_state(bundle.unet).items()}
    if args.calibrate_gradients:
        from few_step_diffusion.sd_models import new_unet, add_lora, epsilon
        from few_step_diffusion.diffusion import coefficients
        from few_step_diffusion.dmd_noise_agreement import normalized_score_direction
        from few_step_diffusion.dmd_response import response_loss
        state = torch.load(parent/"last.pt", map_location="cpu", weights_only=False)
        real_score = new_unet(config, device)
        fake_score = add_lora(new_unet(config, device), config, trainable=False, state=state["fake"])
        del state
        probe_generator = torch.Generator().manual_seed(317213)
        unconditional = bundle.embeddings(["", ""])

    def cosine(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        return ((a@b)/(a.norm()*b.norm()).clamp_min(1e-12)).clamp(-1, 1).item()

    measured = []
    for offset, row in enumerate(tqdm(selected, desc="Frozen teacher responses", unit="caption")):
        index = args.start_prompt+offset
        z = tensors["noise"][2*index:2*index+2]
        teacher = tensors["latents"][2*index:2*index+2].float()
        with torch.inference_mode():
            student = generate_sd_latents(bundle, bundle.unet, grid,
                [row["caption"]]*2, 0, noise=z.to(device)).cpu()
        if not torch.isfinite(student).all():
            raise FloatingPointError("Nonfinite student diagnostic output")
        scales = {}
        for scale in (1, 2, 4, 8):
            s, t, noise = (F.avg_pool2d(x, scale) for x in (student, teacher, z))
            ds, dt, dz = s[1]-s[0], t[1]-t[0], noise[1]-noise[0]
            point_error = (s-t).square().mean()
            center_error = ((s-t).mean(0)).square().mean()
            response_error = (.5*(ds-dt)).square().mean()
            torch.testing.assert_close(point_error, center_error+response_error)
            sr = ds-dz*((ds*dz).sum()/dz.square().sum().clamp_min(1e-12))
            tr = dt-dz*((dt*dz).sum()/dz.square().sum().clamp_min(1e-12))
            scales[str(scale)] = {
                "response_cosine": cosine(ds, dt),
                "student_to_teacher_response_norm": (ds.norm()/dt.norm().clamp_min(1e-12)).item(),
                "response_error_fraction": (response_error/point_error.clamp_min(1e-12)).item(),
                "point_latent_mse": point_error.item(),
                "student_noise_cosine": cosine(ds, dz),
                "teacher_noise_cosine": cosine(dt, dz),
                "noise_residual_response_cosine": cosine(sr, tr),
            }
        record = {"prompt_index": index, "image_id": row["image_id"],
                  "caption": row["caption"], "scales": scales}
        if args.calibrate_gradients:
            generated = student.to(device).detach().requires_grad_(True)
            target = teacher.to(device)
            t = torch.randint(20, 980, (2,), generator=probe_generator).to(device)
            noise = torch.randn(generated.shape, generator=probe_generator).to(device)
            with torch.no_grad():
                conditional = bundle.embeddings([row["caption"]]*2)
                alpha, sigma = coefficients(bundle.alphas, t)
                noisy = alpha*generated+sigma*noise
                real = epsilon(real_score, noisy, t, conditional, unconditional, 7.5, config, device)
                fake = epsilon(fake_score, noisy, t, conditional, None, 1., config, device)
                direction = normalized_score_direction(generated, noisy, real, fake, t, bundle.alphas)
                dmd_norm = (direction/generated.numel()).flatten(1).norm(dim=1).mean().item()
            gradient_sizes = {"dmd": dmd_norm}
            for mode in ("angular", "magnitude", "vector"):
                loss, _ = response_loss(generated, target, mode)
                gradient, = torch.autograd.grad(loss, generated)
                gradient_sizes[mode] = gradient.flatten(1).norm(dim=1).mean().item()
            if min(gradient_sizes.values()) <= 0:
                raise ValueError("Calibration requires nonzero gradients in all compared terms")
            record["gradient_norms"] = gradient_sizes
            record["score_timesteps"] = t.tolist()
        measured.append(record)
    after = adapter_state(bundle.unet)
    unchanged = all(torch.equal(before[k], value.cpu()) for k, value in after.items())
    if (not unchanged or sha256(source) != identity or sha256(parent/"last.pt") != checkpoint_sha or
            sha256(source_run/"last.pt") != source_checkpoint_sha):
        raise RuntimeError("Diagnostic source or fixed reference changed during the diagnostic")
    summary = {}
    for scale in (1, 2, 4, 8):
        summary[str(scale)] = {}
        for key in measured[0]["scales"][str(scale)]:
            values = torch.tensor([r["scales"][str(scale)][key] for r in measured], dtype=torch.float64)
            summary[str(scale)][key] = {"mean": values.mean().item(),
                "median": values.median().item(), "min": values.min().item(), "max": values.max().item()}
    calibration = None
    if args.calibrate_gradients:
        calibration = {"student_weights": "fixed_reference_ema", "probe_seed": 317213,
                       "target_gradient_fractions": [.1, .3, .5], "coefficients": {}}
        for mode in ("angular", "magnitude", "vector"):
            ratios = torch.tensor([r["gradient_norms"][mode]/r["gradient_norms"]["dmd"]
                                   for r in measured], dtype=torch.float64)
            median = ratios.quantile(.5).item()
            calibration["coefficients"][mode] = {
                "unweighted_median_ratio": median,
                "weights": {str(fraction): fraction/median for fraction in (.1, .3, .5)}}
    atomic_json(output, {"policy": "frozen-teacher-finite-response-diagnostic-v1",
        "training_only": True, "optimizer_updates": 0, "weights_unchanged": unchanged,
        "model_identity": identity, "parent_checkpoint_sha256": checkpoint_sha,
        "reference_model_identity": reference_identity, "source_run": str(source_run.relative_to(ROOT)),
        "source_checkpoint_sha256": source_checkpoint_sha,
        "model_role": "evaluated_response_variant" if args.study else "fixed_reference",
        "config_hash": config_fingerprint(config), "teacher_pair_weights_sha256": bank_sha,
        "teacher_pair_protocol_hash": bank_meta["protocol_hash"],
        "prompt_hash": manifest_hash(selected), "start_prompt": args.start_prompt,
        "prompt_count": args.count, "student_nfe": 2*args.count,
        "summary": summary, "measurements": measured, "gradient_calibration": calibration,
        "note": "Independent cached noise pairs: finite secants, not local Jacobians or quality scores."})
    print(json.dumps(summary, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
