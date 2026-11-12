#!/usr/bin/env python3
"""Read-only probe-direction diagnostic; neither training nor an image-quality test."""

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def direction_metrics(update, reference):
    """Compare latent directions with an independent, finite-probe reference."""
    import torch
    update, reference = update.detach().float().flatten(), reference.detach().float().flatten()
    norms = torch.stack([update.norm(), reference.norm()])
    if not torch.isfinite(update).all() or not torch.isfinite(reference).all() or (norms <= 0).any():
        raise ValueError("Direction diagnostic requires finite nonzero estimates")
    return {"cosine_to_reference": ((update*reference).sum()/norms.prod()).clamp(-1, 1).item(),
            "relative_squared_error": ((update-reference).square().sum()/norms[1].square()).item(),
            "norm": norms[0].item()}


def strength_direction(mean, agreement, strength, *, shifted=False):
    """Diagnostic-only sensitivity; does not change the training implementation."""
    import torch
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("Diagnostic strength must be finite and nonnegative")
    if shifted:
        agreement = agreement.roll((agreement.shape[-2]//2, agreement.shape[-1]//2), (-2, -1))
    weights = 1+strength*agreement
    weights = weights.repeat_interleave(mean.shape[-2]//weights.shape[-2], -2)
    weights = weights.repeat_interleave(mean.shape[-1]//weights.shape[-1], -1)
    weighted = weights*mean
    norm = mean.flatten(1).norm(dim=1).reshape(-1, 1, 1, 1)
    denominator = weighted.flatten(1).norm(dim=1).reshape(-1, 1, 1, 1)
    return weighted*(norm/denominator.clamp_min(torch.finfo(torch.float32).tiny))


def sensitivity_metrics(first, second, reference, strengths, patch_size):
    """Measure changes relative to the mean with the same number of probes."""
    import torch
    from few_step_diffusion.dmd_noise_agreement import combine_score_probes
    mean, agreement, _ = combine_score_probes(first, second, "paired_mean", patch_size)
    plain = direction_metrics(mean, reference)
    variants = {}
    for mode in ("agreement", "shifted_agreement"):
        variants[mode] = {}
        for strength in strengths:
            update = strength_direction(mean, agreement, strength, shifted=mode == "shifted_agreement")
            if strength == 1:
                actual, _, _ = combine_score_probes(first, second, mode, patch_size)
                torch.testing.assert_close(update, actual, rtol=0, atol=0)
            measured = direction_metrics(update, reference)
            measured["cosine_gain_over_mean"] = measured["cosine_to_reference"]-plain["cosine_to_reference"]
            variants[mode][f"{strength:g}"] = measured
    return {"paired_mean": plain, "strengths": variants}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--captions", type=int, default=4)
    parser.add_argument("--start-caption", type=int, default=0)
    parser.add_argument("--timesteps", nargs="+", type=int, default=[100, 500, 900])
    parser.add_argument("--reference-pairs", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=91827)
    parser.add_argument("--memory-gib", type=float, default=10)
    parser.add_argument("--strengths", nargs="+", type=float,
                        help="Read-only filter-strength diagnostic; does not select or train a model")
    parser.add_argument("--independent-pair-check", action="store_true",
                        help="Also compare agreement between two independent antithetic-pair means")
    args = parser.parse_args()
    if args.captions < 1 or args.start_caption < 0 or args.reference_pairs < 2 or args.memory_gib <= 0:
        raise ValueError("Use positive caption/memory budgets and at least two independent reference pairs")
    if args.strengths and (any(not math.isfinite(s) or s < 0 for s in args.strengths) or
                           len(set(args.strengths)) != len(args.strengths)):
        raise ValueError("Use distinct finite nonnegative diagnostic strengths")
    if args.independent_pair_check and not args.strengths:
        raise ValueError("Independent-pair comparison requires explicit diagnostic strengths")
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch
    from tqdm.auto import tqdm
    from few_step_diffusion.diffusion import coefficients, make_grid, predict_x0
    from few_step_diffusion.dmd_noise_agreement import combine_score_probes, normalized_score_direction
    from few_step_diffusion.runtime import atomic_json, experiment_lock, seed_everything, sha256
    from few_step_diffusion.sd_data import CaptionStream, manifest_hash
    from few_step_diffusion.sd_models import SDBundle, adapter_state, add_lora, epsilon, new_unet

    with experiment_lock(args.output.parent/args.output.stem):
        if args.output.exists():
            raise FileExistsError("Diagnostic output exists; preserve it instead of replacing it")
        digest = sha256(args.checkpoint)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = state["config"]
        if config["train"]["mode"] != "dmd" or "fake" not in state:
            raise ValueError("Diagnostic needs a complete DMD generator and auxiliary-scorer state")
        seed_everything(args.seed)
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        if free < (args.memory_gib+1)*2**30:
            raise RuntimeError("Insufficient free memory; do not interrupt existing training workers")
        torch.cuda.set_per_process_memory_fraction(args.memory_gib*2**30/total, 0)
        stream = CaptionStream(config, images=False)
        bundle = SDBundle(config, device)
        student = add_lora(new_unet(config, device), config, trainable=False, state=state["student"])
        fake = add_lora(new_unet(config, device), config, trainable=False, state=state["fake"])
        for model in (student, fake, bundle.unet):
            model.eval().requires_grad_(False)
        if any(t < 1 or t >= len(bundle.alphas) for t in args.timesteps):
            raise ValueError("Diagnostic timesteps must lie in the model's nonterminal noise range")
        root = make_grid(len(bundle.alphas), config["train"]["initial_steps"], device)
        empty = bundle.embeddings([""])
        # Local generators prevent caption order and independent-probe counts
        # from changing the candidate's two probe noises.
        latent_rng = torch.Generator(device=device).manual_seed(args.seed)
        pair_rng = torch.Generator(device=device).manual_seed(args.seed+1)
        reference_rng = torch.Generator(device=device).manual_seed(args.seed+2)
        extra_pair_rng = torch.Generator(device=device).manual_seed(args.seed+3)
        measurements = []
        expected = args.captions*len(args.timesteps)
        start_batch = config["train"]["updates_per_stage"]*config["train"]["grad_accumulation"]
        with torch.inference_mode(), tqdm(total=expected, desc="Independent score probes", unit="case") as progress:
            for caption in range(args.captions):
                _, prompts = stream.batch(start_batch+args.start_caption+caption, device)
                prompts = prompts[:1]
                conditions = bundle.embeddings(prompts)
                z = torch.randn((1, 4, 64, 64), device=device, generator=latent_rng)
                terminal = root[0].expand(1)
                generated = predict_x0(z, epsilon(student, z, terminal, conditions, None, 1,
                                                   config, device), terminal, bundle.alphas)
                for timestep in args.timesteps:
                    t = torch.tensor([timestep], device=device)
                    alpha, sigma = coefficients(bundle.alphas, t)

                    def probe_pair(generator):
                        noise = torch.randn(generated.shape, device=device, generator=generator)
                        estimates = []
                        for sign in (1, -1):
                            noisy = alpha*generated+sigma*(sign*noise)
                            real = epsilon(bundle.unet, noisy, t, conditions, empty,
                                           config["model"]["guidance_scale"], config, device)
                            auxiliary = epsilon(fake, noisy, t, conditions, empty, 1, config, device)
                            estimates.append(normalized_score_direction(generated, noisy, real, auxiliary,
                                                                       t, bundle.alphas))
                        return estimates

                    first, second = probe_pair(pair_rng)
                    independent = [direction for _ in range(args.reference_pairs)
                                   for direction in probe_pair(reference_rng)]
                    reference = torch.stack(independent).mean(0)
                    modes = {}
                    for mode in ("paired_mean", "agreement", "shifted_agreement"):
                        update, _, statistics = combine_score_probes(first, second, mode, args.patch_size)
                        modes[mode] = {**direction_metrics(update, reference), **statistics}
                    row = {"caption_index": args.start_caption+caption, "prompt": prompts[0], "timestep": timestep,
                           "reference_norm": reference.norm().item(), "modes": modes}
                    for mode in ("agreement", "shifted_agreement"):
                        modes[mode]["cosine_gain_over_mean"] = (modes[mode]["cosine_to_reference"]-
                                                                modes["paired_mean"]["cosine_to_reference"])
                    if args.strengths:
                        row["sensitivity"] = {"two_probes": sensitivity_metrics(first, second, reference,
                                                                             args.strengths, args.patch_size)}
                        if args.independent_pair_check:
                            third, fourth = probe_pair(extra_pair_rng)
                            # Unlike the opposite members of one pair, these two
                            # antithetic means have independent residual noise.
                            row["sensitivity"]["four_probes"] = sensitivity_metrics(
                                .5*(first+second), .5*(third+fourth), reference, args.strengths, args.patch_size)
                    measurements.append(row)
                    atomic_json(args.output.with_suffix(".progress.json"), {
                        "status": "diagnostic_in_progress", "checkpoint_sha256": digest,
                        "completed_cases": len(measurements), "planned_cases": expected,
                        "measurements": measurements})
                    progress.update(1)
                    print(json.dumps({"caption": caption, "timestep": timestep,
                        "candidate_cosine_gain": modes["agreement"]["cosine_gain_over_mean"],
                        "shifted_cosine_gain": modes["shifted_agreement"]["cosine_gain_over_mean"]}), flush=True)
        for model, key in ((student, "student"), (fake, "fake")):
            for name, value in adapter_state(model).items():
                if not torch.equal(value.cpu(), state[key][name].float()):
                    raise ValueError("Read-only diagnostic unexpectedly changed model weights")
        if sha256(args.checkpoint) != digest:
            raise ValueError("Diagnostic source checkpoint changed")
        averages = {mode: sum(row["modes"][mode]["cosine_gain_over_mean"] for row in measurements)/expected
                    for mode in ("agreement", "shifted_agreement")}
        atomic_json(args.output, {"status": "diagnostic", "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": digest, "checkpoint_step": state["global_step"],
            "weights_unchanged": True, "optimizer_updates": 0,
            "training_pool_only": True, "training_prompt_hash": manifest_hash(stream.records),
            "diagnostic_seed": args.seed, "captions": args.captions, "timesteps": args.timesteps,
            "start_caption": args.start_caption, "generator_state": "student", "scorer_state": "fake",
            "probe_generator_seeds": {"latent": args.seed, "candidate": args.seed+1, "reference": args.seed+2},
            "sensitivity_strengths": args.strengths,
            "independent_pair_check": args.independent_pair_check,
            "additional_pair_seed": args.seed+3 if args.independent_pair_check else None,
            "candidate_score_probes": 2, "independent_reference_probes": 2*args.reference_pairs,
            "reference_noise_pairs": args.reference_pairs,
            "reference_sampling": "Antithetic pairs; each pair is independent of other pairs and the candidate pair",
            "patch_size": args.patch_size, "measurements": measurements,
            "mean_cosine_gain_over_paired_mean": averages,
            "interpretation": "Latent-direction agreement with a finite, independently sampled reference. "
                "Reference estimates remain noisy; this is not ground truth, image quality, "
                "or evidence of statistically significant improvement."})
        print(f"Read-only diagnostic saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
