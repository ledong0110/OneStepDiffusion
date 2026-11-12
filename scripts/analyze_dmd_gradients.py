#!/usr/bin/env python3
"""Measure DMD/LPIPS gradient alignment on training captions without training."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Immutable saved full training state")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7319)
    parser.add_argument("--start-update", type=int, default=18000)
    args = parser.parse_args()
    if args.batches < 1 or args.start_update < 0:
        raise ValueError("Use a positive diagnostic batch count and nonnegative stream offset")
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from tqdm.auto import tqdm
    from few_step_diffusion.diffusion import coefficients, make_grid, predict_x0
    from few_step_diffusion.distillation_objectives import dmd_surrogate
    from few_step_diffusion.dmd_conflict import GradientAccumulator
    from few_step_diffusion.dmd_diversity import load_lpips, regression_and_diversity
    from few_step_diffusion.gradient_diagnostics import alignment_statistics
    from few_step_diffusion.runtime import atomic_json, experiment_lock, seed_everything, sha256
    from few_step_diffusion.sd_data import CaptionStream, manifest_hash
    from few_step_diffusion.sd_models import SDBundle, adapter_state, add_lora, epsilon, new_unet
    from few_step_diffusion.teacher_pairs import TeacherPairs

    if args.output.exists():
        raise FileExistsError("Diagnostic output already exists; inspect it before rerunning")
    digest = sha256(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, tr = state["config"], state["config"]["train"]
    if (tr["mode"] != "dmd" or tr.get("dmd_regression_type") != "lpips_vgg" or
            tr.get("dmd_diversity_weight", 0) != 0):
        raise ValueError("Diagnostic requires the DMD+LPIPS objective")
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    if free < 16*2**30:
        raise RuntimeError("Diagnostic needs 16 GiB free; leave the live training workers untouched")
    torch.cuda.set_per_process_memory_fraction(15*2**30/total, 0)
    stream = CaptionStream(config, images=False)
    bundle = SDBundle(config, device)
    student = add_lora(new_unet(config, device), config, state=state["student"])
    fake = add_lora(new_unet(config, device), config, trainable=False, state=state["fake"])
    named = [(name, p) for name, p in student.named_parameters() if p.requires_grad]
    names, parameters = zip(*named, strict=True)
    metric = load_lpips("vgg", device)
    root = make_grid(len(bundle.alphas), tr["initial_steps"], device)
    with experiment_lock(tr["dmd_pair_cache_dir"]):
        pairs = TeacherPairs(config, bundle, root, tr["dmd_pair_cache_dir"], stream.records)
    empty = bundle.embeddings([""])
    seed_everything(args.seed)
    measurements = []
    for batch in tqdm(range(args.batches), desc="Read-only gradient diagnosis", unit="batch"):
        accumulator = GradientAccumulator(parameters)
        for micro in range(tr["grad_accumulation"]):
            index = (args.start_update+batch)*tr["grad_accumulation"]+micro
            _, prompts = stream.batch(index, device)
            conditions = bundle.embeddings(prompts)
            unconditional = empty.expand(len(prompts), -1, -1)
            z = torch.randn((len(prompts), 4, 64, 64), device=device)
            terminal = root[0].expand(len(z))
            generated = predict_x0(z, epsilon(student, z, terminal, conditions, None, 1,
                                              config, device), terminal, bundle.alphas)
            t = torch.randint(max(1, int(len(bundle.alphas)*tr.get("vsd_t_min", .02))),
                min(len(bundle.alphas), int(len(bundle.alphas)*tr.get("vsd_t_max", .98))+1),
                (len(z),), device=device)
            alpha, sigma = coefficients(bundle.alphas, t)
            with torch.no_grad():
                noisy = alpha*generated.detach()+sigma*torch.randn_like(generated)
                teacher_eps = epsilon(bundle.unet, noisy, t, conditions, unconditional,
                                      config["model"]["guidance_scale"], config, device)
                fake_eps = epsilon(fake, noisy, t, conditions, unconditional, 1, config, device)
            score = dmd_surrogate(generated, noisy, teacher_eps, fake_eps, t, bundle.alphas)
            pair_noise, targets, pair_prompts = pairs.batch(index, device)
            pair_conditions = bundle.embeddings(pair_prompts)
            generated_pair = predict_x0(pair_noise, epsilon(student, pair_noise, terminal,
                pair_conditions, None, 1, config, device), terminal, bundle.alphas)
            regression, _, _ = regression_and_diversity(bundle, metric, generated_pair, targets, config)
            accumulator.add(score, regression, tr["grad_accumulation"])
        row = alignment_statistics(names, accumulator.score, accumulator.perceptual,
                                   tr["dmd_regression_weight"])
        row["batch"] = batch
        measurements.append(row)
        atomic_json(args.output.with_suffix(".progress.json"), {
            "status": "diagnostic_in_progress", "checkpoint_sha256": digest,
            "completed_batches": len(measurements), "planned_batches": args.batches,
            "measurements": measurements})
        print(json.dumps({k: v for k, v in row.items() if k != "blocks"}), flush=True)
        del accumulator
    for model, key in ((student, "student"), (fake, "fake")):
        for name, value in adapter_state(model).items():
            if not torch.equal(value.cpu(), state[key][name].float()):
                raise ValueError("Read-only diagnostic unexpectedly changed model weights")
    if sha256(args.checkpoint) != digest:
        raise ValueError("Diagnostic source checkpoint changed during execution")
    summary_keys = ("global_cosine", "conflicting_block_fraction", "global_relative_correction",
                    "block_relative_correction")
    atomic_json(args.output, {"status": "diagnostic", "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest, "checkpoint_updates": state["global_step"],
        "weights_unchanged": True, "optimizer_updates": 0, "diagnostic_seed": args.seed,
        "training_prompt_hash": manifest_hash(stream.records), "training_pool_only": True,
        "start_update": args.start_update, "batches": args.batches,
        "effective_batch_size": config["data"]["batch_size"]*tr["grad_accumulation"],
        "regression_weight": tr["dmd_regression_weight"],
        "mean": {k: sum(r[k] for r in measurements)/len(measurements) for k in summary_keys},
        "measurements": measurements,
        "interpretation": "Gradient geometry only; no image-quality improvement measurement."})
    print(f"Diagnostic saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
