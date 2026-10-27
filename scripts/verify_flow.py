#!/usr/bin/env python3
"""Verify the epsilon-to-velocity initialization learns before scaling InstaFlow."""

import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="0")
    p.add_argument("--output", default="runs/verification/flow_vp_scale50k")
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch
    from few_step_diffusion.config import load_config
    from few_step_diffusion.diffusion import make_grid, nested_grid
    from few_step_diffusion.runtime import atomic_json, device_for, seed_everything
    from few_step_diffusion.sd_data import caption_records
    from few_step_diffusion.sd_models import SDBundle, generate_sd_latents, restore_export
    from few_step_diffusion.teacher_pairs import TeacherPairs
    from few_step_diffusion.training import train

    output = Path(args.output)
    if (output/"diagnostic.json").is_file():
        saved = json.loads((output/"diagnostic.json").read_text())
        if saved["status"] != "passed":
            raise ValueError("Previous flow diagnostic failed; investigate before reusing this folder")
        print(json.dumps(saved), flush=True)
        return
    config = load_config("configs/instaflow.yaml")
    config.update(output_dir=str(output))
    config["data"].update(train_limit=16)
    config["train"].update(instaflow_pair_count=16, instaflow_prediction_type="epsilon-converted-velocity",
        updates_per_stage=400, grad_accumulation=1, warmup_steps=25, save_every=100, log_every=10)
    config["eval"].update(num_samples=16, num_real=16)
    config["wandb"]["group"] = "diagnostics/scale50k-flow"
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    device = device_for(config)
    bundle = SDBundle(config, device)
    grid = make_grid(len(bundle.alphas), config["train"]["initial_steps"], device)
    pairs = TeacherPairs(config, bundle, grid, output, caption_records(config))
    bundle.unet._fsd_prediction_type = "epsilon-converted-velocity"
    one = nested_grid(grid, 1)

    @torch.no_grad()
    def endpoint_error(current):
        errors = []
        for start in range(0, len(pairs.rows), 4):
            prompts = [r["caption"] for r in pairs.rows[start:start+4]]
            prediction = generate_sd_latents(current, current.unet, one, prompts, config["seed"],
                                            noise=pairs.tensors["noise"][start:start+4].to(device))
            target = pairs.tensors["latents"][start:start+4].to(device)
            errors.append((prediction-target).square().mean().item())
        return sum(errors)/len(errors)

    before = endpoint_error(bundle)
    del bundle
    torch.cuda.empty_cache()
    train(config, resume=(output/"last.pt").exists())
    bundle = SDBundle(config, device)
    restore_export(bundle, output/"stage_1")
    after = endpoint_error(bundle)
    status = "passed" if after < before and torch.isfinite(torch.tensor(after)) else "failed"
    result = {"status": status, "before_teacher_endpoint_mse": before,
              "after_teacher_endpoint_mse": after, "ratio": after/before,
              "updates": 400, "pair_count": 16, "prediction_type": "epsilon-converted-velocity",
              "note": "Small training-coupling diagnostic; not a quality benchmark or held-out improvement claim."}
    atomic_json(output/"diagnostic.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if status != "passed":
        raise RuntimeError("Converted-flow endpoint error did not improve; large InstaFlow run stays gated")


if __name__ == "__main__":
    main()
