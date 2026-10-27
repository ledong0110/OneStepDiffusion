#!/usr/bin/env python3
"""Generate authentic teacher samples for the draft's qualitative layout preview."""

import argparse
import base64
import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", help="Physical GPU index; omit and pass --cpu for CPU generation")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--config", default="configs/sd15.yaml")
    p.add_argument("--steps", nargs="+", type=int, help="Teacher step counts; SD defaults to 64")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--output")
    args = p.parse_args()
    os.chdir(ROOT)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from PIL import Image
    import torch
    from tqdm.auto import tqdm
    from few_step_diffusion.config import dataset_name, image_size, is_stable_diffusion, load_config
    from few_step_diffusion.diffusion import make_grid, nested_grid
    from few_step_diffusion.models import load_model
    from few_step_diffusion.runtime import atomic_json, device_for, seed_everything
    from few_step_diffusion.sampling import generate

    config = load_config(args.config)
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    if args.cpu:
        config.update(device="cpu", precision="fp32")
    seed_everything(12345)
    device = device_for(config)
    sd = is_stable_diffusion(config)
    if sd:
        from few_step_diffusion.sd_models import SDBundle, generate_sd, PREVIEW_PROMPTS, PREVIEW_LABELS
        bundle = SDBundle(config, device)
        model, scheduler = bundle.unet, bundle.scheduler
    else:
        model, scheduler = load_model(config)
        model = model.to(device).eval().requires_grad_(False)
    alphas = scheduler.alphas_cumprod.to(device)
    root = make_grid(len(alphas), config["train"]["initial_steps"], device)
    records = []
    size = image_size(config)
    for steps in tqdm(args.steps or ([64] if sd else [64, 4, 1]), desc="Teacher reference samples", unit="grid"):
        batches, generator = [], torch.Generator().manual_seed(12345)
        for start in range(0, 8, args.batch_size):
            count = min(args.batch_size, 8-start)
            noise = torch.randn((count, 4, 64, 64) if sd else (count, 3, size, size), generator=generator).to(device)
            if sd:
                batches.append(generate_sd(bundle, model, nested_grid(root, steps),
                    PREVIEW_PROMPTS[start:start+count], 12345, config["model"]["guidance_scale"], noise=noise).cpu())
            else:
                batches.append(generate(model, alphas, nested_grid(root, steps), count, 12345,
                                        config, device, noise=noise).cpu())
        images = torch.cat(batches)
        encoded = []
        for image in ((images.clamp(-1, 1)+1)*127.5).round().byte().cpu():
            buffer = io.BytesIO()
            Image.fromarray(image.permute(1, 2, 0).numpy()).save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        records.append({"status": "preview", "method": "teacher", "sampling_steps": steps,
                        "nfe": steps*(2 if sd else 1),
                        "seed": 0, "eval_seed": 12345, "sample_images_png_base64": encoded,
                        "dataset": dataset_name(config), "image_size": size,
                        **({"sample_prompts": PREVIEW_PROMPTS, "sample_labels": PREVIEW_LABELS,
                            "teacher_guidance_scale": config["model"]["guidance_scale"]} if sd else {}),
                        "model_identity": config["model"]["source"]+"@"+config["model"]["revision"]})
    default_output = ("report/assets/teacher_preview_sd15.json" if sd else
                      "report/assets/teacher_preview_celebahq.json" if dataset_name(config) == "CelebA-HQ"
                      else "report/assets/teacher_preview.json")
    path = ROOT / (args.output or default_output)
    atomic_json(path, {"purpose": "Pretrained teacher preview only; no student results or metric claims",
                       "records": records})
    print(f"Teacher parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(path)


if __name__ == "__main__":
    main()
