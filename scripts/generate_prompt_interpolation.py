#!/usr/bin/env python3
"""Generate real dog-to-cat text-embedding interpolation with the 36k proposal."""

import argparse
import base64
import io
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from few_step_diffusion.runtime import atomic_json, seed_everything, sha256
    from few_step_diffusion.sd_models import generate_sd
    from generate_active_qualitative import low_memory_bundle

    source = Path("results/sd15/dmd_diversity_study/dmd_perceptual_diversity_seed42_1steps.result.json")
    r = json.loads(source.read_text())
    config = r["resolved_config"]
    checkpoint = Path(config["output_dir"])/"stage_1"
    if r["cumulative_training_updates"] != 36000 or sha256(checkpoint/"adapter.safetensors") != r["model_identity"]:
        raise ValueError("Interpolation requires the evaluated full 36k proposal")
    prompts = [f"A DSLR photograph of a {animal} reading an open book, sitting at an outdoor cafe table."
               for animal in ("dog", "cat")]
    weights, seeds = [0, .25, .5, .75, 1], [12546, 12548, 12549]
    path = Path("results/sd15/active_one_step/prompt_interpolation.json")
    identity = {"status": "qualitative_interpolation", "protocol": "linear-clip-hidden-state-v1",
        "method": r["method"], "model_identity": r["model_identity"], "source_file": str(source),
        "cumulative_training_updates": 36000, "sampling_steps": 1, "nfe": 1,
        "prompts": prompts, "weights": weights, "candidate_noise_seeds": seeds,
        "inference_auxiliary_device": "CPU text encoder and VAE; native-precision GPU U-Net"}
    payload = json.loads(path.read_text()) if path.exists() else {**identity, "rows": []}
    if any(payload.get(k) != v for k,v in identity.items()):
        raise ValueError("Existing interpolation belongs to another checkpoint/protocol")
    seed_everything(42)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(.014, 0)
    bundle, metadata = low_memory_bundle(config, device, checkpoint)
    grid = torch.tensor(metadata["timesteps"], device=device)
    dog, cat = [bundle.embeddings([p]) for p in prompts]
    for seed in tqdm(seeds, desc="Dog-to-cat interpolation rows"):
        if any(row["noise_seed"] == seed for row in payload["rows"]):
            continue
        noise = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(seed)).to(device)
        fixed = noise.clone()
        items = []
        for weight in weights:
            embedding = torch.lerp(dog, cat, weight)
            pixels = generate_sd(bundle, bundle.unet, grid, [prompts[0]], seed, 1, noise=noise,
                                 embeddings=(embedding, None))
            if not torch.equal(noise, fixed) or not torch.isfinite(pixels).all():
                raise ValueError("Interpolation changed its fixed noise or produced a nonfinite output")
            array = ((pixels[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="PNG")
            items.append({"weight": weight, "png_base64": base64.b64encode(buffer.getvalue()).decode()})
        payload["rows"].append({"noise_seed": seed, "images": items})
        atomic_json(path, payload)
    print(f"Interpolation outputs: {path}", flush=True)


if __name__ == "__main__":
    main()
