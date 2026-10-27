"""Held-out same-prompt diversity measured with an unused LPIPS backbone."""

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm.auto import tqdm

from .config import config_fingerprint
from .diffusion import make_grid
from .dmd_diversity import load_lpips, lpips_identity
from .runtime import atomic_json, device_for, seed_everything, sha256
from .sd_data import caption_records, manifest_hash
from .sd_models import SDBundle, generate_sd, restore_export


def evaluate_diversity(config, checkpoint, output, result_file=None, prompt_count=128):
    seed_everything(config["seed"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = device_for(config)
    records = caption_records(config, "validation", prompt_count)
    seeds = [config["eval"]["seed"]+i for i in range(4)]
    noises = torch.stack([torch.randn((prompt_count, 4, 64, 64),
        generator=torch.Generator().manual_seed(seed)) for seed in seeds], dim=1)
    bundle = SDBundle(config, device)
    if checkpoint == "teacher":
        grid = make_grid(len(bundle.alphas), config["train"]["initial_steps"], device)
        guidance = config["model"]["guidance_scale"]
        identity = config["model"]["source"]+"@"+config["model"]["revision"]
    else:
        metadata = restore_export(bundle, checkpoint)
        grid = torch.tensor(metadata["timesteps"], device=device)
        if len(grid) != 1:
            raise ValueError("DMD study diversity evaluation requires a one-step student")
        guidance, identity = 1., sha256(Path(checkpoint)/"adapter.safetensors")
    rendered = []
    with torch.no_grad():
        for index, record in enumerate(tqdm(records, desc="Held-out prompts x four noise seeds",
                                            disable=not config.get("progress_bar", True))):
            batch = min(2, config["eval"]["batch_size"])
            images = torch.cat([generate_sd(bundle, bundle.unet, grid,
                [record["caption"]]*len(noises[index, start:start+batch]), seeds[0], guidance,
                noise=noises[index, start:start+batch].to(device)).clamp(-1, 1)
                for start in range(0, 4, batch)])
            if not torch.isfinite(images).all():
                raise FloatingPointError("Nonfinite held-out diversity samples")
            # Evaluate the same 8-bit image representation used in the FID pipeline.
            rendered.append(((images+1)*127.5).round().byte().cpu())
    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metric = load_lpips("alex", device)
    pairs = list(itertools.combinations(range(4), 2))
    scores = []
    with torch.no_grad():
        for images in tqdm(rendered, desc="LPIPS-Alex across seeds",
                           disable=not config.get("progress_bar", True)):
            normalized = images.to(device).float()/127.5-1
            distances = metric(torch.stack([normalized[i] for i, _ in pairs]),
                               torch.stack([normalized[j] for _, j in pairs])).flatten()
            scores.append(distances.mean().item())
    save_image(torch.cat(rendered[:8]).float()/255, output/"four_seeds.png", nrow=4)
    measured = {"diversity_protocol": f"coco2017-val-first{prompt_count}-four-seeds-lpips-alex-v1",
        "diversity_prompt_hash": manifest_hash(records), "diversity_prompt_count": prompt_count,
        "diversity_seed_list": seeds, "diversity_samples_per_prompt": 4,
        "diversity_lpips_alex_mean": float(np.mean(scores)),
        "diversity_lpips_alex_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.,
        "diversity_lpips_identity": lpips_identity("alex"),
        "diversity_pairwise_scores_per_prompt": scores,
        "diversity_generation_nfe": prompt_count*4*len(grid)*(2 if guidance > 1 else 1),
        "diversity_sample_grid": str(output/"four_seeds.png")}
    atomic_json(output/"diversity.json", {"status": "diagnostic", "model_identity": identity,
        "config_hash": config_fingerprint(config), **measured})
    if result_file is not None:
        result = json.loads(Path(result_file).read_text())
        if result["model_identity"] != identity or result["config_hash"] != config_fingerprint(config):
            raise ValueError("Diversity and FID results refer to different models/configurations")
        atomic_json(result_file, {**result, **measured})
    return measured
