#!/usr/bin/env python3
"""Regenerate a fixed validation-caption pool for honest matched-model sample selection."""

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torchvision.utils import save_image
from tqdm.auto import tqdm

from few_step_diffusion.runtime import atomic_json, sha256
from few_step_diffusion.sd_data import caption_records, manifest_hash
from few_step_diffusion.sd_evaluation import load_clip, text_alignment
from few_step_diffusion.sd_models import SDBundle, generate_sd, load_adapter, restore_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("artifacts/assignment2_sample_selection/validation128"))
    args = parser.parse_args()
    selection = json.loads(Path("configs/assignment2_selected_method.json").read_text())
    sources = [Path("results/sd15/dmd_diversity_study/dmd_perceptual_seed42_1steps.result.json"),
               Path(f"results/sd15/{selection['study']}/{selection['method']}_seed42_1steps.result.json")]
    if sha256(sources[1]) != selection["result_sha256"]:
        raise ValueError("Proposal result differs from the selected checkpoint")
    results = [json.loads(p.read_text()) for p in sources]
    for key in ("eval_seed", "teacher_identity", "evaluation_prompt_hash", "precision", "timesteps"):
        if results[0][key] != results[1][key]:
            raise ValueError(f"Unmatched paired-evaluation setting: {key}")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(.06, device)
    config = results[0]["resolved_config"]
    rows = caption_records(config, "validation", args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    binding = {"models": [r["model_identity"] for r in results], "prompt_hash": manifest_hash(rows),
               "eval_seed": results[0]["eval_seed"], "timesteps": results[0]["timesteps"],
               "count": args.count, "inference_batch_size": 1}
    binding_path = args.output/"binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("Candidate directory is bound to another paired protocol")
    if not binding_path.exists() and any(args.output.iterdir()):
        raise ValueError("Unbound existing images cannot be adopted as matched candidates")
    atomic_json(binding_path, binding)
    bundle = SDBundle(config, device)
    grid = torch.tensor(results[0]["timesteps"], device=device)
    models = []
    for index, result in enumerate(results):
        export = Path(result["resolved_config"]["output_dir"])/"stage_1"
        if sha256(export/"adapter.safetensors") != result["model_identity"]:
            raise ValueError("Paired samples must use the evaluated model")
        if index == 0:
            restore_export(bundle, export)
        else:
            load_adapter(bundle.unet, load_file(str(export/"adapter.safetensors")))
        folder = args.output/result["method"]
        folder.mkdir(exist_ok=True)
        rng = torch.Generator().manual_seed(result["eval_seed"])
        paths = []
        for i, row in enumerate(tqdm(rows, desc=result["method"])):
            z = torch.randn((1, 4, 64, 64), generator=rng).to(device)
            path = folder/f"{i:03d}_{row['image_id']}.png"
            if not path.exists():
                image = generate_sd(bundle, bundle.unet, grid, [row["caption"]], result["eval_seed"], noise=z)
                save_image((image.clamp(-1, 1)+1)/2, path)
            paths.append(str(path))
        models.append({"method": result["method"], "model_identity": result["model_identity"],
                       "result_file": str(sources[index]), "result_sha256": sha256(sources[index]),
                       "paths": paths, "updates": result["cumulative_training_updates"]})
    del bundle
    gc.collect()
    torch.cuda.empty_cache()
    model, processor = load_clip(device)
    prompts = [r["caption"] for r in rows]
    for item in models:
        scores = []
        for start in range(0, len(rows), 8):
            scores.extend(text_alignment(model, processor, item["paths"][start:start+8], prompts[start:start+8], device))
        item["clip_scores"] = [float(s) for s in scores]
        item["image_sha256"] = [sha256(Path(p)) for p in item["paths"]]
    ranking = sorted(range(len(rows)), key=lambda i: models[1]["clip_scores"][i]-models[0]["clip_scores"][i], reverse=True)
    atomic_json(args.output/"candidates.json", {"protocol": "paired-native-validation-candidate-review-v1",
        "rows": rows, "prompt_hash": manifest_hash(rows), "eval_seed": results[0]["eval_seed"],
        "noise_policy": "CPU generator stream; identical initial latents and caption order for both models",
        "sampling_steps": 1, "nfe": 1, "inference_batch_size": 1, "models": models,
        "review_order_by_clip_difference": ranking,
        "selection_policy": "CLIP differences prioritize visual review; no claim of typical performance or updated full-set metrics."})


if __name__ == "__main__":
    main()
