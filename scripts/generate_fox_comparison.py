#!/usr/bin/env python3
"""Generate the same selected fox-astronaut prompt/noise with completed 36k models."""

import argparse
import base64
import gc
import io
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--eval-seed", type=int, required=True)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--name", choices=("fox", "cartoon", "modern", "goku"), default="fox")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from PIL import Image
    from few_step_diffusion.diffusion import make_grid
    from few_step_diffusion.qualitative import panel_specs
    from few_step_diffusion.runtime import atomic_json, seed_everything, sha256
    from few_step_diffusion.sd_models import generate_sd
    from generate_active_qualitative import low_memory_bundle

    seed_everything(42)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(.014, 0)
    source = Path(f"results/sd15/active_one_step/qualitative/seed{args.eval_seed}/dmd_perceptual_diversity_1steps.qualitative.json")
    if args.candidate_index is None:
        proposal = json.loads(source.read_text())
        item = panel_specs(args.eval_seed)["comparison"][1]
        own = proposal["panels"]["comparison"][1]
        if {k:v for k,v in own.items() if k != "png_base64"} != item:
            raise ValueError("Selected fox sample does not match its recorded prompt/noise")
        out = Path(f"results/sd15/active_one_step/fox_demo_seed{args.eval_seed}.json")
    else:
        source = Path(f"results/sd15/active_one_step/{args.name}_candidates.json")
        proposal = json.loads(source.read_text())
        own = next(c for c in proposal["candidates"] if c["index"] == args.candidate_index)
        item = {k:own[k] for k in ("prompt", "noise_seed")}
        out = Path(f"results/sd15/active_one_step/{args.name}_demo_candidate{args.candidate_index}.json")
    payload = json.loads(out.read_text()) if out.exists() else {
        "schema_version": 1, "status": "qualitative_selection", "image_size": 512,
        "prompt": item["prompt"], "noise_seed": item["noise_seed"], "eval_seed": args.eval_seed,
        "inference_auxiliary_device": "CPU text encoder and VAE; native-precision GPU U-Net",
        "methods": {"dmd_perceptual_diversity": {"model_identity": proposal["model_identity"],
            "source_file": str(source), "png_base64": own["png_base64"], "sampling_steps": 1, "nfe": 1}}}
    if payload["prompt"] != item["prompt"] or payload["noise_seed"] != item["noise_seed"]:
        raise ValueError("Existing fox comparison has a different prompt/noise")
    plan = json.loads(Path("configs/active_study.json").read_text())
    sources = {j["method"]: Path(j["result_dir"])/f"{j['method']}_seed42_1steps.result.json"
               for j in plan["jobs"] if j["method"] in ("baseline", "consistency", "dmd", "swiftbrush_lora", "instaflow")}
    sources["teacher"] = Path("results/sd15/scale50k/teacher_seed0_64steps.result.json")
    for method, path in sources.items():
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        if method in payload["methods"]:
            if payload["methods"][method]["model_identity"] != record["model_identity"]:
                raise ValueError("Existing fox sample has a different checkpoint identity")
            continue
        config = record["resolved_config"]
        checkpoint = None if method == "teacher" else Path(config["output_dir"])/"stage_1"
        if checkpoint and (record.get("cumulative_training_updates") != 36000 or
                           sha256(checkpoint/"adapter.safetensors") != record["model_identity"]):
            raise ValueError("Fox comparisons require evaluated 36k student checkpoints")
        print(f"Generate matched fox sample: {method}", flush=True)
        bundle, metadata = low_memory_bundle(config, device, checkpoint)
        if method == "teacher":
            grid = make_grid(len(bundle.alphas), 64, device)
            guidance = config["model"]["guidance_scale"]
        else:
            grid = torch.tensor(metadata["timesteps"], device=device)
            guidance = 1
        noise = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(item["noise_seed"])).to(device)
        try:
            pixels = generate_sd(bundle, bundle.unet, grid, [item["prompt"]], args.eval_seed, guidance, noise=noise)
        except torch.OutOfMemoryError:
            if method != "swiftbrush_lora":
                raise
            # FP32 SwiftBrush can exceed the small inference allowance: retain
            # FP32 math and move only this sample to CPU instead of changing it.
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(bundle.unet, recurse=True)
            bundle.unet.to("cpu")
            bundle.device = torch.device("cpu")
            bundle.alphas = bundle.alphas.cpu()
            def cpu_embeddings(prompts, encoder=bundle.text_encoder, tokenizer=bundle.tokenizer):
                return encoder(tokenizer(prompts, padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors="pt").input_ids)[0]
            bundle.embeddings = cpu_embeddings
            torch.cuda.empty_cache()
            pixels = generate_sd(bundle, bundle.unet, grid.cpu(), [item["prompt"]], args.eval_seed,
                                 guidance, noise=noise.cpu())
        if not torch.isfinite(pixels).all():
            raise FloatingPointError("Nonfinite fox sample")
        array = ((pixels[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        payload["methods"][method] = {"model_identity": record["model_identity"], "source_file": str(path),
            "png_base64": base64.b64encode(buffer.getvalue()).decode(), "sampling_steps": record["sampling_steps"],
            "nfe": record["nfe"], "precision": record["precision"]}
        atomic_json(out, payload)
        # Remove closures as well as module references before the next backbone.
        bundle.embeddings = None
        bundle.decode = None
        del bundle, pixels, noise
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(out, payload)
    print(f"Fox comparison: {out}", flush=True)


if __name__ == "__main__":
    main()
