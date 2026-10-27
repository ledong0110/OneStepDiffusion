#!/usr/bin/env python3
"""Generate the original fixed prompt-control panel with the completed new proposal."""

import argparse
import base64
import io
import json
import os
from pathlib import Path


def low_memory_bundle(config, device, checkpoint=None):
    """CPU text/VAE and sequential U-Net offload preserve native execution precision."""
    import torch
    from accelerate import cpu_offload
    from few_step_diffusion.sd_models import SDBundle, restore_export
    bundle = SDBundle(config, torch.device("cpu"))
    if config["precision"] == "bf16":
        bundle.unet.to(dtype=torch.bfloat16)
    metadata = restore_export(bundle, checkpoint) if checkpoint else {}
    cpu_offload(bundle.unet, device, offload_buffers=True)
    bundle.unet.eval()
    bundle.device = device
    bundle.alphas = bundle.alphas.to(device)

    @torch.no_grad()
    def embeddings(prompts):
        tokens = bundle.tokenizer(prompts, padding="max_length", truncation=True,
            max_length=bundle.tokenizer.model_max_length, return_tensors="pt")
        return bundle.text_encoder(tokens.input_ids)[0].to(device)

    def decode(latents):
        return bundle.vae.decode(latents.float().cpu()/bundle.vae.config.scaling_factor).sample

    bundle.embeddings = embeddings
    bundle.decode = decode
    return bundle, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[12345, 12346, 12347, 12348])
    parser.add_argument("--fox-extra", action="store_true", help="Try six explicit fox/helmet prompt candidates")
    parser.add_argument("--demo-kind", choices=("cartoon", "modern", "goku"), help="Generate a selected custom-demo candidate pool")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from few_step_diffusion.config import config_fingerprint
    from few_step_diffusion.qualitative import panel_specs, protocol_hash
    from few_step_diffusion.runtime import atomic_json, device_for, seed_everything, sha256
    from few_step_diffusion.sd_models import generate_sd

    result = Path("results/sd15/dmd_diversity_study/dmd_perceptual_diversity_seed42_1steps.result.json")
    record = json.loads(result.read_text())
    config = record["resolved_config"]
    checkpoint = Path(config["output_dir"])/"stage_1"
    if (record.get("status") != "complete" or record.get("cumulative_training_updates") != 36000 or
            config_fingerprint(config) != record["config_hash"] or
            sha256(checkpoint/"adapter.safetensors") != record["model_identity"]):
        raise ValueError("Prompt control requires the evaluated full 36,000-step proposal checkpoint")
    identity = {"method": record["method"], "sampling_steps": 1, "nfe": 1,
        "model_identity": record["model_identity"], "seed": record["seed"], "precision": record["precision"],
        "teacher_identity": record["teacher_identity"], "lora_rank": record["lora_rank"],
        "training_updates": record["training_updates"], "cumulative_training_updates": 36000}
    seed_everything(config["seed"])
    device = device_for(config)
    free, _ = torch.cuda.mem_get_info(device)
    if free < 1.5*1024**3:
        raise RuntimeError("Need 1.5 GiB free for single-image inference with CPU offload")
    torch.cuda.set_per_process_memory_fraction(.014, torch.cuda.current_device())
    bundle, metadata = low_memory_bundle(config, device, checkpoint)
    bundle.unet._fsd_prediction_type = metadata.get("student_prediction_type", "epsilon")
    bundle.unet.eval()
    grid = torch.tensor(metadata["timesteps"], device=device)
    if args.fox_extra or args.demo_kind:
        kind = args.demo_kind or "fox"
        prompts = [
            "A portrait photograph of a red fox astronaut wearing a white spacesuit and a large round transparent helmet, orange fur and white cheeks, planet Earth and stars in the background.",
            "A professional portrait photograph of a red fox wearing an astronaut helmet and a white NASA space suit, orange face with pointy ears inside the glass helmet, Earth visible behind.",
            "A cinematic close-up portrait of a red fox astronaut in an orange space suit, a clear bubble helmet around the fox head, Earth viewed from space behind, photorealistic.",
        ]
        if kind == "cartoon":
            prompts = [
                "A cute cartoon cat reading a book, colorful children's book illustration, clean bold outlines, warm colors.",
                "A colorful cartoon illustration of a small red car parked beside a yellow house, thick outlines, cheerful colors.",
                "A cute 3D cartoon rabbit holding a carrot, soft pastel colors, a simple light blue background.",
            ]
        elif kind == "modern":
            prompts = [
                "A photorealistic architectural photograph of a modern glass house beside a calm lake at dusk, warm interior lights, mountains in the background, clean contemporary design.",
                "A professional automotive photograph of a sleek silver sports car parked outside a modern glass building at blue hour, realistic reflections, cinematic lighting, sharp detail.",
                "A photorealistic interior photograph of a modern minimalist living room, a cream sofa, floor-to-ceiling windows, warm sunlight, indoor plants, contemporary design.",
            ]
        if kind == "goku":
            prompts = [
                "A detailed anime drawing of Son Goku from Dragon Ball on white paper on a wooden desk, spiky black hair, orange martial arts gi, blue belt and wristbands, glowing blue and purple aura, vibrant marker colors.",
                "A photograph of a colorful marker drawing of Goku, spiky dark blue hair, muscular arms, orange gi, blue belt, glowing blue and pink energy aura, white paper on a wooden tabletop, anime style.",
                "Son Goku from Dragon Ball, dynamic anime illustration on white drawing paper, orange martial arts outfit, blue wristbands and belt, spiky black hair, blue energy aura, bold ink outlines, vibrant colored pencil shading.",
            ]
        path = Path(f"results/sd15/active_one_step/{kind}_candidates.json")
        payload = json.loads(path.read_text()) if path.exists() else {
            "status": "qualitative_selection", "model_identity": record["model_identity"],
            "cumulative_training_updates": 36000, "candidates": []}
        if payload["model_identity"] != record["model_identity"]:
            raise ValueError("Fox candidate checkpoint changed")
        for i, (prompt, noise_seed) in enumerate((p, s) for p in prompts for s in (12346, 12349)):
            if any(c["index"] == i for c in payload["candidates"]):
                continue
            noise = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(noise_seed)).to(device)
            pixels = generate_sd(bundle, bundle.unet, grid, [prompt], noise_seed, 1, noise=noise)
            if not torch.isfinite(pixels).all():
                raise FloatingPointError("Nonfinite custom-demo candidate")
            array = ((pixels[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="PNG")
            payload["candidates"].append({"index": i, "prompt": prompt, "noise_seed": noise_seed,
                "png_base64": base64.b64encode(buffer.getvalue()).decode()})
            atomic_json(path, payload)
            print(f"{kind} candidate {i}: {path}", flush=True)
        return
    for eval_seed in args.eval_seeds:
        output = Path(f"results/sd15/active_one_step/qualitative/seed{eval_seed}/dmd_perceptual_diversity_1steps.qualitative.json")
        current = {**identity, "eval_seed": eval_seed, "protocol_hash": protocol_hash(eval_seed),
                   "inference_auxiliary_device": "CPU text encoder and VAE; native-precision GPU U-Net"}
        if output.exists():
            saved = json.loads(output.read_text())
            if any(saved.get(k) != v for k, v in current.items()):
                raise ValueError("Existing prompt-control outputs belong to another checkpoint/protocol")
            if args.resume:
                print(f"Reuse verified prompt-control outputs: {output}")
                continue
            raise FileExistsError(f"{output}; use --resume")
        panels = {}
        for kind in ("comparison", "control"):
            panels[kind] = []
            for item in tqdm(panel_specs(eval_seed)[kind], desc=f"36k proposal seed{eval_seed} {kind}"):
                noise = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(item["noise_seed"]))
                pixels = generate_sd(bundle, bundle.unet, grid, [item["prompt"]], eval_seed,
                                     guidance=1, noise=noise.to(device))
                if not torch.isfinite(pixels).all():
                    raise FloatingPointError("Nonfinite native prompt-control output")
                array = ((pixels[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()
                buffer = io.BytesIO()
                Image.fromarray(array).save(buffer, format="PNG")
                panels[kind].append({**item, "png_base64": base64.b64encode(buffer.getvalue()).decode()})
        atomic_json(output, {"schema_version": 1, "status": "qualitative", "image_size": 512,
            **current, "guidance_scale": 1, "panels": panels,
            "adaptation": "New DMD perceptual/diversity proposal, rank-32 LoRA, 36k total steps"})
        print(f"Prompt-control outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
