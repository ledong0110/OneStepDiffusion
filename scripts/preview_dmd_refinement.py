#!/usr/bin/env python3
"""Preserve a live DMD checkpoint and compare native samples with the fixed reference.

This produces qualitative diagnostics only. It never invents a final evaluation
record or replaces the full validation metrics used for method selection.
"""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    from few_step_diffusion.dmd_response import MODES as RESPONSE_MODES, SETTINGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("project", "norm_match", "block_project", "block_norm_match",
                                 "agreement", "paired_mean", "shifted_agreement", *RESPONSE_MODES))
    parser.add_argument("--step", type=int, help="Use an immutable archived update instead of the rolling checkpoint")
    parser.add_argument("--study", choices=("dmd_conflict", "dmd_block_conflict",
                                           "dmd_noise_agreement", "dmd_pair_agreement", "dmd_response_strength",
                                           "dmd_response_final"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--memory-gib", type=float, default=4.)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from build_active_report import draw_samples
    from generate_active_qualitative import low_memory_bundle
    from few_step_diffusion.config import config_fingerprint, load_config
    from few_step_diffusion.runtime import atomic_json, experiment_lock, sha256
    from few_step_diffusion.sd_models import export_adapter, generate_sd

    study = args.study or ("dmd_noise_agreement" if args.mode in ("agreement", "paired_mean", "shifted_agreement") else
                           "dmd_block_conflict" if args.mode.startswith("block_") else "dmd_conflict")
    run = ROOT/"runs/sd15"/study/args.mode
    config = load_config(run/"config.yaml")
    manifest = json.loads((ROOT/"configs"/f"{study}.json").read_text())
    lineage = json.loads((run/"refinement.json").read_text())
    if lineage["parent_checkpoint_sha256"] != manifest["parent_checkpoint_sha256"]:
        raise ValueError("Preview parent differs from the pinned experiment")
    # Training replaces last.pt atomically. Copying its opened inode gives a
    # complete immutable state even when the next checkpoint replaces the path.
    with experiment_lock(run/"preview_lock"), tempfile.TemporaryDirectory(dir=run) as temporary:
        copied = Path(temporary)/"checkpoint.pt"
        source = (run/"checkpoint_previews"/f"step_{args.step:06d}"/"checkpoint.pt"
                  if args.step is not None else run/"last.pt")
        shutil.copyfile(source, copied)
        digest = sha256(copied)
        state = torch.load(copied, map_location="cpu", weights_only=False)
        if config_fingerprint(state["config"]) != config_fingerprint(config):
            raise ValueError("Checkpoint configuration differs from the running experiment")
        step = state["global_step"]
        if args.step is not None and step != args.step:
            raise ValueError("Archived checkpoint does not match the requested update")
        if step != state["stage_step"] or not 0 < step <= config["train"]["updates_per_stage"]:
            raise ValueError("Invalid one-step checkpoint counter")
        out = run/"checkpoint_previews"/f"step_{step:06d}"
        labels = {"project": "Global proj.", "norm_match": "Global norm",
                  "block_project": "Module proj.", "block_norm_match": "Module norm",
                  "agreement": "Agreement", "paired_mean": "Paired mean", "shifted_agreement": "Shifted map"}
        if study == "dmd_pair_agreement":
            labels.update(agreement="Pair agreement", paired_mean="Four-probe mean", shifted_agreement="Shifted pairs")
        for key, (mode, fraction) in SETTINGS.items():
            labels[key] = f"{mode.capitalize()} {fraction:.0%}"
        display_labels = {"dmd_perceptual": "DMD + LPIPS", config["name"]: labels[args.mode]}
        out.mkdir(parents=True, exist_ok=True)
        snapshot = out/"checkpoint.pt"
        if snapshot.exists() and sha256(snapshot) != digest:
            raise ValueError("Existing checkpoint snapshot differs at the same step")
        if not snapshot.exists():
            copied.replace(snapshot)
        if (out/"samples.json").exists():
            saved = json.loads((out/"samples.json").read_text())
            if saved["checkpoint_sha256"] != digest:
                raise ValueError("Existing diagnostic came from a different checkpoint")
            draw_samples(saved["records"], list(range(len(saved["records"][0]["sample_prompts"]))),
                out/"comparison", display_labels=display_labels)
            print(f"Verified existing checkpoint preview: {out}")
            return
        metadata = {"method": config["name"], "seed": config["seed"],
            "initial_steps": config["train"]["initial_steps"], "global_step": step,
            "cumulative_training_updates": lineage["parent_training_updates"]+step,
            "status": "qualitative", "checkpoint_sha256": digest}
        export_adapter(state["ema"], out/"adapter", torch.tensor([999]), config, metadata)
        del state
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        if free < (args.memory_gib+1)*2**30:
            raise RuntimeError("Insufficient free memory for checkpoint preview")
        torch.cuda.set_per_process_memory_fraction(args.memory_gib*2**30/total, 0)
        bundle, _ = low_memory_bundle(config, device, out/"adapter")
        payload = json.loads((ROOT/"results/sd15/active_one_step/main_comparison.json").read_text())
        rows = payload["rows"]
        reference = payload["methods"]["dmd_perceptual"]
        if reference["model_identity"] != manifest["reference_baseline"]["model_identity"]:
            raise ValueError("Reference figure differs from the frozen DMD checkpoint")
        generator = torch.Generator().manual_seed(payload["eval_seed"])
        noise = torch.cat([torch.randn((2, 4, 64, 64), generator=generator) for _ in range(4)])
        images = []
        grid = torch.tensor([999], device=device)
        for row in tqdm(rows, desc=f"{args.mode} checkpoint {step}", unit="image"):
            z = noise[row["noise_stream_index"]:row["noise_stream_index"]+1]
            if hashlib.sha256(z.numpy().tobytes()).hexdigest() != row["noise_sha256"]:
                raise ValueError("Initial noise differs from the fixed reference figure")
            pixels = generate_sd(bundle, bundle.unet, grid, [row["prompt"]],
                                 payload["eval_seed"], 1, noise=z.to(device))
            if not torch.isfinite(pixels).all():
                raise FloatingPointError("Nonfinite checkpoint sample")
            array = ((pixels[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="PNG")
            images.append(base64.b64encode(buffer.getvalue()).decode())
        common = {"sample_prompts": [r["prompt"] for r in rows],
            "image_size": 512, "dataset": "COCO-2017",
            "sample_image_ids": [r["key"] for r in rows], "sample_labels": [r["label"] for r in rows],
            "eval_seed": payload["eval_seed"], "evaluation_prompt_hash": payload["protocol_hash"],
            "sampling_steps": 1, "nfe": 1, "precision": config["precision"]}
        columns = [{**common, "method": "dmd_perceptual", "model_identity": reference["model_identity"],
            "source_file": reference["source_file"],
            "sample_images_png_base64": [r["png_base64"] for r in reference["rows"]]},
            {**common, **metadata, "source_file": str(snapshot.relative_to(ROOT)),
             "model_identity": sha256(out/"adapter/adapter.safetensors"),
             "sample_images_png_base64": images}]
        selection = draw_samples(columns, list(range(len(rows))), out/"comparison",
            display_labels=display_labels)
        atomic_json(out/"samples.json", {"status": "qualitative", "checkpoint_sha256": digest,
            "cumulative_training_updates": metadata["cumulative_training_updates"],
            "selection": selection, "records": columns,
            "note": "Prespecified illustrative prompts; no quantitative improvement claim."})
        print(f"Native checkpoint preview: {out}", flush=True)


if __name__ == "__main__":
    main()
