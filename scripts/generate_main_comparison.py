#!/usr/bin/env python3
"""Generate matched dog, teddy-bear and lighthouse rows for the active paper."""

import argparse
import base64
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("dmd_perceptual_diversity", "baseline", "consistency", "dmd",
           "swiftbrush_lora", "instaflow")
OUTPUT = Path("results/sd15/active_one_step/main_comparison.json")


def refresh(gpu, memory_gib):
    import torch
    from PIL import Image
    from tqdm.auto import tqdm
    from build_active_report import load_result, verify_common
    from generate_active_qualitative import low_memory_bundle
    from few_step_diffusion.runtime import atomic_json
    from few_step_diffusion.sd_models import generate_sd

    plan = json.loads((ROOT/"configs/active_study.json").read_text())
    sources = {j["method"]: ROOT/j["result_dir"]/f"{j['method']}_seed42_1steps.result.json"
               for j in plan["jobs"] if j["method"] in METHODS}
    records = {m: load_result(ROOT, p) for m, p in sources.items() if p.exists()}
    teacher = load_result(ROOT, ROOT/"results/sd15/scale50k/teacher_seed0_64steps.result.json")
    verify_common([*records.values(), teacher])
    preview = json.loads((ROOT/"report/assets/teacher_preview_sd15.json").read_text())["records"][0]
    if (preview["model_identity"] != teacher["model_identity"] or
            preview["teacher_guidance_scale"] != teacher["teacher_guidance_scale"]):
        raise ValueError("Teacher preview differs from the evaluated teacher")
    reference = records["dmd_perceptual_diversity"]
    generator = torch.Generator().manual_seed(12345)
    # Reproduce the original teacher preview's CPU noise stream, two at a time.
    noise = torch.cat([torch.randn((2, 4, 64, 64), generator=generator) for _ in range(4)])
    descriptions = [("dog", 7, "preset", "Dog in flowers"),
                    ("teddy", 4, "validation", "COCO 776"),
                    ("lighthouse", 3, "preset", "Lighthouse")]
    rows = []
    for key, index, source, label in descriptions:
        prompt = (reference if source == "validation" else preview)["sample_prompts"][index]
        rows.append({"key": key, "prompt": prompt, "label": label, "source": source,
            "noise_stream_index": index,
            "noise_sha256": hashlib.sha256(noise[index].numpy().tobytes()).hexdigest()})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    path = ROOT/OUTPUT
    payload = json.loads(path.read_text()) if path.exists() else {
        "schema_version": 1, "status": "qualitative_selection", "protocol": "matched-main-figure-v1",
        "eval_seed": 12345, "image_size": 512, "rows": rows, "protocol_hash": digest,
        "noise_protocol": "CPU torch.Generator seed 12345; eight latent samples in batches of two",
        "methods": {}}
    if payload["rows"] != rows or payload["protocol_hash"] != digest:
        raise ValueError("Main comparison has a different prompt/noise protocol")
    if "teacher" not in payload["methods"]:
        payload["methods"]["teacher"] = {"model_identity": teacher["model_identity"],
            "sampling_steps": 64, "nfe": 128, "precision": teacher["precision"],
            "rows": [{**row, "png_base64": (teacher if row["source"] == "validation" else preview)
                      ["sample_images_png_base64"][row["noise_stream_index"]]} for row in rows]}
        atomic_json(path, payload)
    for method in METHODS:
        if method not in records:
            continue
        record = records[method]
        if method in payload["methods"]:
            if payload["methods"][method]["model_identity"] != record["model_identity"]:
                raise ValueError("Main comparison uses an outdated checkpoint")
            continue
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        allowance = int(memory_gib*2**30)
        if free < allowance+2**30:
            raise RuntimeError(f"GPU {gpu} needs {memory_gib+1:g} GiB free for figure inference")
        torch.cuda.set_per_process_memory_fraction(allowance/total, 0)
        config = record["resolved_config"]
        bundle = None
        try:
            bundle, metadata = low_memory_bundle(config, device, ROOT/config["output_dir"]/"stage_1")
            grid = torch.tensor(metadata["timesteps"], device=device)
            items = []
            for row in tqdm(rows, desc=f"Main figure: {method}", unit="image"):
                index = row["noise_stream_index"]
                if row["source"] == "validation":
                    if record["sample_prompts"][index] != row["prompt"]:
                        raise ValueError("Teddy-bear comparison caption differs across methods")
                    encoded = record["sample_images_png_base64"][index]
                else:
                    pixels = generate_sd(bundle, bundle.unet, grid, [row["prompt"]],
                        12345, 1, noise=noise[index:index+1].to(device))
                    if not torch.isfinite(pixels).all():
                        raise FloatingPointError("Nonfinite main-figure image")
                    array = ((pixels[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()
                    buffer = io.BytesIO()
                    Image.fromarray(array).save(buffer, format="PNG")
                    encoded = base64.b64encode(buffer.getvalue()).decode()
                    del pixels
                items.append({**row, "png_base64": encoded})
            payload["methods"][method] = {"model_identity": record["model_identity"],
                "sampling_steps": 1, "nfe": 1, "precision": record["precision"],
                "cumulative_training_updates": 36000, "source_file": record["source_file"], "rows": items}
            atomic_json(path, payload)
            print(f"Main figure complete: {method}", flush=True)
        finally:
            if bundle is not None:
                bundle.embeddings = None
                bundle.decode = None
                del bundle
            gc.collect()
            torch.cuda.empty_cache()
    return all(m in payload["methods"] for m in METHODS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--memory-gib", type=float, default=4)
    parser.add_argument("--watch", action="store_true", help="Also generate newly evaluated baseline rows")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    from few_step_diffusion.runtime import experiment_lock
    with experiment_lock(ROOT/"artifacts/main_comparison"):
        while True:
            try:
                complete = refresh(args.gpu, args.memory_gib)
            except Exception as exc:
                if not args.watch:
                    raise
                print(f"Main comparison will retry: {exc}", flush=True)
                complete = False
            if complete or not args.watch:
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
