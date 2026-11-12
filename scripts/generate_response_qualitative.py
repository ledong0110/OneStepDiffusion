#!/usr/bin/env python3
"""Generate prompt control and embedding interpolation from an evaluated response model.

The prompts and displayed seeds come from the existing report selection. They
are fixed before seeing the new model's images; earlier outputs are preserved.
"""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("dmd_response_strength", "dmd_response_final", "independent_response_36k"), default="dmd_response_final")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gpu", default="2")
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
    from analyze_teacher_responses import completed_source
    from generate_active_qualitative import low_memory_bundle
    from few_step_diffusion.qualitative import load_qualitative, panel_specs, protocol_hash
    from few_step_diffusion.runtime import atomic_json, experiment_lock, sha256
    from few_step_diffusion.sd_models import generate_sd

    torch.set_num_threads(2)
    independent = args.study == "independent_response_36k"
    if independent:
        from few_step_diffusion.config import load_config
        from few_step_diffusion.independent_response import validate_result
        plan = json.loads((ROOT/"configs/independent_response_36k.json").read_text())
        job = next(j for j in plan["jobs"] if j["method"] == args.variant)
        config = load_config(ROOT/job["config"])
        run = ROOT/job["run_dir"]
        source = ROOT/"results/sd15/independent36k"/f"{args.variant}_seed{config['seed']}_1steps.result.json"
        record = json.loads(source.read_text())
        validate_result(record)
        if record["num_samples"] != 5000 or record["num_real"] != 5000:
            raise ValueError("Independent qualitative panels require completed full-validation metrics")
        if sha256(run/"stage_1/adapter.safetensors") != record["model_identity"]:
            raise ValueError("Independent qualitative model differs from the evaluated checkpoint")
    else:
        run, config, plan = completed_source(ROOT, args.study, args.variant)
        job = next(j for j in plan["jobs"] if j["key"] == args.variant)
        source = ROOT/job["result_dir"]/f"{job['method']}_seed{plan['seed']}_1steps.result.json"
        record = json.loads(source.read_text())
        if record.get("diversity_lpips_alex_mean") is None:
            raise ValueError("Final qualitative generation waits for quality and diversity evaluation")
    selection = json.loads((ROOT/"configs/report_selection.json").read_text())
    eval_seed = selection["prompt_control_eval_seed"]
    seeds = selection["interpolation_noise_seeds"]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Interpolation requires distinct prespecified noise seeds")
    output = run/"qualitative"
    identity = {key: record[key] for key in ("method", "model_identity", "seed", "precision",
        "teacher_identity", "lora_rank", "training_updates", "cumulative_training_updates")}
    identity.update(sampling_steps=1, nfe=1, source_result=str(source.relative_to(ROOT)),
                    source_result_sha256=sha256(source),
                    inference_auxiliary_device="CPU text encoder and VAE; native-precision GPU U-Net")
    control_path = output/f"seed{eval_seed}"/f"{record['method']}_1steps.qualitative.json"
    control_identity = {**identity, "eval_seed": eval_seed, "protocol_hash": protocol_hash(eval_seed)}
    interpolation_path = output/"prompt_interpolation.json"
    prompts = [f"A DSLR photograph of a {animal} reading an open book, sitting at an outdoor cafe table."
               for animal in ("dog", "cat")]
    weights = [0, .25, .5, .75, 1]
    interpolation_identity = {**identity, "status": "qualitative_interpolation",
        "protocol": "linear-clip-hidden-state-v1", "prompts": prompts, "weights": weights,
        "candidate_noise_seeds": seeds, "selection_policy": "Reuse prespecified earlier-report seeds without reselection."}

    def encode(pixels):
        if not torch.isfinite(pixels).all():
            raise FloatingPointError("Nonfinite native qualitative output")
        array = ((pixels[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    with experiment_lock(run/"qualitative_generation"):
        main_path = output/"main_comparison.json"
        main = json.loads(main_path.read_text()) if independent and main_path.exists() else None
        if main and main["methods"][record["method"]]["model_identity"] != record["model_identity"]:
            raise ValueError("Saved main figure belongs to another evaluated checkpoint")
        control = json.loads(control_path.read_text()) if control_path.exists() else None
        if control:
            if any(control.get(k) != v for k, v in control_identity.items()):
                raise ValueError("Saved prompt control belongs to a different evaluated model or protocol")
            load_qualitative(control_path.parent, [record])
        interpolation = (json.loads(interpolation_path.read_text()) if interpolation_path.exists()
                         else {**interpolation_identity, "rows": []})
        if any(interpolation.get(k) != v for k, v in interpolation_identity.items()):
            raise ValueError("Saved interpolation belongs to a different evaluated model or protocol")
        completed_seeds = [row["noise_seed"] for row in interpolation["rows"]]
        if len(set(completed_seeds)) != len(completed_seeds) or any(s not in seeds for s in completed_seeds):
            raise ValueError("Saved interpolation has unexpected or duplicate noise seeds")
        for row in interpolation["rows"]:
            if [item["weight"] for item in row["images"]] != weights:
                raise ValueError("Saved interpolation weights differ")
            for item in row["images"]:
                image = Image.open(io.BytesIO(base64.b64decode(item["png_base64"], validate=True)))
                if image.format != "PNG" or image.size != (512, 512):
                    raise ValueError("Saved interpolation images must be native 512x512 PNGs")
        if control and set(completed_seeds) == set(seeds) and (not independent or main):
            print(f"Verified existing evaluated-model qualitative panels: {output}")
            return
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        if args.memory_gib <= 0 or free < (args.memory_gib+1)*2**30:
            raise RuntimeError("Insufficient free memory for final qualitative generation")
        torch.cuda.set_per_process_memory_fraction(args.memory_gib*2**30/total, 0)
        bundle, metadata = low_memory_bundle(config, device, run/"stage_1")
        bundle.unet.eval()
        grid = torch.tensor(metadata["timesteps"], device=device)
        if grid.tolist() != [999]:
            raise ValueError("Final qualitative generation requires the evaluated one-step endpoint")
        if independent and main is None:
            protocol = json.loads((ROOT/selection["main_comparison_result"]).read_text())
            rows = protocol["rows"]
            generator = torch.Generator().manual_seed(record["eval_seed"])
            noise = torch.cat([torch.randn((2, 4, 64, 64), generator=generator) for _ in range(4)])
            images = []
            for row in tqdm(rows, desc="Independent proposal main figure"):
                index = row["noise_stream_index"]
                if hashlib.sha256(noise[index].numpy().tobytes()).hexdigest() != row["noise_sha256"]:
                    raise ValueError("Main figure initial noise differs from the baseline columns")
                if row["source"] == "validation":
                    if record["sample_prompts"][index] != row["prompt"]:
                        raise ValueError("Main validation caption differs from the evaluated output")
                    png = record["sample_images_png_base64"][index]
                else:
                    pixels = generate_sd(bundle, bundle.unet, grid, [row["prompt"]],
                        record["eval_seed"], 1, noise=noise[index:index+1].to(device))
                    png = encode(pixels)
                images.append({**row, "png_base64": png})
            atomic_json(main_path, {"schema_version": 1, "protocol": "matched-main-figure-v1",
                "eval_seed": record["eval_seed"], "image_size": 512, "rows": rows,
                "protocol_hash": protocol["protocol_hash"], "methods": {record["method"]: {
                    **identity, "rows": images}}})
        if control is None:
            panels = {}
            for kind in ("comparison", "control"):
                panels[kind] = []
                for item in tqdm(panel_specs(eval_seed)[kind], desc=f"{args.variant} {kind}"):
                    noise = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(item["noise_seed"]))
                    pixels = generate_sd(bundle, bundle.unet, grid, [item["prompt"]], eval_seed,
                                         guidance=1, noise=noise.to(device))
                    panels[kind].append({**item, "png_base64": encode(pixels)})
            atomic_json(control_path, {"schema_version": 1, "status": "qualitative", "image_size": 512,
                **control_identity, "guidance_scale": 1, "panels": panels,
                "adaptation": "Evaluated rank-32 response extension; prespecified earlier-report prompts and noise."})
        dog, cat = [bundle.embeddings([prompt]) for prompt in prompts]
        for seed in tqdm(seeds, desc=f"{args.variant} embedding interpolation"):
            if seed in completed_seeds:
                continue
            noise = torch.randn((1, 4, 64, 64), generator=torch.Generator().manual_seed(seed)).to(device)
            fixed = noise.clone()
            items = []
            for weight in weights:
                pixels = generate_sd(bundle, bundle.unet, grid, [prompts[0]], seed, 1, noise=noise,
                    embeddings=(torch.lerp(dog, cat, weight), None))
                if not torch.equal(noise, fixed):
                    raise ValueError("Interpolation changed its fixed noise")
                items.append({"weight": weight, "png_base64": encode(pixels)})
            interpolation["rows"].append({"noise_seed": seed, "images": items})
            atomic_json(interpolation_path, interpolation)
    print(f"Native final-model qualitative panels: {output}", flush=True)


if __name__ == "__main__":
    main()
