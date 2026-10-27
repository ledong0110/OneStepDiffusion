"""Fixed-caption COCO evaluation, including text alignment and end-to-end GPU latency."""

import base64
import io
from pathlib import Path
import time
import uuid
import yaml

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm.auto import tqdm

from .config import config_fingerprint, experiment_config
from .diffusion import make_grid, nested_grid
from .distillation_objectives import CONSISTENCY_SAMPLER, FLOW_SAMPLER, PRECONDITIONED_FLOW_SAMPLER
from .evaluation import metric_cache_context
from .runtime import atomic_json, device_for, provenance, seed_everything, sha256, synchronize
from .sd_data import COCOReferences, caption_records, manifest_hash
from .sd_models import SDBundle, generate_sd, restore_export

CLIP_SOURCE = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
SD_SAMPLER = "SD15-DDIM-eta0-unclipped-nested-v1"
SWIFTBRUSH_SOURCE_SAMPLER = "SD15-SwiftBrush-source-x0-v1"
TIMING_SCOPE = "prompt-encoding+denoising+VAE-decoding"


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_SOURCE, revision=CLIP_REVISION,
        use_safetensors=False).to(device).eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(CLIP_SOURCE, revision=CLIP_REVISION, use_fast=False)
    return model, processor


@torch.no_grad()
def text_alignment(model, processor, images, prompts, device):
    inputs = processor(text=prompts, images=[Image.open(path).convert("RGB") for path in images],
                       padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model(**inputs)
    cosine = (outputs.image_embeds*outputs.text_embeds).sum(-1)
    return 100*cosine.clamp_min(0)


class DiskImages(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)


def benchmark_sd(bundle, model, grid, config, prompts, guidance):
    ev, device = config["eval"], bundle.device
    count = ev["benchmark_batch_size"]
    prompts = (prompts*((count+len(prompts)-1)//len(prompts)))[:count]
    noise = torch.randn((count, 4, 64, 64), generator=torch.Generator().manual_seed(ev["seed"])).to(device)
    for _ in range(ev["benchmark_warmup"]):
        generate_sd(bundle, model, grid, prompts, ev["seed"], guidance, noise=noise)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(ev["benchmark_repeats"]):
        synchronize(device)
        start = time.perf_counter()
        generate_sd(bundle, model, grid, prompts, ev["seed"], guidance, noise=noise)
        synchronize(device)
        timings.append(1000*(time.perf_counter()-start)/count)
    return {"latency_ms_per_image": float(np.median(timings)),
            "latency_p95_ms_per_image": float(np.percentile(timings, 95)),
            "benchmark_batch_size": count, "benchmark_repeats": len(timings),
            "latency_scope": TIMING_SCOPE,
            "peak_inference_memory_mb": torch.cuda.max_memory_allocated(device)/2**20
            if device.type == "cuda" else 0}


def evaluate_sd(config, checkpoint="teacher", steps=None, result_dir="results", skip_metrics=False):
    seed_everything(config["seed"])
    device = device_for(config)
    ev = config["eval"]
    records = caption_records(config, "validation", ev["num_samples"])
    references = caption_records(config, "validation", ev["num_real"])
    training_prompt_hash = manifest_hash(caption_records(config))
    bundle = SDBundle(config, device)
    root = make_grid(len(bundle.alphas), config["train"]["initial_steps"], device)
    teacher_id = config["model"]["source"] + "@" + config["model"]["revision"]
    metadata = {}
    if checkpoint == "teacher":
        grid = nested_grid(root, steps or len(root))
        method, train_seed, updates = "teacher", 0, 0
        model_identity = teacher_id
        guidance = config["model"]["guidance_scale"]
    else:
        metadata = restore_export(bundle, checkpoint)
        training_config = Path(checkpoint).parent/"config.yaml"
        if training_config.is_file():
            saved_train = yaml.safe_load(training_config.read_text())["train"]
            for key in ("edge_weight", "edge_schedule"):
                metadata.setdefault(key, saved_train[key])
        if metadata.get("training_prompt_hash") != training_prompt_hash:
            raise ValueError("Evaluation training-caption manifest differs from the checkpoint")
        grid = torch.tensor(metadata["timesteps"], dtype=torch.long, device=device)
        if steps is not None and steps != len(grid):
            raise ValueError("Use the distilled checkpoint's saved inference grid")
        method, train_seed, updates = metadata["method"], metadata["seed"], metadata["global_step"]
        model_identity = sha256(Path(checkpoint)/(
            "diffusion_pytorch_model.safetensors" if metadata.get("student_parameterization") == "full-unet"
            else "adapter.safetensors"))
        guidance = 1.0
        if ev.get("comparison_steps") is not None and len(grid) != ev["comparison_steps"]:
            raise ValueError("Checkpoint step count differs from the matched student comparison")
    path = Path(result_dir) / f"{method}_seed{train_seed}_{len(grid)}steps.result.json"
    if path.exists():
        raise FileExistsError(f"Result already exists: {path}")
    env = provenance(config, device)
    prompts = [row["caption"] for row in records]
    timing = benchmark_sd(bundle, bundle.unet, grid, config, prompts[:8], guidance)
    artifact = Path("artifacts/eval") / f"sd15_{method}_seed{train_seed}_{len(grid)}steps_{uuid.uuid4().hex[:8]}"
    artifact.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(ev["seed"])
    paths, first_images, portable = [], [], []
    with tqdm(total=len(records), desc=f"SD {method} | {len(grid)} steps", unit="image",
              dynamic_ncols=True, disable=not config.get("progress_bar", True)) as progress:
        for start in range(0, len(records), ev["batch_size"]):
            batch_prompts = prompts[start:start+ev["batch_size"]]
            noise = torch.randn((len(batch_prompts), 4, 64, 64), generator=generator).to(device)
            images = generate_sd(bundle, bundle.unet, grid, batch_prompts, ev["seed"], guidance, noise=noise)
            if not torch.isfinite(images).all():
                raise FloatingPointError("Nonfinite SD images")
            images = ((images.clamp(-1, 1)+1)*127.5).round().byte().cpu()
            for offset, pixels in enumerate(images):
                destination = artifact / f"{start+offset:05d}.png"
                image = Image.fromarray(pixels.permute(1, 2, 0).numpy())
                image.save(destination)
                paths.append(destination)
                if len(first_images) < 8:
                    first_images.append(pixels)
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    portable.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
            progress.update(len(batch_prompts))
    save_image(torch.stack(first_images).float()/255, artifact/"samples.png", nrow=4)
    metrics = {"fid": None, "inception_score_mean": None, "inception_score_std": None,
               "clip_score_mean": None, "clip_score_std": None}
    # Release SD weights before loading Inception/CLIP to minimize evaluation VRAM.
    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not skip_metrics:
        if len(records) < 10:
            raise ValueError("Use --skip-metrics for fewer than 10 images")
        import torch_fidelity
        real = COCOReferences(config, ev["num_real"])
        with metric_cache_context():
            raw = torch_fidelity.calculate_metrics(input1=DiskImages(paths), input2=real,
                input2_cache_name=f"coco2017_center512_{manifest_hash(references)}",
                cache_root=str(Path(config["data"]["root"])/"metric_cache"), cuda=device.type == "cuda",
                batch_size=ev["batch_size"], fid=True, isc=True, kid=False, prc=False, verbose=True)
        clip_model, processor = load_clip(device)
        scores = []
        for start in tqdm(range(0, len(paths), ev["batch_size"]), desc="CLIP alignment",
                          disable=not config.get("progress_bar", True)):
            scores.extend(text_alignment(clip_model, processor, paths[start:start+ev["batch_size"]],
                prompts[start:start+ev["batch_size"]], device).cpu().tolist())
        metrics = {"fid": float(raw["frechet_inception_distance"]),
                   "inception_score_mean": float(raw["inception_score_mean"]),
                   "inception_score_std": float(raw["inception_score_std"]),
                   "clip_score_mean": float(np.mean(scores)), "clip_score_std": float(np.std(scores))}
    result = {"schema_version": 1, "status": "smoke" if skip_metrics else "complete",
              "resolved_config": experiment_config(config),
              "config_hash": config_fingerprint(config), "method": method, "seed": train_seed,
              **({"comparison_steps": ev["comparison_steps"]} if "comparison_steps" in ev else {}),
              "sampling_steps": len(grid), "nfe": len(grid)*(2 if guidance > 1 else 1),
              "denoiser_branches_per_step": 2 if guidance > 1 else 1,
              "dataset": "COCO-2017", "image_size": 512, "num_samples": ev["num_samples"],
              "num_real": ev["num_real"], "reference_split": "validation", "eval_seed": ev["seed"],
              "sampler": SWIFTBRUSH_SOURCE_SAMPLER if metadata.get("student_prediction_type") == "swiftbrush-source-x0"
                         else CONSISTENCY_SAMPLER if metadata.get("student_prediction_type") == "consistency-x0"
                         else PRECONDITIONED_FLOW_SAMPLER if metadata.get("student_prediction_type") == "epsilon-converted-velocity"
                         else FLOW_SAMPLER if metadata.get("student_prediction_type") == "rectified-flow-velocity"
                         else SD_SAMPLER, "precision": config["precision"],
              "metric_implementation": "torch-fidelity-0.3.0", "training_updates": updates,
              "clip_model_identity": CLIP_SOURCE+"@"+CLIP_REVISION,
              "clip_score_definition": "100*max(cosine(image,text),0); mean over caption-image pairs",
              "teacher_guidance_scale": config["model"]["guidance_scale"], "guidance_scale": guidance,
              "lora_rank": 0 if config["train"].get("vsd_recipe") == "swiftbrush-source-v1"
                           and config["train"].get("student_parameterization", "full-unet") == "full-unet"
                           else config["model"]["lora_rank"],
              "student_parameterization": "teacher" if checkpoint == "teacher"
                                          else metadata.get("student_parameterization", "lora"),
              "prompt_protocol": "coco2017-first-caption-by-id-v1",
              "evaluation_prompt_hash": manifest_hash(records),
              "reference_manifest_hash": manifest_hash(references),
              "distillation_prompt_hash": training_prompt_hash,
              "training_seconds": metadata.get("training_seconds", 0),
              "student_prediction_type": metadata.get("student_prediction_type", "epsilon"),
              **{key: metadata[key] for key in ("consistency_sigma_data", "dmd_regression_weight",
                  "teacher_pair_count", "teacher_pair_generation_seconds", "teacher_pair_nfe",
                  "paired_regression_batch_size", "edge_weight", "edge_schedule", "adaptation",
                  "reflow_pair_nfe", "reflow_pair_generation_seconds", "reflow_updates",
                  "flow_distillation_updates", "flow_time_mapping", "source_commit", "source_url",
                  "terminal_alpha_cumprod", "auxiliary_lora_rank", "auxiliary_lora_alpha",
                  "auxiliary_lora_targets", "auxiliary_trainable_parameters", "dmd_study_policy",
                  "dmd_study_parent_config_hash", "dmd_regression_type", "dmd_diversity_weight",
                  "dmd_pair_seeds", "teacher_pair_unique_prompts", "teacher_pair_protocol_hash",
                  "teacher_pair_weights_sha256", "perceptual_metric") if key in metadata},
              "training_hardware": metadata.get("training_hardware"),
              "auxiliary_updates": metadata.get("auxiliary_updates", 0),
              "training_uses_real_images": metadata.get("training_uses_real_images", False),
              "effective_batch_size": metadata.get("effective_batch_size", 0),
              "teacher_identity": teacher_id, "initial_steps": config["train"]["initial_steps"],
              "peak_training_memory_mb": metadata.get("peak_training_memory_mb", 0),
              "trainable_parameters": metadata.get("trainable_parameters", 0),
              "timesteps": grid.cpu().tolist(), "model_identity": model_identity,
              "sample_grid": str(artifact/"samples.png"), "sample_images_png_base64": portable,
              "sample_prompts": prompts[:8], "sample_image_ids": [row["image_id"] for row in records[:8]],
              "environment": env, **metrics, **timing}
    for key in ("continuation", "continuation_only_steps", "cumulative_training_updates",
                "cumulative_training_seconds", "cumulative_auxiliary_updates",
                "continuation_policy", "parent_training_prompt_hash", "parent_training_updates"):
        if key in metadata:
            result[key] = metadata[key]
    result["generated_images_retained"] = (len(paths) if ev.get("retain_generated_images", True)
                                           else min(8, len(paths)))
    atomic_json(path, result)
    if not ev.get("retain_generated_images", True):
        # Only remove images produced by this evaluation, after saving all metrics.
        for generated in paths[8:]:
            generated.unlink()
    print(f"Result: {path}", flush=True)
    return path
