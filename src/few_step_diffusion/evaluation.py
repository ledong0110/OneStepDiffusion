import json
import base64
import io
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torchvision.utils import save_image

from .data import dataset
from .config import config_fingerprint, dataset_name, image_size
from .diffusion import make_grid, nested_grid
from .models import load_model
from .runtime import (atomic_json, device_for, provenance, seed_everything, sha256, synchronize)
from .sampling import generate


def metric_cache_context():
    """Allow only NumPy statistics types used by locally computed FID caches.

    torch-fidelity 0.3 predates torch.load's weights_only default. Its own cached
    mean/covariance arrays need these types; do not disable weights_only globally.
    """
    from numpy._core.multiarray import _reconstruct

    return torch.serialization.safe_globals([
        _reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType,
    ])


class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


def benchmark(model, alphas, grid, config, device):
    ev = config["eval"]
    batch_size = ev["benchmark_batch_size"]
    generator = torch.Generator().manual_seed(ev["seed"])
    size = image_size(config)
    noise = torch.randn((batch_size, 3, size, size), generator=generator).to(device)
    for _ in range(ev["benchmark_warmup"]):
        generate(model, alphas, grid, batch_size, ev["seed"], config, device, noise=noise)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(ev["benchmark_repeats"]):
        synchronize(device)
        start = time.perf_counter()
        generate(model, alphas, grid, batch_size, ev["seed"], config, device, noise=noise)
        synchronize(device)
        timings.append(1000 * (time.perf_counter() - start) / batch_size)
    return {"latency_ms_per_image": float(np.median(timings)),
            "latency_p95_ms_per_image": float(np.percentile(timings, 95)),
            "benchmark_batch_size": batch_size, "benchmark_repeats": len(timings),
            "peak_inference_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda" else 0}


def evaluate(config, checkpoint="teacher", steps=None, result_dir="results", skip_metrics=False):
    from .config import is_stable_diffusion
    if is_stable_diffusion(config):
        from .sd_evaluation import evaluate_sd
        return evaluate_sd(config, checkpoint, steps, result_dir, skip_metrics)
    seed_everything(config["seed"])
    device = device_for(config)
    ev = config["eval"]
    size = image_size(config)
    data_name = dataset_name(config)
    metadata = {}
    if checkpoint == "teacher":
        model, scheduler = load_model(config)
        if config["model"]["source"] in ("scratch", "tiny") and not config["data"]["fake"]:
            raise ValueError("An untrained scratch model cannot be evaluated as the pretrained teacher")
        root = make_grid(scheduler.config.num_train_timesteps,
                         config["train"]["initial_steps"], device)
        grid = nested_grid(root, steps or len(root))
        method, train_seed, train_updates = "teacher", 0, 0
        source = Path(config["model"]["source"])
        model_identity = (sha256(source / "diffusion_pytorch_model.safetensors") if source.is_dir()
                          else config["model"]["source"] + "@" + config["model"]["revision"])
    else:
        checkpoint = Path(checkpoint)
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        model, scheduler = load_model(config, str(checkpoint))
        grid = torch.tensor(metadata["timesteps"], dtype=torch.long, device=device)
        if steps is not None and steps != len(grid):
            raise ValueError("A distilled checkpoint must use its saved inference grid")
        if metadata.get("fake_data") and not config["data"]["fake"]:
            raise ValueError("Smoke checkpoints cannot produce research results")
        method, train_seed = metadata["method"], metadata["seed"]
        train_updates = metadata["global_step"]
        model_identity = sha256(checkpoint / "diffusion_pytorch_model.safetensors")
    result_path = Path(result_dir) / f"{method}_seed{train_seed}_{len(grid)}steps.result.json"
    if result_path.exists():
        raise FileExistsError(f"Result already exists: {result_path}. Use a new --result-dir")
    model = model.to(device).eval().requires_grad_(False)
    alphas = scheduler.alphas_cumprod.to(device)
    env = provenance(config, device)
    timing = benchmark(model, alphas, grid, config, device)
    generated = []
    generator = torch.Generator().manual_seed(ev["seed"])
    with tqdm(total=ev["num_samples"], desc=f"{method} | {len(grid)} NFE", unit="image",
              dynamic_ncols=True, disable=not config.get("progress_bar", True)) as progress:
        for start in range(0, ev["num_samples"], ev["batch_size"]):
            count = min(ev["batch_size"], ev["num_samples"] - start)
            noise = torch.randn((count, 3, size, size), generator=generator).to(device)
            x = generate(model, alphas, grid, count, ev["seed"], config, device, noise=noise)
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite generated images")
            generated.append(((x.clamp(-1, 1) + 1) * 127.5).round().byte().cpu())
            progress.update(count)
    images = torch.cat(generated)
    artifact = Path("artifacts/eval") / f"{method}_seed{train_seed}_{len(grid)}steps_{uuid.uuid4().hex[:8]}"
    artifact.mkdir(parents=True, exist_ok=True)
    save_image(images[:64].float() / 255, artifact / "samples.png", nrow=8)
    portable_samples = []
    for pixels in images[:8]:
        buffer = io.BytesIO()
        Image.fromarray(pixels.permute(1, 2, 0).numpy()).save(buffer, format="PNG")
        portable_samples.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    metrics = {"fid": None, "inception_score_mean": None, "inception_score_std": None}
    if not skip_metrics:
        if config["data"]["fake"]:
            raise ValueError("Use --skip-metrics for smoke data; fake images are not research results")
        if ev["num_samples"] < 10:
            raise ValueError("FID/IS requires at least 10 generated images in this pipeline")
        import torch_fidelity

        real = dataset(config, ev["reference_split"])
        if ev["num_real"] > len(real):
            raise ValueError("num_real exceeds the reference split size")
        with metric_cache_context():
            raw = torch_fidelity.calculate_metrics(
                input1=ImageDataset(images), input2=ImageDataset(real[:ev["num_real"]]),
                input2_cache_name=(f"{data_name}_{size}_{ev['reference_split']}_{ev['num_real']}"
                                   f"_{config['data'].get('revision', 'original')}"),
                cache_root=str(Path(config["data"]["root"]) / "metric_cache"),
                cuda=device.type == "cuda", batch_size=ev["batch_size"],
                fid=True, isc=True, kid=False, prc=False, verbose=True,
            )
        metrics = {"fid": float(raw["frechet_inception_distance"]),
                   "inception_score_mean": float(raw["inception_score_mean"]),
                   "inception_score_std": float(raw["inception_score_std"])}
    result = {"schema_version": 1, "status": "smoke" if skip_metrics else "complete",
              "config_hash": config_fingerprint(config),
              "method": method, "seed": train_seed, "sampling_steps": len(grid),
              "nfe": len(grid), "dataset": data_name, "image_size": size,
              "num_samples": ev["num_samples"], "num_real": ev["num_real"],
              "reference_split": ev["reference_split"], "eval_seed": ev["seed"],
              "sampler": "DDIM-eta0-unclipped-nested-v1", "precision": config["precision"],
              "metric_implementation": "torch-fidelity-0.3.0", "training_updates": train_updates,
              "training_seconds": metadata.get("training_seconds", 0.0),
              "training_hardware": metadata.get("training_hardware"),
              "auxiliary_updates": metadata.get("auxiliary_updates", 0),
              "training_uses_real_images": metadata.get("training_uses_real_images", method != "teacher"),
              "effective_batch_size": metadata.get("effective_batch_size", 0),
              "teacher_identity": metadata.get("teacher_identity", model_identity),
              "initial_steps": metadata.get("initial_steps", config["train"]["initial_steps"]),
              "peak_training_memory_mb": metadata.get("peak_training_memory_mb", 0.0),
              "timesteps": grid.cpu().tolist(), "model_identity": model_identity,
              "sample_grid": str(artifact / "samples.png"), "environment": env,
              "sample_images_png_base64": portable_samples,
              **metrics, **timing}
    atomic_json(result_path, result)
    print(f"Result: {result_path}", flush=True)
    return result_path
