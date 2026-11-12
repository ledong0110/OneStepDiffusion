"""Optional execution optimizations; the recent-baseline recipe stays unchanged."""
import os
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .vae_cache import PosteriorBatch


def options():
    enabled = set(filter(None, os.environ.get("FSD_RECENT_OPTIMIZATIONS", "").split(",")))
    if enabled-{"fused_adamw", "no_checkpoint", "vae_helper"}:
        raise ValueError("Unknown recent-baseline optimization")
    return enabled


def configure_models(student, fake, enabled):
    if "no_checkpoint" in enabled:
        student.disable_gradient_checkpointing()
        fake.disable_gradient_checkpointing()


def configure_optimizers(optimizers, enabled, device):
    if "fused_adamw" not in enabled:
        return
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["fused"], group["foreach"] = True, False
        # Old checkpoints keep Adam's scalar step on CPU. Fused CUDA Adam
        # requires it on the parameter device; moments and counts are retained.
        for state in optimizer.state.values():
            if "step" in state:
                state["step"] = state["step"].to(device)


def micro_factor(accumulation):
    factor = int(os.environ.get("FSD_RECENT_MICRO_FACTOR", "1"))
    if factor < 1 or accumulation % factor:
        raise ValueError("Execution microbatch factor must divide gradient accumulation")
    return factor


def execution_metadata(config):
    factor = micro_factor(config["train"]["grad_accumulation"])
    metadata = {"optimizations": sorted(options()), "micro_factor": factor,
            "execution_microbatch": config["data"]["batch_size"]*factor,
            "execution_accumulation": config["train"]["grad_accumulation"]//factor,
            "effective_batch_size": config["data"]["batch_size"]*config["train"]["grad_accumulation"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "vae_helper_memory_gib": os.environ.get("FSD_VAE_HELPER_MEMORY_GIB")}
    if os.environ.get("FSD_RECENT_VAE_CACHE"):
        saved = json.loads((Path(os.environ["FSD_RECENT_VAE_CACHE"])/"metadata.json").read_text())
        if saved.get("status") != "complete":
            raise ValueError("Training cannot use an incomplete posterior cache")
        metadata.update(posterior_cache_sha256=saved["posterior_sha256"], vae_decoder_device="cpu",
                        posterior_sampling="fresh-noise-on-primary-gpu")
    return metadata


def grouped_microbatch(loader, stream, first, device, real, factor):
    if factor == 1:
        return loader(stream, first, device, real=real)
    parts = [loader(stream, first+i, device, real=real) for i in range(factor)]
    if real and isinstance(parts[0][0], PosteriorBatch):
        images = PosteriorBatch(torch.cat([p[0].mean for p in parts]), torch.cat([p[0].std for p in parts]))
    else:
        images = torch.cat([part[0] for part in parts]) if real else None
    return images, [prompt for _, prompts in parts for prompt in prompts]


class RealEncoder:
    """Overlap frozen VAE encoding on a helper GPU with generator inference.

    Posterior noise is still drawn on the training GPU at its original point
    in the loop, preserving that device's random-number sequence.
    """
    def __init__(self, bundle, enabled):
        self.bundle = bundle
        self.main = getattr(bundle, "device", next(bundle.unet.parameters()).device)
        if self.main.type == "cuda":
            self.main = torch.device("cuda", self.main.index if self.main.index is not None else torch.cuda.current_device())
            bundle.device = self.main
        self.device = self.main
        self.enabled = "vae_helper" in enabled
        if os.environ.get("FSD_RECENT_VAE_CACHE"):
            # Real-image encoding now comes from immutable posterior statistics.
            # CPU decoding is needed only for occasional previews; no helper GPU
            # allocation can interrupt regular cached training.
            self.device = torch.device("cpu")
            bundle.vae.to(self.device)
            def decode_cached(latents):
                values = [bundle.vae.decode(part.float().cpu()/bundle.vae.config.scaling_factor).sample.to(self.main)
                          for part in latents.split(1)]
                return torch.cat(values)
            bundle.decode = decode_cached
            return
        if self.enabled:
            self.device = torch.device(os.environ.get("FSD_VAE_HELPER_DEVICE", "cuda:1"))
            main_index = self.main.index if self.main.index is not None else torch.cuda.current_device()
            if self.device.type != "cuda" or self.device.index == main_index:
                raise ValueError("VAE helper must be a separate CUDA device")
            free, total = torch.cuda.mem_get_info(self.device)
            cap = float(os.environ.get("FSD_VAE_HELPER_MEMORY_GIB", "3"))
            if free < (cap+.5)*2**30:
                raise RuntimeError("Not enough memory on the VAE helper GPU")
            with torch.cuda.device(self.device):
                torch.cuda.set_per_process_memory_fraction(cap*2**30/total, self.device)
            bundle.vae.to(self.device)
            torch.cuda.set_device(self.main)
            # Checkpoint previews still decode on this frozen VAE. Transfer
            # their latents explicitly instead of leaving a device mismatch.
            def decode(latents):
                values = [bundle.vae.decode(part.to(self.device).float()/bundle.vae.config.scaling_factor).sample.to(self.main)
                          for part in latents.split(1)]
                return torch.cat(values)
            bundle.decode = decode

    @torch.no_grad()
    def start(self, images):
        if isinstance(images, PosteriorBatch):
            return images
        if not self.enabled:
            return None
        parts = [self.bundle.vae.encode(part).latent_dist for part in images.split(2)]
        return SimpleNamespace(mean=torch.cat([part.mean for part in parts]),
                               std=torch.cat([part.std for part in parts]))

    @torch.no_grad()
    def finish(self, images, posterior):
        if posterior is None:
            return self.bundle.encode(images)
        mean = posterior.mean.to(self.main, non_blocking=True)
        std = posterior.std.to(self.main, non_blocking=True)
        noise = torch.randn(mean.shape, device=self.main, dtype=mean.dtype)
        return (mean+std*noise)*self.bundle.vae.config.scaling_factor
