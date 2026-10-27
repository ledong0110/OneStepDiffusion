"""Unconditional pixel-space adaptation of SwiftBrush's VSD training objective.

This is not the original text-to-image/LoRA/CFG implementation. It uses the
same sigma^2-weighted score difference and alpha^2-weighted scorer denoising
loss, but updates a full auxiliary U-Net on CIFAR-10-sized synthetic samples.
Reference: https://github.com/VinAIResearch/SwiftBrush/blob/main/train_swiftbrush.py
"""

import copy
import json
import math
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from torchvision.utils import save_image

from .diffusion import coefficients, predict_x0
from .config import image_size
from .models import load_model
from .runtime import atomic_json, autocast, sha256
from .sampling import generate


def vsd_surrogate(images, real_epsilon, fake_epsilon, sigma):
    gradient = sigma.square() * (real_epsilon.float() - fake_epsilon.float())
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite VSD score gradient")
    target = (images.float() - gradient).detach()
    return 0.5 * (images.float() - target).square().mean()


def train_vsd(config, state, device, output, checkpoint, environment, tracker):
    from .training import export_model, save_checkpoint, update_ema

    tr = config["train"]
    teacher, scheduler = load_model(config)
    teacher = teacher.to(device).eval().requires_grad_(False)
    student = copy.deepcopy(teacher).train().requires_grad_(True)
    fake = copy.deepcopy(teacher).train().requires_grad_(True)
    ema = copy.deepcopy(teacher).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=tr["learning_rate"],
                                 weight_decay=tr["weight_decay"])
    fake_optimizer = torch.optim.AdamW(fake.parameters(), lr=tr.get("fake_learning_rate", 0.0001),
                                      weight_decay=tr["weight_decay"])
    alphas = scheduler.alphas_cumprod.to(device)
    grid = torch.tensor([len(alphas)-1], dtype=torch.long, device=device)
    first_step, previous_elapsed = 0, 0.0
    if state is not None:
        student.load_state_dict(state["student"])
        ema.load_state_dict(state["ema"])
        fake.load_state_dict(state["fake"])
        optimizer.load_state_dict(state["optimizer"])
        fake_optimizer.load_state_dict(state["fake_optimizer"])
        first_step, previous_elapsed = state["global_step"], state["elapsed_seconds"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    source = Path(config["model"]["source"])
    teacher_identity = (sha256(source / "diffusion_pytorch_model.safetensors") if source.is_dir()
                        else config["model"]["source"] + "@" + config["model"]["revision"])
    total = tr["updates_per_stage"]
    batch_size = config["data"]["batch_size"]
    size = image_size(config)
    fake_updates = tr.get("fake_updates_per_step", 1)
    low = max(1, int(tr.get("vsd_t_min", 0.02) * len(alphas)))
    high = min(len(alphas), int(tr.get("vsd_t_max", 0.98) * len(alphas)) + 1)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with tqdm(range(first_step, total), total=total, initial=first_step,
              desc="SwiftBrush-inspired VSD | 1 NFE", unit="update", dynamic_ncols=True,
              disable=not config.get("progress_bar", True)) as progress:
        for step in progress:
            warmup = min(1.0, (step+1) / max(1, tr["warmup_steps"]))
            fraction = max(0, step-tr["warmup_steps"]) / max(1, total-tr["warmup_steps"])
            decay = warmup * 0.5 * (1 + math.cos(math.pi * fraction))
            optimizer.param_groups[0]["lr"] = tr["learning_rate"] * decay
            fake_optimizer.param_groups[0]["lr"] = tr.get("fake_learning_rate", 0.0001) * decay
            optimizer.zero_grad(set_to_none=True)
            generated, loss_value = [], 0.0
            for _ in range(tr["grad_accumulation"]):
                z = torch.randn((batch_size, 3, size, size), device=device)
                t_gen = grid.expand(batch_size)
                with autocast(config, device):
                    prediction = student(z, t_gen).sample
                images = predict_x0(z, prediction, t_gen, alphas)
                generated.append(images.detach())
                noise = torch.randn_like(images)
                t = torch.randint(low, high, (batch_size,), device=device)
                a, sigma = coefficients(alphas, t)
                noisy = a * images.detach() + sigma * noise
                with torch.no_grad(), autocast(config, device):
                    real_eps = teacher(noisy, t).sample
                    fake_eps = fake(noisy, t).sample
                loss = vsd_surrogate(images, real_eps, fake_eps, sigma)
                (loss / tr["grad_accumulation"]).backward()
                loss_value += loss.detach().item() / tr["grad_accumulation"]
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), tr["max_grad_norm"],
                                                      error_if_nonfinite=True)
            optimizer.step()
            fake_loss_value = 0.0
            for _ in range(fake_updates):
                fake_optimizer.zero_grad(set_to_none=True)
                for images in generated:
                    noise = torch.randn_like(images)
                    t = torch.randint(len(alphas), (batch_size,), device=device)
                    a, sigma = coefficients(alphas, t)
                    with autocast(config, device):
                        prediction = fake(a * images + sigma * noise, t).sample
                    fake_loss = (a * (prediction.float() - noise)).square().mean()
                    if not torch.isfinite(fake_loss):
                        raise FloatingPointError("Nonfinite auxiliary scorer loss")
                    (fake_loss / tr["grad_accumulation"]).backward()
                    fake_loss_value += fake_loss.detach().item() / (tr["grad_accumulation"] * fake_updates)
                torch.nn.utils.clip_grad_norm_(fake.parameters(), tr["max_grad_norm"], error_if_nonfinite=True)
                fake_optimizer.step()
            update_ema(ema, student, tr["ema_decay"])
            global_step = step+1
            elapsed = previous_elapsed + time.perf_counter() - started
            record = {"global_step": global_step, "stage_step": global_step, "stage_steps": 1,
                      "loss": loss_value, "vsd_surrogate": loss_value, "fake_denoising_loss": fake_loss_value,
                      "auxiliary_updates": global_step * fake_updates, "grad_norm": float(grad_norm),
                      "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": elapsed}
            if global_step % tr["log_every"] == 0 or step == 0 or global_step == total:
                with (output / "train.jsonl").open("a") as f:
                    f.write(json.dumps(record, allow_nan=False) + "\n")
                tracker.log(record)
                if not config.get("progress_bar", True):
                    print(json.dumps(record), flush=True)
            progress.set_postfix(vsd=f"{loss_value:.4g}", fake=f"{fake_loss_value:.4g}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2g}", refresh=False)
            if global_step % tr["save_every"] == 0 or global_step == total:
                save_checkpoint(checkpoint, student, ema, optimizer, config, 0, global_step, global_step,
                                elapsed, extra={"fake": fake.state_dict(),
                                                "fake_optimizer": fake_optimizer.state_dict()})
            if global_step % tr["preview_every"] == 0 or global_step == total:
                images = generate(ema, alphas, grid, 16, config["eval"]["seed"], config, device)
                save_image((images.cpu().clamp(-1, 1)+1)/2, output / "preview_1.png", nrow=4)
                tracker.image(output / "preview_1.png", global_step, 1)
    elapsed = previous_elapsed + time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0
    metadata = {"method": config["name"], "seed": config["seed"], "global_step": total,
                "initial_steps": tr["initial_steps"], "stage_steps": 1, "fake_data": config["data"]["fake"],
                "training_seconds": elapsed, "peak_training_memory_mb": peak,
                "training_hardware": environment["hardware"],
                "teacher_identity": teacher_identity, "effective_batch_size": batch_size*tr["grad_accumulation"],
                "auxiliary_updates": total*fake_updates, "training_uses_real_images": False,
                "adaptation": "SwiftBrush-inspired unconditional VSD; full auxiliary U-Net; no CFG/LoRA"}
    export_model(ema, scheduler, output / "stage_1", grid, metadata)
    atomic_json(output / "run.json", {"status": "complete", **metadata, "config": config,
                                     "environment": environment})
    tracker.summarize({"global_step": total, "auxiliary_updates": total*fake_updates,
                       "training_seconds": elapsed, "peak_training_memory_mb": peak})
    tqdm.write(f"VSD training complete: {output}")
