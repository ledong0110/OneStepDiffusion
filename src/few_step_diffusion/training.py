"""Single-GPU teacher training and two-to-one progressive DDIM distillation."""

import copy
import json
import math
import time
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm
from torchvision.utils import save_image

from .data import BatchStream
from .config import experiment_config
from .diffusion import (coefficients, collapsed_x0_target, ddim_step, distillation_loss,
                        make_grid, predict_x0)
from .models import load_model
from .runtime import atomic_json, autocast, device_for, provenance, seed_everything, sha256
from .tracking import TrainingTracker


@torch.no_grad()
def update_ema(ema, student, decay):
    for target, current in zip(ema.parameters(), student.parameters(), strict=True):
        target.lerp_(current.detach(), 1 - decay)
    for target, current in zip(ema.buffers(), student.buffers(), strict=True):
        target.copy_(current)


def export_model(model, scheduler, path, grid, metadata):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    scheduler.save_pretrained(path)
    atomic_json(path / "metadata.json", {**metadata, "timesteps": grid.cpu().tolist()})


def save_checkpoint(path, student, ema, optimizer, config, stage_index, stage_step, global_step,
                    elapsed_seconds, extra=None):
    state = {"student": student.state_dict(), "ema": ema.state_dict(),
             "optimizer": optimizer.state_dict(), "config": config,
             "stage_index": stage_index, "stage_step": stage_step, "global_step": global_step,
             "elapsed_seconds": elapsed_seconds, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    state.update(extra or {})
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(config, resume=False):
    seed_everything(config["seed"])
    device = device_for(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "last.pt"
    if (checkpoint.exists() or (output / "run.json").exists()) and not resume:
        raise ValueError(f"Run already exists: {output}. Use --resume or a new output_dir")
    state = None
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"No checkpoint to resume: {checkpoint}")
        # Only load locally produced checkpoints you trust. Model exports use safetensors.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if experiment_config(state["config"]) != experiment_config(config):
            raise ValueError("Resume config differs from saved config; restore the original settings")
    elif "continuation" in config:
        from .continuation import initialize_continuation
        state = initialize_continuation(config, output)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    env = provenance(config, device)
    atomic_json(output / "environment.json", env)
    with TrainingTracker(config, env, output, resume=resume) as tracker:
        from .config import is_stable_diffusion
        if is_stable_diffusion(config):
            if config["train"].get("vsd_recipe") == "swiftbrush-source-v1":
                from .sd_swiftbrush import train_swiftbrush_source
                train_swiftbrush_source(config, state, device, output, checkpoint, env, tracker)
            else:
                from .sd_training import train_sd
                train_sd(config, state, device, output, checkpoint, env, tracker)
        elif config["train"]["mode"] == "vsd":
            from .vsd import train_vsd
            train_vsd(config, state, device, output, checkpoint, env, tracker)
        else:
            _train_progressive(config, state, device, output, checkpoint, env, tracker)


def _train_progressive(config, state, device, output, checkpoint, env, tracker):
    data = BatchStream(config)
    teacher, scheduler = load_model(config)
    teacher = teacher.to(device).eval().requires_grad_(False)
    alphas = scheduler.alphas_cumprod.to(device)
    root = make_grid(len(alphas), config["train"]["initial_steps"], device)
    tr = config["train"]
    source = Path(config["model"]["source"])
    teacher_identity = (sha256(source / "diffusion_pytorch_model.safetensors") if source.is_dir()
                        else config["model"]["source"] + "@" + config["model"]["revision"])
    stages = tr["stages"] if tr["mode"] == "distill" else [tr["initial_steps"]]
    start_stage = state["stage_index"] if state else 0
    global_step = state["global_step"] if state else 0
    previous_elapsed = state.get("elapsed_seconds", 0.0) if state else 0.0
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for stage_index, steps in enumerate(stages):
        if stage_index < start_stage:
            continue
        if stage_index > 0:
            teacher, scheduler = load_model(config, str(output / f"stage_{stages[stage_index-1]}"))
            teacher = teacher.to(device).eval().requires_grad_(False)
        student = copy.deepcopy(teacher).requires_grad_(True).train()
        ema = copy.deepcopy(teacher).requires_grad_(False).eval()
        optimizer = torch.optim.AdamW(student.parameters(), lr=tr["learning_rate"],
                                      weight_decay=tr["weight_decay"])
        first_step = 0
        if state and stage_index == start_stage:
            student.load_state_dict(state["student"])
            ema.load_state_dict(state["ema"])
            optimizer.load_state_dict(state["optimizer"])
            first_step = state["stage_step"]
            torch.set_rng_state(state["torch_rng"])
            if device.type == "cuda" and state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            state = None
        student_grid = root[::len(root) // steps]
        teacher_grid = root[::len(root) // (2 * steps)] if tr["mode"] == "distill" else root
        teacher_endpoints = torch.cat((teacher_grid, teacher_grid.new_tensor([-1])))
        print(f"Stage {stage_index+1}/{len(stages)}: {steps} sampling steps", flush=True)
        with tqdm(range(first_step, tr["updates_per_stage"]), total=tr["updates_per_stage"],
                  initial=first_step, desc=f"Stage {stage_index+1}/{len(stages)} | {steps} NFE",
                  unit="update", dynamic_ncols=True, disable=not config.get("progress_bar", True)) as progress_bar:
            for step in progress_bar:
                warmup = min(1.0, (step + 1) / max(1, tr["warmup_steps"]))
                progress = (step - tr["warmup_steps"]) / max(1, tr["updates_per_stage"] - tr["warmup_steps"])
                decay = 0.5 * (1 + math.cos(math.pi * max(0, progress)))
                for group in optimizer.param_groups:
                    group["lr"] = tr["learning_rate"] * warmup * decay
                optimizer.zero_grad(set_to_none=True)
                loss_value, parts = 0.0, {}
                for micro in range(tr["grad_accumulation"]):
                    x0 = data.batch(global_step * tr["grad_accumulation"] + micro, device)
                    noise = torch.randn_like(x0)
                    if tr["mode"] == "teacher":
                        t = torch.randint(len(alphas), (len(x0),), device=device)
                    else:
                        interval = torch.randint(steps, (len(x0),), device=device)
                        t, mid, end = (teacher_endpoints[2*interval + offset] for offset in (0, 1, 2))
                    a, sigma = coefficients(alphas, t)
                    xt = a * x0 + sigma * noise
                    if tr["mode"] == "distill":
                        with torch.no_grad(), autocast(config, device):
                            eps = teacher(xt, t).sample
                            xm = ddim_step(xt, eps, t, mid, alphas)
                            eps_mid = teacher(xm, mid).sample
                            xs = ddim_step(xm, eps_mid, mid, end, alphas)
                            target = collapsed_x0_target(xt, xs, t, end, alphas)
                    with autocast(config, device):
                        prediction = student(xt, t).sample
                    if tr["mode"] == "teacher":
                        loss = (prediction.float() - noise).square().mean()
                        parts = {"epsilon_mse": loss.detach().item()}
                    else:
                        loss, parts = distillation_loss(
                            predict_x0(xt, prediction, t, alphas), target, t, alphas,
                            tr["min_snr_gamma"], tr["edge_weight"], tr["edge_schedule"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss; inspect the last checkpoint and precision")
                    (loss / tr["grad_accumulation"]).backward()
                    loss_value += loss.detach().item() / tr["grad_accumulation"]
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), tr["max_grad_norm"],
                                                          error_if_nonfinite=True)
                optimizer.step()
                update_ema(ema, student, tr["ema_decay"])
                global_step += 1
                elapsed = previous_elapsed + time.perf_counter() - start_time
                if (step + 1) % tr["log_every"] == 0 or step == 0 or step+1 == tr["updates_per_stage"]:
                    record = {"stage_steps": steps, "stage_step": step+1, "global_step": global_step,
                              "loss": loss_value, "grad_norm": float(grad_norm), "elapsed_seconds": elapsed,
                              "learning_rate": optimizer.param_groups[0]["lr"], **parts}
                    with (output / "train.jsonl").open("a") as f:
                        f.write(json.dumps(record, allow_nan=False) + "\n")
                    tracker.log(record)
                    if not config.get("progress_bar", True):
                        print(json.dumps(record), flush=True)
                progress_bar.set_postfix(loss=f"{loss_value:.4g}", lr=f"{optimizer.param_groups[0]['lr']:.2g}",
                                     refresh=False)
                if (step + 1) % tr["save_every"] == 0 or step + 1 == tr["updates_per_stage"]:
                    save_checkpoint(checkpoint, student, ema, optimizer, config, stage_index,
                                    step+1, global_step, elapsed)
                if (step + 1) % tr["preview_every"] == 0 or step + 1 == tr["updates_per_stage"]:
                    from .sampling import generate

                    preview = generate(ema, alphas, student_grid, 16, config["eval"]["seed"],
                                       config, device)
                    save_image((preview.cpu().clamp(-1, 1) + 1) / 2,
                               output / f"preview_{steps}.png", nrow=4)
                    tracker.image(output / f"preview_{steps}.png", global_step, steps)
        export_model(ema, scheduler, output / f"stage_{steps}", student_grid,
                     {"method": config["name"], "seed": config["seed"],
                      "global_step": global_step, "initial_steps": tr["initial_steps"],
                      "training_seconds": previous_elapsed + time.perf_counter() - start_time,
                      "training_hardware": env["hardware"],
                      "teacher_identity": teacher_identity,
                      "effective_batch_size": config["data"]["batch_size"] * tr["grad_accumulation"],
                      "peak_training_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20
                      if device.type == "cuda" else 0,
                      "stage_steps": steps, "fake_data": config["data"]["fake"]})
        del student, ema, optimizer
    elapsed = previous_elapsed + time.perf_counter() - start_time
    atomic_json(output / "run.json", {"status": "complete", "method": config["name"],
                "seed": config["seed"], "global_step": global_step, "training_seconds": elapsed,
                "peak_training_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda" else 0, "config": config, "environment": env})
    tracker.summarize({"global_step": global_step, "training_seconds": elapsed})
    print(f"Training complete: {output}", flush=True)
