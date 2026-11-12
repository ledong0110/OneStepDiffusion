"""One-step SD1.5 DMD2 / JS f-distill, with explicit student-update accounting.

Core objectives and 9:1 auxiliary/student schedule follow the pinned FastGen
SD1.5 recipes. Local differences: rank-32 student, COCO's two training pools,
batch 16 via accumulation, discrete VP timesteps, BF16 compute, adapter EMA.
The fake scorer is fully trainable; the discriminator sees real training images.
"""

from copy import deepcopy
import json
import os
import subprocess
import sys
import time

import torch
from tqdm.auto import tqdm

from .diffusion import coefficients, make_grid, nested_grid, predict_x0
from .recent_objectives import (DensityRatioWeights, SD15Discriminator, SOURCE_COMMIT,
                                SOURCE_URL, discriminator_loss, distribution_surrogate,
                                generator_adversarial_loss, phase_for_update)
from .runtime import atomic_json, autocast, sha256
from .sd_data import CaptionStream, caption_records, manifest_hash, pixels
from .sd_models import (SDBundle, adapter_state, add_lora, epsilon, export_adapter,
                        load_adapter, new_unet, update_adapter_ema)
from .sd_training import preview, restore_rng
from .recent_performance import (RealEncoder, configure_models, configure_optimizers,
                                 execution_metadata, grouped_microbatch, micro_factor, options)


class _BottleneckReady(Exception):
    def __init__(self, value):
        self.value = value


def teacher_features(model, noisy, t, conditions, config, device):
    """Return the frozen conditional mid-block feature, preserving input gradients.

    Stop before the unused up path, equivalent to upstream return_features_early.
    The temporary hook is always removed, including on numerical exceptions.
    """
    def capture(_module, _args, output):
        raise _BottleneckReady(output)
    hook = model.mid_block.register_forward_hook(capture)
    try:
        with autocast(config, device):
            model(noisy, t, encoder_hidden_states=conditions)
    except _BottleneckReady as ready:
        return ready.value
    finally:
        hook.remove()
    raise RuntimeError("SD1.5 teacher did not produce its bottleneck feature")


def make_streams(config):
    first, second = deepcopy(config), deepcopy(config)
    first["data"].update(train_offset=0, train_limit=50000, sampling="replacement")
    second["data"].update(train_offset=50000, train_limit=68287, sampling="shuffled_epochs")
    # Read/download only the real training images used by auxiliary batches.
    return [CaptionStream(first, images=False), CaptionStream(second, images=False)]


def microbatch(stream, index, device, real=False):
    rows = [stream.records[i] for i in stream.indices(index).tolist()]
    images = None
    if real:
        cache_path = os.environ.get("FSD_RECENT_VAE_CACHE")
        if cache_path:
            from .vae_cache import PosteriorCache
            if not hasattr(stream, "posterior_cache"):
                stream.posterior_cache = PosteriorCache(cache_path, stream.config)
            images = stream.posterior_cache.batch(rows)
        else:
            images = torch.stack([pixels(stream.config, row) for row in rows]).to(device).float()/127.5-1
    return images, [row["caption"] for row in rows]


def noised(x, t, noise, alphas):
    alpha, sigma = coefficients(alphas, t)
    return alpha*x+sigma*noise


def sample_t(batch, device):
    return torch.randint(1, 1000, (batch,), device=device)


def save_recent(path, config, student, ema, fake, disc, ratios, optimizers, step, elapsed):
    state = {"config": config, "student": {k: v.cpu() for k, v in adapter_state(student).items()},
             "ema": {k: v.cpu() for k, v in ema.items()},
             "fake": {k: v.detach().cpu() for k, v in fake.state_dict().items()},
             "discriminator": {k: v.detach().cpu() for k, v in disc.state_dict().items()},
             "density_ratio": ratios.state_dict(),
             "optimizers": [opt.state_dict() for opt in optimizers],
             "global_step": step, "stage_step": step, "stage_index": 0,
             "auxiliary_updates": step*(config["train"]["student_update_frequency"]-1),
             "elapsed_seconds": elapsed, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    state["recent_execution"] = execution_metadata(config)
    from .checkpoint_storage import save_destination
    path = save_destination(path)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def checkpoint_evaluation(config, export, output, step):
    """Portable qualitative evaluation at every saved checkpoint; final scores use 5k."""
    target = output/"checkpoint_evaluations"/f"step_{step:06d}"
    result = target/f"{config['name']}_seed{config['seed']}_1steps.result.json"
    if result.exists():
        existing = json.loads(result.read_text())
        if existing.get("model_identity") != sha256(export/"adapter.safetensors"):
            raise ValueError("Saved checkpoint preview belongs to a different model")
        return
    target.mkdir(parents=True, exist_ok=True)
    count = config["train"]["checkpoint_eval_samples"]
    args = [sys.executable, "-m", "few_step_diffusion.cli", "evaluate", "--config", str(output/"config.yaml"),
            "--checkpoint", str(export), "--steps", "1", "--result-dir", str(target), "--skip-metrics",
            "--set", f"eval.num_samples={count}", "--set", "eval.num_real=8",
            "--set", "eval.batch_size=1", "--set", "eval.benchmark_batch_size=1",
            "--set", "eval.benchmark_repeats=1"]
    with (target/"evaluation.log").open("a") as log:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, FSD_MAX_GPU_MEMORY_GIB="8"))


def train_recent(config, state, device, output, checkpoint, env, tracker, *, stop_after=None):
    tr, recipe = config["train"], config["recent_baseline"]
    logical_accum = tr["grad_accumulation"]
    factor = micro_factor(logical_accum)
    total, accum = tr["updates_per_stage"], logical_accum//factor
    enabled = options()
    execution = execution_metadata(config)
    if state and state.get("recent_execution"):
        saved_execution = state["recent_execution"]
        if any(saved_execution[k] != execution[k] for k in ("optimizations", "micro_factor")):
            raise ValueError("Resume requires the recorded execution settings; use the queue worker")
        if saved_execution.get("posterior_cache_sha256") and saved_execution["posterior_cache_sha256"] != execution.get("posterior_cache_sha256"):
            raise ValueError("Resume requires the same frozen posterior cache")
    aux_per_student = tr["student_update_frequency"]-1
    streams = make_streams(config)
    bundle = SDBundle(config, device)
    teacher = bundle.unet.eval().requires_grad_(False)
    student = add_lora(new_unet(config, device), config)
    # Full fake scorer with FP32 optimizer/master parameters, BF16 autocast compute.
    fake_config = deepcopy(config)
    fake_config["precision"] = "fp32"
    fake = new_unet(fake_config, device).train().requires_grad_(True)
    if config["model"].get("gradient_checkpointing", True):
        fake.enable_gradient_checkpointing()
    configure_models(student, fake, enabled)
    encoder = RealEncoder(bundle, enabled)
    disc = SD15Discriminator(teacher.config.block_out_channels[-1]).to(device)
    ratios = DensityRatioWeights().to(device)
    params = [p for p in student.parameters() if p.requires_grad]
    optimizers = [torch.optim.AdamW(p, lr=tr[lr], weight_decay=tr["weight_decay"], betas=(.9, .999))
                  for p, lr in [(params, "learning_rate"), (fake.parameters(), "fake_learning_rate"),
                                (disc.parameters(), "discriminator_learning_rate")]]
    opt, fake_opt, disc_opt = optimizers
    ema = adapter_state(student)
    first = state["global_step"] if state else 0
    end = min(total, first+stop_after) if stop_after is not None else total
    elapsed_before = state.get("elapsed_seconds", 0.) if state else 0.
    if state:
        if state["auxiliary_updates"] != first*aux_per_student:
            raise ValueError("Checkpoint auxiliary/student update accounting is inconsistent")
        load_adapter(student, state["student"])
        ema = {k: v.to(device) for k, v in state["ema"].items()}
        fake.load_state_dict(state["fake"])
        disc.load_state_dict(state["discriminator"])
        ratios.load_state_dict(state["density_ratio"])
        for optimizer, saved in zip(optimizers, state["optimizers"], strict=True):
            optimizer.load_state_dict(saved)
        restore_rng(state, device)
        state = None
    configure_optimizers(optimizers, enabled, device)
    env["recent_execution"] = execution
    atomic_json(output/"environment.json", env)
    empty = bundle.embeddings([""])
    grid = nested_grid(make_grid(len(bundle.alphas), tr["initial_steps"], device), 1)
    started = time.perf_counter()
    all_records = streams[0].records+streams[1].records
    if manifest_hash(all_records) != manifest_hash(caption_records(config)):
        raise ValueError("Two-phase training pools do not cover the declared COCO training data")
    phases = [{"count": len(stream.records), "manifest_hash": manifest_hash(stream.records),
               "updates": recipe["phase1_updates"] if i == 0 else total-recipe["phase1_updates"],
               "sampling": stream.config["data"]["sampling"]} for i, stream in enumerate(streams)]

    def generate(z, conditional):
        return predict_x0(z, epsilon(student, z, grid[0], conditional, None, 1, config, device),
                          grid[0], bundle.alphas)

    def predict(model, x, t, conditional, guidance=1):
        return predict_x0(x, epsilon(model, x, t, conditional, empty.expand(len(x), -1, -1),
                                    guidance, config, device), t, bundle.alphas)

    def classify(x, t, conditional):
        feat = teacher_features(teacher, x, t, conditional, config, device)
        with autocast(config, device):
            return disc(feat).float()

    def export(step, elapsed, path):
        metadata = {"method": config["name"], "seed": config["seed"], "global_step": step,
            "cumulative_training_updates": step, "initial_steps": tr["initial_steps"], "stage_steps": 1,
            "training_seconds": elapsed, "training_hardware": env["hardware"],
            "teacher_identity": config["model"]["source"]+"@"+config["model"]["revision"],
            "training_prompt_hash": manifest_hash(all_records), "training_prompt_count": len(all_records),
            "training_data_phases": phases, "effective_batch_size": execution["effective_batch_size"],
            "trainable_parameters": sum(p.numel() for p in params), "training_uses_real_images": True,
            "auxiliary_updates": step*aux_per_student, "discriminator_updates": step*aux_per_student,
            "source_commit": SOURCE_COMMIT, "source_url": SOURCE_URL, "recent_baseline": recipe,
            "student_update_frequency": tr["student_update_frequency"], "gan_weight": tr["gan_weight"],
            "recent_execution": execution,
            "f_divergence": "js" if tr["mode"] == "f_distill" else "reverse-kl",
            "auxiliary_parameterization": "full-unet", "auxiliary_trainable_parameters": sum(p.numel() for p in fake.parameters()),
            "student_prediction_type": "epsilon", "student_parameterization": "lora", "edge_weight": 0.,
            "edge_schedule": tr["edge_schedule"], "edge_space": "none", "fake_data": False,
            "peak_training_memory_mb": torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0.,
            "adaptation": "FastGen SD1.5 DMD2/JS f-distill objectives and bottleneck discriminator; rank-32 student, COCO two-phase data, discrete VP grid, BF16 compute, adapter EMA; full fake scorer; batch-global ratio normalization"}
        export_adapter(ema, path, grid, config, metadata)
        return metadata

    with tqdm(range(first, end), total=total, initial=first, desc=f"{config['name']} | 1 NFE",
              unit="student update", dynamic_ncols=True, disable=not config.get("progress_bar", True)) as bar:
        for step in bar:
            phase, local = phase_for_update(config, step)
            stream = streams[phase]
            student.eval()
            fake.requires_grad_(True)
            fake.train()
            disc.requires_grad_(True)
            disc.train()
            fake_value = disc_value = 0.
            for auxiliary in range(aux_per_student):
                fake_opt.zero_grad(set_to_none=True)
                disc_opt.zero_grad(set_to_none=True)
                for micro in range(accum):
                    real, prompts = grouped_microbatch(microbatch, stream,
                        (local*aux_per_student+auxiliary)*logical_accum+micro*factor,
                        encoder.device, True, factor)
                    posterior = encoder.start(real)
                    conditional = bundle.embeddings(prompts)
                    with torch.no_grad():
                        z = torch.randn((len(prompts), 4, 64, 64), device=device)
                        generated = generate(z, conditional)
                        t = sample_t(len(z), device)
                        noisy = noised(generated, t, torch.randn_like(generated), bundle.alphas)
                        real_latents = encoder.finish(real, posterior)
                        real_t = sample_t(len(z), device)
                        noisy_real = noised(real_latents, real_t, torch.randn_like(real_latents), bundle.alphas)
                        ff = teacher_features(teacher, noisy, t, conditional, config, device)
                        rf = teacher_features(teacher, noisy_real, real_t, conditional, config, device)
                    fake_loss = (predict(fake, noisy, t, conditional)-generated).square().mean()
                    with autocast(config, device):
                        disc_loss = discriminator_loss(disc(rf), disc(ff))
                    loss = fake_loss+disc_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite auxiliary denoising/adversarial objective")
                    (loss/accum).backward()
                    fake_value += fake_loss.detach().item()/(accum*aux_per_student)
                    disc_value += disc_loss.detach().item()/(accum*aux_per_student)
                for module, optimizer in [(fake, fake_opt), (disc, disc_opt)]:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), tr["max_grad_norm"], error_if_nonfinite=True)
                    optimizer.step()
                fake_opt.zero_grad(set_to_none=True)
                disc_opt.zero_grad(set_to_none=True)
            fake.requires_grad_(False)
            fake.eval()
            disc.requires_grad_(False)
            disc.eval()
            prepared, logits = [], []
            # A single batch-global prepass is required for f-distill under accumulation.
            with torch.no_grad():
                for micro in range(accum):
                    _, prompts = grouped_microbatch(microbatch, stream,
                        local*logical_accum+micro*factor, device, False, factor)
                    conditional = bundle.embeddings(prompts)
                    z = torch.randn((len(prompts), 4, 64, 64), device=device)
                    generated = generate(z, conditional)
                    t = sample_t(len(z), device)
                    noise = torch.randn_like(generated)
                    noisy = noised(generated, t, noise, bundle.alphas)
                    real_x0 = predict(teacher, noisy, t, conditional, config["model"]["guidance_scale"])
                    fake_x0 = predict(fake, noisy, t, conditional)
                    prepared.append((z, conditional, t, noise, real_x0, fake_x0))
                    if tr["mode"] == "f_distill":
                        logits.append(classify(noisy, t, conditional))
            weights = (ratios(torch.cat(logits), torch.cat([b[2] for b in prepared])) if logits else
                       torch.ones(sum(len(b[0]) for b in prepared), device=device))
            opt.zero_grad(set_to_none=True)
            student.train()
            dm_value = gan_value = 0.
            offset = 0
            for z, conditional, t, noise, real_x0, fake_x0 in prepared:
                generated = generate(z, conditional)
                noisy = noised(generated, t, noise, bundle.alphas)
                loss_dm = distribution_surrogate(generated, real_x0, fake_x0, weights[offset:offset+len(z)])
                loss_gan = generator_adversarial_loss(classify(noisy, t, conditional))
                loss = loss_dm+tr["gan_weight"]*loss_gan
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite student distribution/adversarial objective")
                (loss/accum).backward()
                offset += len(z)
                dm_value += loss_dm.detach().item()/accum
                gan_value += loss_gan.detach().item()/accum
            norm = torch.nn.utils.clip_grad_norm_(params, tr["max_grad_norm"], error_if_nonfinite=True)
            opt.step()
            update_adapter_ema(ema, student, tr["ema_decay"])
            global_step = step+1
            elapsed = elapsed_before+time.perf_counter()-started
            metrics = {"global_step": global_step, "stage_step": global_step, "stage_steps": 1,
                "loss": dm_value+tr["gan_weight"]*gan_value, "distribution_loss": dm_value,
                "generator_gan_loss": gan_value, "fake_loss": fake_value, "discriminator_loss": disc_value,
                "auxiliary_updates": global_step*aux_per_student, "training_phase": phase+1,
                "learning_rate": tr["learning_rate"], "grad_norm": float(norm), "elapsed_seconds": elapsed,
                "weight_min": weights.min().item(), "weight_max": weights.max().item()}
            if global_step % tr["log_every"] == 0 or global_step in (1, total):
                with (output/"train.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics)+"\n")
                tracker.log(metrics)
            bar.set_postfix(loss=f"{metrics['loss']:.4g}", phase=phase+1, auxiliary=global_step*aux_per_student)
            if global_step % tr["save_every"] == 0 or global_step == end:
                save_recent(checkpoint, config, student, ema, fake, disc, ratios, optimizers, global_step, elapsed)
                saved = output/"checkpoints"/f"step_{global_step:06d}"
                export(global_step, elapsed, saved)
                with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                    preview(bundle, student, ema, grid, output, tracker, global_step)
                checkpoint_evaluation(config, saved, output, global_step)
            del prepared, logits
    elapsed = elapsed_before+time.perf_counter()-started
    if end < total:
        atomic_json(output/"preflight.json", {"status": "passed", "global_step": end,
                    "auxiliary_updates": end*aux_per_student,
                    "peak_memory_mb": torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0})
        return
    export(total, elapsed, output/"stage_1")
    atomic_json(output/"run.json", {"status": "complete", "method": config["name"], "seed": config["seed"],
        "global_step": total, "cumulative_training_updates": total, "auxiliary_updates": total*aux_per_student,
        "training_seconds": elapsed, "training_data_phases": phases, "config": config, "environment": env})
    tracker.summarize({"global_step": total, "auxiliary_updates": total*aux_per_student, "training_seconds": elapsed})
