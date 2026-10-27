"""SwiftBrush's source training recipe on the project's SD1.5/COCO prompts.

Audited against VinAIResearch/SwiftBrush commit a346fec (see SOURCE_COMMIT).
This is a training-recipe port, not a reproduction of its SD2.1/JourneyDB study.
The existing rank-matched, LoRA-generator VSD experiment remains unchanged.
"""

from contextlib import contextmanager
from copy import deepcopy
import json
import random
import time

from diffusers.training_utils import EMAModel
import torch
from tqdm.auto import tqdm
from torchvision.utils import save_image

from .diffusion import coefficients
from .runtime import atomic_json
from .sd_data import CaptionStream, manifest_hash
from .sd_models import (PREVIEW_PROMPTS, SDBundle, adapter_state, add_lora, epsilon,
                        export_adapter, generate_sd, load_adapter, new_unet)
from .vsd import vsd_surrogate

SOURCE_COMMIT = "a346fec56484a5de54e062b1b4251f1af1e0bce0"
SOURCE_URL = f"https://github.com/VinAIResearch/SwiftBrush/blob/{SOURCE_COMMIT}/train_swiftbrush.py"
# Upstream predict_original() explicitly uses 0.0047 rather than reading alpha_T.
TERMINAL_ALPHA_CUMPROD = .0047


def source_original(noise, prediction):
    return ((noise - (1-TERMINAL_ALPHA_CUMPROD)**.5 * prediction)
            / TERMINAL_ALPHA_CUMPROD**.5)


def source_scorer_loss(prediction, noise, alpha):
    # Scale both operands before the FP32 MSE, as in the reference trainer.
    return torch.nn.functional.mse_loss((alpha*prediction).float(), (alpha*noise).float())


@contextmanager
def using_source_ema(student, ema):
    training = student.training
    parameters = [p for p in student.parameters() if p.requires_grad]
    ema.store(parameters)
    ema.copy_to(parameters)
    student.eval()
    try:
        yield
    finally:
        ema.restore(parameters)
        student.train(training)


def save_source_state(path, config, student, ema, optimizer, fake, fake_optimizer,
                      global_step, elapsed):
    lora_student = config["train"].get("student_parameterization") == "lora"
    state = {"config": config, "student": adapter_state(student) if lora_student else student.state_dict(), "ema": ema.state_dict(),
             "optimizer": optimizer.state_dict(), "fake": adapter_state(fake),
             "fake_optimizer": fake_optimizer.state_dict(), "global_step": global_step,
             "stage_index": 0, "stage_step": global_step, "auxiliary_updates": global_step,
             "elapsed_seconds": elapsed, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "python_rng": random.getstate(), "source_commit": SOURCE_COMMIT}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


@torch.no_grad()
def preview_source(bundle, student, ema, grid, output, tracker, global_step):
    noise = torch.randn((8, 4, 64, 64),
        generator=torch.Generator().manual_seed(bundle.config["eval"]["seed"]))
    images = []
    with using_source_ema(student, ema):
        for start in range(0, 8, 2):
            images.append(generate_sd(bundle, student, grid, PREVIEW_PROMPTS[start:start+2],
                bundle.config["eval"]["seed"], noise=noise[start:start+2].to(bundle.device)).cpu())
    path = output/"preview_1.png"
    save_image((torch.cat(images).clamp(-1, 1)+1)/2, path, nrow=4)
    tracker.image(path, global_step, 1)


def train_swiftbrush_source(config, state, device, output, checkpoint, env, tracker):
    tr = config["train"]
    lora_student = tr.get("student_parameterization") == "lora"
    data = CaptionStream(config, images=False)
    bundle = SDBundle(config, device)
    # Disabling F's adapters supplies the frozen real teacher, exactly as upstream.
    fake_config = deepcopy(config)
    if lora_student:
        fake_config["model"].update(lora_rank=tr["fake_lora_rank"], lora_alpha=tr["fake_lora_alpha"],
                                    lora_targets=tr["fake_lora_targets"], lora_init=True)
    fake = add_lora(bundle.unet, fake_config)
    student = new_unet(config, device).float()
    if lora_student:
        student = add_lora(student, config)
    else:
        student.train().requires_grad_(True)
    student._fsd_prediction_type = "swiftbrush-source-x0"
    if config["model"].get("gradient_checkpointing", True):
        student.enable_gradient_checkpointing()
    parameters = [p for p in student.parameters() if p.requires_grad]
    fake_parameters = [p for p in fake.parameters() if p.requires_grad]
    ema = EMAModel(parameters, decay=tr["ema_decay"])
    optimizer = torch.optim.Adam(parameters, lr=tr["learning_rate"],
                                 weight_decay=tr["weight_decay"])
    fake_optimizer = torch.optim.Adam(fake_parameters, lr=tr["fake_learning_rate"],
                                      weight_decay=tr["weight_decay"])
    first, previous_elapsed = 0, 0.0
    if state is not None:
        if state.get("source_commit") != SOURCE_COMMIT:
            raise ValueError("Checkpoint does not belong to this SwiftBrush source recipe")
        if lora_student:
            load_adapter(student, state["student"])
        else:
            student.load_state_dict(state["student"])
        load_adapter(fake, state["fake"])
        ema.load_state_dict(state["ema"])
        ema.to(device=device)
        optimizer.load_state_dict(state["optimizer"])
        fake_optimizer.load_state_dict(state["fake_optimizer"])
        first, previous_elapsed = state["global_step"], state["elapsed_seconds"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    empty = bundle.embeddings([""])
    grid = torch.tensor([len(bundle.alphas)-1], device=device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with tqdm(range(first, tr["updates_per_stage"]), initial=first,
              total=tr["updates_per_stage"], desc=f"SwiftBrush recipe | {'LoRA' if lora_student else 'full'} G | 1 NFE",
              unit="update", dynamic_ncols=True, disable=not config.get("progress_bar", True)) as bar:
        for step in bar:
            optimizer.zero_grad(set_to_none=True)
            fake_optimizer.zero_grad(set_to_none=True)
            loss_value, fake_loss_value = 0., 0.
            # The README uses batch=16/accumulation=1 and one dropout coin per batch.
            # Share that coin over our microbatches to preserve its batch semantics.
            drop_text = random.random() < .1
            for micro in range(tr["grad_accumulation"]):
                _, prompts = data.batch(step*tr["grad_accumulation"]+micro, device)
                conditional = bundle.embeddings(prompts)
                unconditional = empty.expand(len(prompts), -1, -1)
                z = torch.randn((len(prompts), 4, 64, 64), device=device)
                terminal = grid.expand(len(z))
                generated = source_original(z, epsilon(student, z, terminal,
                    conditional, None, 1, config, device))
                t = torch.randint(20, 981, (len(z),), device=device)
                alpha, sigma = coefficients(bundle.alphas, t)
                with torch.no_grad():
                    noisy = alpha*generated.detach()+sigma*torch.randn_like(generated)
                    fake.disable_adapters()
                    try:
                        real_eps = epsilon(fake, noisy, t, conditional, unconditional,
                                           config["model"]["guidance_scale"], config, device)
                    finally:
                        fake.enable_adapters()
                    fake_eps = epsilon(fake, noisy, t, conditional, unconditional,
                                       config["model"]["guidance_scale"], config, device)
                loss = vsd_surrogate(generated, real_eps, fake_eps, sigma)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite source-recipe generator loss")
                (loss/tr["grad_accumulation"]).backward()
                loss_value += loss.detach().item()/tr["grad_accumulation"]
                if micro+1 == tr["grad_accumulation"]:
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, tr["max_grad_norm"],
                                                              error_if_nonfinite=True)
                    optimizer.step()
                # F learns from this SAME detached, pre-G-update output. No fresh
                # student forward, replacement prompts, or post-update generation.
                with torch.no_grad():
                    noise = torch.randn_like(generated)
                    t = torch.randint(len(bundle.alphas), (len(z),), device=device)
                    alpha, sigma = coefficients(bundle.alphas, t)
                    noisy = alpha*generated.detach()+sigma*noise
                    conditions = unconditional if drop_text else conditional
                prediction = epsilon(fake, noisy, t, conditions, None, 1, config, device)
                fake_loss = source_scorer_loss(prediction, noise, alpha)
                if not torch.isfinite(fake_loss):
                    raise FloatingPointError("Nonfinite source-recipe auxiliary loss")
                (fake_loss/tr["grad_accumulation"]).backward()
                fake_loss_value += fake_loss.detach().item()/tr["grad_accumulation"]
            fake_grad_norm = torch.nn.utils.clip_grad_norm_(fake_parameters, tr["max_grad_norm"],
                                                           error_if_nonfinite=True)
            fake_optimizer.step()
            ema.step(parameters)
            global_step = step+1
            elapsed = previous_elapsed+time.perf_counter()-started
            if global_step % tr["log_every"] == 0 or step == 0 or global_step == tr["updates_per_stage"]:
                record = {"stage_steps": 1, "stage_step": global_step, "global_step": global_step,
                          "loss": loss_value, "vsd_loss": loss_value, "fake_loss": fake_loss_value,
                          "grad_norm": float(grad_norm), "fake_grad_norm": float(fake_grad_norm),
                          "learning_rate": tr["learning_rate"], "fake_learning_rate": tr["fake_learning_rate"],
                          "ema_decay": ema.cur_decay_value, "auxiliary_updates": global_step,
                          "elapsed_seconds": elapsed}
                with (output/"train.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False)+"\n")
                tracker.log(record)
                if not config.get("progress_bar", True):
                    print(json.dumps(record), flush=True)
            bar.set_postfix(vsd=f"{loss_value:.4g}", scorer=f"{fake_loss_value:.4g}", refresh=False)
            if global_step % tr["save_every"] == 0 or global_step == tr["updates_per_stage"]:
                save_source_state(checkpoint, config, student, ema, optimizer, fake, fake_optimizer,
                                  global_step, elapsed)
            if global_step % tr["preview_every"] == 0 or global_step == tr["updates_per_stage"]:
                preview_source(bundle, student, ema, grid, output, tracker, global_step)
    elapsed = previous_elapsed+time.perf_counter()-started
    metadata = {"method": config["name"], "seed": config["seed"],
                "global_step": tr["updates_per_stage"], "auxiliary_updates": tr["updates_per_stage"],
                "initial_steps": tr["initial_steps"], "timesteps": grid.cpu().tolist(),
                "training_seconds": elapsed, "training_hardware": env["hardware"],
                "effective_batch_size": config["data"]["batch_size"]*tr["grad_accumulation"],
                "trainable_parameters": sum(p.numel() for p in parameters),
                "auxiliary_trainable_parameters": sum(p.numel() for p in fake_parameters),
                "training_uses_real_images": False, "training_prompt_hash": manifest_hash(data.records),
                "training_prompt_count": len(data.records), "student_parameterization": "lora" if lora_student else "full-unet",
                "student_prediction_type": "swiftbrush-source-x0",
                "source_commit": SOURCE_COMMIT, "source_url": SOURCE_URL,
                "terminal_alpha_cumprod": TERMINAL_ALPHA_CUMPROD,
                "auxiliary_lora_rank": fake_config["model"]["lora_rank"],
                "auxiliary_lora_alpha": fake_config["model"]["lora_alpha"],
                "auxiliary_lora_targets": fake_config["model"]["lora_targets"],
                "lora_rank": config["model"]["lora_rank"] if lora_student else 0,
                "model_family": "stable-diffusion", "model_source": config["model"]["source"],
                "model_revision": config["model"]["revision"], "image_size": 512,
                "teacher_identity": config["model"]["source"]+"@"+config["model"]["revision"],
                "teacher_guidance_scale": config["model"]["guidance_scale"], "student_guidance_scale": 1.,
                "peak_training_memory_mb": torch.cuda.max_memory_allocated(device)/2**20
                    if device.type == "cuda" else 0,
                "adaptation": "SwiftBrush training recipe: " + ("LoRA G" if lora_student else "full U-Net G") + ", q/v LoRA F, Adam, constant LR, "
                              "dynamic EMA, same detached samples; SD1.5/COCO/budget and microbatching adaptation"}
    destination = output/"stage_1"
    from .continuation import continuation_metadata
    metadata.update(continuation_metadata(config, output, tr["updates_per_stage"],
                                         elapsed, tr["updates_per_stage"]))
    with using_source_ema(student, ema):
        if lora_student:
            export_adapter(adapter_state(student), destination, grid, config, metadata)
        else:
            student.save_pretrained(destination, safe_serialization=True)
    atomic_json(destination/"metadata.json", metadata)
    atomic_json(output/"run.json", {"status": "complete", "method": config["name"],
        "seed": config["seed"], "global_step": tr["updates_per_stage"], "config": config,
        "training_seconds": elapsed, "auxiliary_updates": tr["updates_per_stage"], "environment": env,
        **continuation_metadata(config, output, tr["updates_per_stage"], elapsed, tr["updates_per_stage"])})
    tracker.summarize({"global_step": tr["updates_per_stage"], "training_seconds": elapsed})
    print(f"Training complete: {output}", flush=True)
