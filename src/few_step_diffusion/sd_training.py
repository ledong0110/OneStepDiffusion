"""Caption-conditioned LoRA progressive distillation and SwiftBrush-inspired VSD."""

import json
import math
import time

from safetensors.torch import load_file
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torchvision.utils import save_image
from tqdm.auto import tqdm

from .diffusion import (coefficients, collapsed_x0_target, ddim_step, distillation_loss,
                        edge_error, make_grid, nested_grid, predict_x0)
from .distillation_objectives import dmd_surrogate, rectified_flow_batch
from .runtime import atomic_json, experiment_lock
from .sd_data import CaptionStream, manifest_hash
from .sd_models import (PREVIEW_PROMPTS, SDBundle, adapter_state, add_lora, epsilon,
                        consistency_prediction, set_consistency_prediction,
                        export_adapter, generate_sd, load_adapter, new_unet,
                        update_adapter_ema, using_adapter)
from .sd_models import flow_prediction
from .vsd import vsd_surrogate


def sd_distillation_loss(bundle, prediction, target, t):
    tr = bundle.config["train"]
    loss, parts = distillation_loss(prediction, target, t, bundle.alphas,
                                    tr["min_snr_gamma"], 0, tr["edge_schedule"])
    if tr["edge_weight"]:
        # Target has no gradient. Decoder checkpointing limits RGB-edge memory cost.
        with torch.no_grad():
            target_rgb = bundle.decode(target)
        predicted_rgb = activation_checkpoint(bundle.decode, prediction, use_reentrant=False)
        edge = edge_error(predicted_rgb, target_rgb)
        alpha, sigma = coefficients(bundle.alphas, t)
        weights = (alpha/sigma.clamp_min(1e-8)).square().flatten().clamp(max=tr["min_snr_gamma"])
        gate = alpha.square().flatten() if tr["edge_schedule"] == "noise_aware" else torch.ones_like(weights)
        loss = loss + tr["edge_weight"] * (weights*gate*edge).mean()
        parts["rgb_edge_mse"] = edge.detach().mean().item()
    return loss, parts


def save_state(path, config, student, ema, optimizer, stage, step, global_step, elapsed,
               fake=None, fake_optimizer=None, auxiliary_updates=0):
    state = {"config": config, "student": {k: v.cpu() for k, v in adapter_state(student).items()},
             "ema": {k: v.cpu() for k, v in ema.items()}, "optimizer": optimizer.state_dict(),
             "stage_index": stage, "stage_step": step, "global_step": global_step,
             "elapsed_seconds": elapsed, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             "auxiliary_updates": auxiliary_updates}
    if fake is not None:
        state["fake"] = {k: v.cpu() for k, v in adapter_state(fake).items()}
        state["fake_optimizer"] = fake_optimizer.state_dict()
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_rng(state, device):
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda" and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def preview(bundle, student, ema, grid, output, tracker, global_step):
    images = []
    noise = torch.randn((8, 4, 64, 64), generator=torch.Generator().manual_seed(bundle.config["eval"]["seed"]))
    with using_adapter(student, ema):
        for start in range(0, 8, 2):
            images.append(generate_sd(bundle, student, grid, PREVIEW_PROMPTS[start:start+2],
                bundle.config["eval"]["seed"], noise=noise[start:start+2].to(bundle.device)).cpu())
    path = output / f"preview_{len(grid)}.png"
    save_image((torch.cat(images).clamp(-1, 1)+1)/2, path, nrow=4)
    tracker.image(path, global_step, len(grid))


def train_sd(config, state, device, output, checkpoint, env, tracker):
    tr = config["train"]
    vsd = tr["mode"] in ("vsd", "dmd")
    dmd = tr["mode"] == "dmd"
    consistency = tr["mode"] == "consistency"
    instaflow = tr["mode"] == "instaflow"
    endpoint_continuation = config.get("continuation", {}).get("instaflow_endpoint_only", False)
    data = CaptionStream(config, images=not (vsd or instaflow))
    bundle = SDBundle(config, device)
    teacher = bundle.unet
    root = make_grid(len(bundle.alphas), tr["initial_steps"], device)
    stages = [1] if vsd or consistency or instaflow else tr["stages"]
    start_stage = state["stage_index"] if state else 0
    global_step = state["global_step"] if state else 0
    auxiliary_updates = state.get("auxiliary_updates", 0) if state else 0
    previous_elapsed = state.get("elapsed_seconds", 0) if state else 0
    started = time.perf_counter()
    teacher_id = config["model"]["source"] + "@" + config["model"]["revision"]
    empty = bundle.embeddings([""])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pairs = None
    perceptual_metric = None
    if dmd and (tr.get("dmd_regression_type", "latent_mse") == "lpips_vgg" or tr.get("dmd_diversity_weight", 0)):
        from .dmd_diversity import load_lpips
        perceptual_metric = load_lpips("vgg", device)
    if dmd or (instaflow and not endpoint_continuation):
        from .teacher_pairs import TeacherPairs
        if dmd and tr.get("dmd_pair_cache_dir"):
            with experiment_lock(tr["dmd_pair_cache_dir"]):
                pairs = TeacherPairs(config, bundle, root, tr["dmd_pair_cache_dir"], data.records)
        else:
            pairs = TeacherPairs(config, bundle, root, output, data.records)
    for stage_index, steps in enumerate(stages):
        if stage_index < start_stage:
            continue
        previous = None
        if stage_index > 0:
            previous = load_file(str(output / f"stage_{stages[stage_index-1]}" / "adapter.safetensors"),
                                 device=str(device))
            if stage_index == start_stage or stage_index == 1:
                add_lora(teacher, config, trainable=False, state=previous)
            else:
                load_adapter(teacher, previous)
        student = add_lora(new_unet(config, device), config, state=previous)
        if instaflow:
            student._fsd_prediction_type = tr.get("instaflow_prediction_type", "rectified-flow-velocity")
        distilled_pairs = None
        consistency_target = None
        if consistency:
            set_consistency_prediction(student, tr.get("consistency_sigma_data", .5))
            consistency_target = add_lora(new_unet(config, device), config, trainable=False,
                                          state=adapter_state(student))
        parameters = [p for p in student.parameters() if p.requires_grad]
        ema = adapter_state(student)
        optimizer = torch.optim.AdamW(parameters, lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
        fake = fake_optimizer = None
        fake_parameters = []
        if vsd:
            fake = add_lora(new_unet(config, device), config)
            fake_parameters = [p for p in fake.parameters() if p.requires_grad]
            fake_optimizer = torch.optim.AdamW(fake_parameters,
                lr=tr.get("fake_learning_rate", 0.0001), weight_decay=tr["weight_decay"])
        first_step = 0
        if state and stage_index == start_stage:
            load_adapter(student, state["student"])
            ema = {k: v.to(device) for k, v in state["ema"].items()}
            optimizer.load_state_dict(state["optimizer"])
            if vsd:
                load_adapter(fake, state["fake"])
                fake_optimizer.load_state_dict(state["fake_optimizer"])
            first_step = state["stage_step"]
            restore_rng(state, device)
            state = None
        if consistency:
            load_adapter(consistency_target, ema)
        if endpoint_continuation:
            from .teacher_pairs import ContinuationFlowPairs
            pairs = ContinuationFlowPairs(config, bundle, student, root, output, data.records)
            distilled_pairs = pairs
        student_grid = nested_grid(root, steps)
        endpoints = (torch.cat((root, root.new_tensor([-1]))) if consistency else None if vsd or instaflow else
                     torch.cat((nested_grid(root, 2*steps), root.new_tensor([-1]))))
        guidance = config["model"]["guidance_scale"] if stage_index == 0 else 1.0
        with tqdm(range(first_step, tr["updates_per_stage"]), total=tr["updates_per_stage"],
                  initial=first_step, desc=f"SD LoRA {tr['mode'].upper()} | {steps} steps",
                  unit="update", dynamic_ncols=True, disable=not config.get("progress_bar", True)) as bar:
            for step in bar:
                if instaflow and step >= tr["updates_per_stage"]//2 and distilled_pairs is None:
                    from .teacher_pairs import reflow_pairs
                    distilled_pairs = reflow_pairs(config, bundle, student, ema, root, output, pairs)
                warmup = min(1, (step+1)/max(1, tr["warmup_steps"]))
                progress = max(0, (step-tr["warmup_steps"])/max(1, tr["updates_per_stage"]-tr["warmup_steps"]))
                lr = tr["learning_rate"]*warmup*0.5*(1+math.cos(math.pi*progress))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss_value, parts, averaged_parts = 0.0, {}, {}
                for micro in range(tr["grad_accumulation"]):
                    images, prompts = data.batch(global_step*tr["grad_accumulation"]+micro, device)
                    conditional = bundle.embeddings(prompts)
                    unconditional = empty.expand(len(prompts), -1, -1)
                    if instaflow:
                        source = distilled_pairs if distilled_pairs is not None else pairs
                        pair_noise, target_latents, pair_prompts = source.batch(
                            global_step*tr["grad_accumulation"]+micro, device)
                        conditional = bundle.embeddings(pair_prompts)
                        progress = (torch.zeros(len(pair_prompts), device=device) if distilled_pairs is not None
                                    else torch.rand(len(pair_prompts), device=device))
                        noisy, velocity, t = rectified_flow_batch(pair_noise, target_latents, progress)
                        prediction = flow_prediction(bundle, student, noisy, progress, conditional)
                        loss = (prediction-velocity).square().mean()
                        parts = {"flow_velocity_mse": loss.detach().item(),
                                 "flow_phase": "distill" if distilled_pairs is not None else "reflow"}
                    elif vsd:
                        z = torch.randn((len(prompts), 4, 64, 64), device=device)
                        terminal = root[0].expand(len(z))
                        predicted = epsilon(student, z, terminal, conditional, None, 1, config, device)
                        generated = predict_x0(z, predicted, terminal, bundle.alphas)
                        t = torch.randint(max(1, int(len(bundle.alphas)*tr.get("vsd_t_min", .02))),
                            min(len(bundle.alphas), int(len(bundle.alphas)*tr.get("vsd_t_max", .98))+1),
                            (len(z),), device=device)
                        alpha, sigma = coefficients(bundle.alphas, t)
                        with torch.no_grad():
                            noisy = alpha*generated.detach()+sigma*torch.randn_like(generated)
                            real_eps = epsilon(teacher, noisy, t, conditional, unconditional,
                                               guidance, config, device)
                            fake.eval()
                            fake_eps = epsilon(fake, noisy, t, conditional, unconditional,
                                               1 if dmd else guidance, config, device)
                            fake.train()
                        if dmd:
                            score_loss = dmd_surrogate(generated, noisy, real_eps, fake_eps, t, bundle.alphas)
                            pair_noise, target_latents, pair_prompts = pairs.batch(
                                global_step*tr["grad_accumulation"]+micro, device)
                            pair_embeddings = bundle.embeddings(pair_prompts)
                            reg_eps = epsilon(student, pair_noise, terminal, pair_embeddings, None, 1, config, device)
                            reg_latents = predict_x0(pair_noise, reg_eps, terminal, bundle.alphas)
                            from .dmd_diversity import regression_and_diversity
                            regression_loss, diversity_loss, diversity_parts = regression_and_diversity(
                                bundle, perceptual_metric, reg_latents, target_latents, config)
                            loss = (score_loss+tr["dmd_regression_weight"]*regression_loss+
                                    tr.get("dmd_diversity_weight", 0)*diversity_loss)
                            parts = {"dmd_score_loss": score_loss.detach().item(),
                                     "dmd_regression_loss": regression_loss.detach().item(), **diversity_parts}
                            parts["dmd_regression_lpips" if tr.get("dmd_regression_type") == "lpips_vgg"
                                  else "dmd_regression_mse"] = regression_loss.detach().item()
                        else:
                            loss = vsd_surrogate(generated, real_eps, fake_eps, sigma)
                            parts = {"vsd_loss": loss.detach().item()}
                    elif consistency:
                        with torch.no_grad():
                            clean = bundle.encode(images)
                            indices = torch.randint(len(root), (len(clean),), device=device)
                            t, end = endpoints[indices], endpoints[indices+1]
                            alpha, sigma = coefficients(bundle.alphas, t)
                            noisy = alpha*clean+sigma*torch.randn_like(clean)
                            real_eps = epsilon(teacher, noisy, t, conditional, unconditional,
                                               guidance, config, device)
                            xs = ddim_step(noisy, real_eps, t, end, bundle.alphas)
                            target = consistency_prediction(bundle, consistency_target, xs, end,
                                conditional, tr.get("consistency_sigma_data", .5))
                        predicted_x0 = consistency_prediction(bundle, student, noisy, t, conditional,
                            tr.get("consistency_sigma_data", .5))
                        loss = (predicted_x0-target).square().mean()
                        parts = {"consistency_mse": loss.detach().item()}
                    else:
                        with torch.no_grad():
                            clean = bundle.encode(images)
                            indices = torch.randint(steps, (len(clean),), device=device)*2
                            t, mid, end = endpoints[indices], endpoints[indices+1], endpoints[indices+2]
                            alpha, sigma = coefficients(bundle.alphas, t)
                            noisy = alpha*clean+sigma*torch.randn_like(clean)
                            real_eps = epsilon(teacher, noisy, t, conditional, unconditional,
                                               guidance, config, device)
                            xm = ddim_step(noisy, real_eps, t, mid, bundle.alphas)
                            real_mid = epsilon(teacher, xm, mid, conditional, unconditional,
                                               guidance, config, device)
                            xs = ddim_step(xm, real_mid, mid, end, bundle.alphas)
                            target = collapsed_x0_target(noisy, xs, t, end, bundle.alphas)
                        predicted = epsilon(student, noisy, t, conditional, None, 1, config, device)
                        loss, parts = sd_distillation_loss(bundle,
                            predict_x0(noisy, predicted, t, bundle.alphas), target, t)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite SD loss; inspect checkpoint and precision")
                    (loss/tr["grad_accumulation"]).backward()
                    loss_value += loss.detach().item()/tr["grad_accumulation"]
                    for key, value in parts.items():
                        if isinstance(value, (int, float)):
                            averaged_parts[key] = averaged_parts.get(key, 0.)+value/tr["grad_accumulation"]
                        else:
                            averaged_parts[key] = value
                parts = averaged_parts
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, tr["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                update_adapter_ema(ema, student, tr["ema_decay"])
                if consistency:
                    load_adapter(consistency_target, ema)
                if vsd:
                    for auxiliary in range(tr.get("fake_updates_per_step", 1)):
                        fake_optimizer.zero_grad(set_to_none=True)
                        fake_loss_value = 0.0
                        for micro in range(tr["grad_accumulation"]):
                            _, prompts = data.batch((global_step*tr.get("fake_updates_per_step", 1)+auxiliary)
                                                     *tr["grad_accumulation"]+micro, device)
                            conditional = bundle.embeddings(prompts)
                            with torch.no_grad():
                                z = torch.randn((len(prompts), 4, 64, 64), device=device)
                                terminal = root[0].expand(len(z))
                                student.eval()
                                generated = predict_x0(z, epsilon(student, z, terminal, conditional,
                                    None, 1, config, device), terminal, bundle.alphas)
                                student.train()
                                t = torch.randint(len(bundle.alphas), (len(z),), device=device)
                                alpha, sigma = coefficients(bundle.alphas, t)
                                noise = torch.randn_like(generated)
                                noisy = alpha*generated+sigma*noise
                                drop = (torch.zeros(len(z), dtype=torch.bool, device=device) if dmd else
                                        torch.rand(len(z), device=device) < .1)
                                conditions = torch.where(drop[:, None, None], empty, conditional)
                            fake_eps = epsilon(fake, noisy, t, conditions, None, 1, config, device)
                            fake_loss = ((fake_eps-noise).square().mean() if dmd else
                                         (alpha.square()*(fake_eps-noise).square()).mean())
                            if not torch.isfinite(fake_loss):
                                raise FloatingPointError("Nonfinite SD auxiliary denoising loss")
                            (fake_loss/tr["grad_accumulation"]).backward()
                            fake_loss_value += fake_loss.detach().item()/tr["grad_accumulation"]
                        torch.nn.utils.clip_grad_norm_(fake_parameters, tr["max_grad_norm"], error_if_nonfinite=True)
                        fake_optimizer.step()
                        auxiliary_updates += 1
                        parts["fake_loss"] = fake_loss_value
                global_step += 1
                elapsed = previous_elapsed+time.perf_counter()-started
                if (step+1) % tr["log_every"] == 0 or step == 0 or step+1 == tr["updates_per_stage"]:
                    record = {"stage_steps": steps, "stage_step": step+1, "global_step": global_step,
                              "loss": loss_value, "learning_rate": lr, "grad_norm": float(grad_norm),
                              "auxiliary_updates": auxiliary_updates, "elapsed_seconds": elapsed, **parts}
                    with (output / "train.jsonl").open("a") as handle:
                        handle.write(json.dumps(record, allow_nan=False)+"\n")
                    tracker.log(record)
                    if not config.get("progress_bar", True):
                        print(json.dumps(record), flush=True)
                bar.set_postfix(loss=f"{loss_value:.4g}", lr=f"{lr:.2g}", refresh=False)
                if (step+1) % tr["save_every"] == 0 or step+1 == tr["updates_per_stage"]:
                    save_state(checkpoint, config, student, ema, optimizer, stage_index, step+1,
                               global_step, elapsed, fake, fake_optimizer, auxiliary_updates)
                if (step+1) % tr["preview_every"] == 0 or step+1 == tr["updates_per_stage"]:
                    preview(bundle, student, ema, student_grid, output, tracker, global_step)
        metadata = {"method": config["name"], "seed": config["seed"], "global_step": global_step,
                    "initial_steps": tr["initial_steps"], "training_seconds": previous_elapsed+time.perf_counter()-started,
                    "training_hardware": env["hardware"], "teacher_identity": teacher_id,
                    "effective_batch_size": config["data"]["batch_size"]*tr["grad_accumulation"],
                    "trainable_parameters": sum(p.numel() for p in parameters),
                    "auxiliary_updates": auxiliary_updates, "training_uses_real_images": not (vsd or instaflow),
                    "training_prompt_hash": manifest_hash(data.records), "training_prompt_count": len(data.records),
                    "peak_training_memory_mb": torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0,
                    "stage_steps": steps, "fake_data": False,
                    "edge_space": "decoded-RGB" if tr["edge_weight"] else "none",
                    "edge_weight": tr["edge_weight"], "edge_schedule": tr["edge_schedule"],
                    "student_prediction_type": tr.get("instaflow_prediction_type", "rectified-flow-velocity") if instaflow else "consistency-x0" if consistency else "epsilon",
                    **({**pairs.metadata, **distilled_pairs.metadata,
                        "reflow_updates": 0 if endpoint_continuation else tr["updates_per_stage"]//2,
                        "flow_distillation_updates": tr["updates_per_stage"] if endpoint_continuation else tr["updates_per_stage"]-tr["updates_per_stage"]//2,
                        "flow_time_mapping": "nearest-VP-SNR" if tr.get("instaflow_prediction_type") == "epsilon-converted-velocity" else "round(999*(1-progress))"} if instaflow else {}),
                    **({"consistency_sigma_data": tr.get("consistency_sigma_data", .5),
                        "consistency_training_grid": root.cpu().tolist()} if consistency else {}),
                    **({"dmd_regression_weight": tr["dmd_regression_weight"],
                        "paired_regression_batch_size": config["data"]["batch_size"]*tr["grad_accumulation"],
                        **pairs.metadata} if dmd else {}),
                    "adaptation": ("InstaFlow-inspired: cached DDIM pairs, uniform-time reflow, Euler-recoupled one-step distillation; rank-matched LoRA; " +
                                   ("VP epsilon-to-velocity conversion" if tr.get("instaflow_prediction_type") == "epsilon-converted-velocity" else "direct velocity, 999 time scale")) if instaflow else
                                  "DMD-inspired normalized score matching + cached latent L2 regression; LoRA G/F" if dmd else
                                  "VP consistency distillation; adjacent DDIM teacher/EMA endpoints; latent L2; LoRA" if consistency else
                                  "SD1.5 LoRA student; conditional VSD with LoRA auxiliary scorer" if vsd
                                  else "SD1.5 LoRA progressive distillation; guidance baked into student"}
        from .continuation import continuation_metadata
        metadata.update(continuation_metadata(config, output, global_step,
            metadata["training_seconds"], auxiliary_updates))
        if dmd and "dmd_study" in config:
            from .dmd_diversity import study_metadata
            metadata.update(study_metadata(config, pairs, perceptual_metric))
        export_adapter(ema, output/f"stage_{steps}", student_grid, config, metadata)
        del student, optimizer, fake, fake_optimizer, ema, parameters, fake_parameters, consistency_target
    elapsed = previous_elapsed+time.perf_counter()-started
    atomic_json(output / "run.json", {"status": "complete", "method": config["name"],
                "seed": config["seed"], "global_step": global_step, "auxiliary_updates": auxiliary_updates,
                "training_seconds": elapsed, "config": config, "environment": env,
                **continuation_metadata(config, output, global_step, elapsed, auxiliary_updates)})
    tracker.summarize({"global_step": global_step, "auxiliary_updates": auxiliary_updates,
                       "training_seconds": elapsed})
    print(f"Training complete: {output}", flush=True)
