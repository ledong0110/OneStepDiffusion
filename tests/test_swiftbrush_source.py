import copy
import json

from safetensors.torch import load_file
import pytest
import torch

from few_step_diffusion import sd_models, sd_swiftbrush, training
from few_step_diffusion.config import load_config
from few_step_diffusion.vsd import vsd_surrogate
from test_sd_training import TinyBundle, TinyStream, tiny_unet


def test_swiftbrush_source_recipe_defaults_and_objective_gradients():
    config = load_config("configs/sd15_swiftbrush_source.yaml")
    assert config["train"]["learning_rate"] == 1e-6
    assert config["train"]["fake_learning_rate"] == 1e-3
    assert config["train"]["weight_decay"] == config["train"]["warmup_steps"] == 0
    assert config["model"]["lora_rank"] == 64
    assert config["model"]["lora_alpha"] == 108
    assert config["model"]["lora_targets"] == ["to_q", "to_v"]
    assert config["model"]["guidance_scale"] == 4.5
    lora_config = load_config("configs/sd15_swiftbrush_lora.yaml")
    assert lora_config["model"]["lora_rank"] == 32
    assert lora_config["model"]["lora_targets"] == ["to_q", "to_k", "to_v", "to_out.0"]
    assert lora_config["train"]["fake_lora_rank"] == 64
    assert lora_config["train"]["fake_lora_alpha"] == 108
    assert lora_config["train"]["fake_lora_targets"] == ["to_q", "to_v"]
    noise, prediction = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8, requires_grad=True)
    original = sd_swiftbrush.source_original(noise, prediction)
    torch.testing.assert_close(original, (noise-(1-.0047)**.5*prediction)/.0047**.5)
    # Check the injected gradient all the way back to the epsilon generator,
    # including the terminal reparameterization's sign and magnitude.
    real, fake = torch.randn_like(noise), torch.randn_like(noise)
    sigma = torch.tensor([.2, .8])[:, None, None, None]
    vsd_surrogate(original, real, fake, sigma).backward()
    expected = -(1-.0047)**.5/.0047**.5 * sigma.square()*(real-fake)/prediction.numel()
    torch.testing.assert_close(prediction.grad, expected, atol=1e-6, rtol=1e-5)
    scorer = torch.randn_like(noise, requires_grad=True)
    alpha = torch.tensor([.3, .9])[:, None, None, None]
    loss = sd_swiftbrush.source_scorer_loss(scorer, noise, alpha)
    loss.backward()
    torch.testing.assert_close(scorer.grad, 2*alpha.square()*(scorer.detach()-noise)/scorer.numel())


@pytest.mark.parametrize("parameterization", ["full-unet", "lora"])
def test_source_recipe_full_generator_resume_and_export(tmp_path, monkeypatch, parameterization):
    lora = parameterization == "lora"
    config = load_config("configs/sd15_swiftbrush_lora.yaml" if lora else "configs/sd15_swiftbrush_source.yaml")
    config.update(device="cpu", precision="fp32", progress_bar=False)
    config["model"].update(lora_rank=2, lora_alpha=3, gradient_checkpointing=False)
    if lora:
        config["train"]["fake_lora_rank"] = 2
    config["train"].update(updates_per_stage=3, grad_accumulation=2, save_every=1, log_every=1)
    config["wandb"]["mode"] = "disabled"
    monkeypatch.setattr(sd_swiftbrush, "SDBundle", TinyBundle)
    monkeypatch.setattr(sd_swiftbrush, "CaptionStream", TinyStream)
    monkeypatch.setattr(sd_swiftbrush, "new_unet", tiny_unet)
    monkeypatch.setattr(sd_swiftbrush, "preview_source", lambda *args: None)
    generator_calls = []
    original_epsilon = sd_swiftbrush.epsilon

    def recorded(model, *args, **kwargs):
        if getattr(model, "_fsd_prediction_type", None) == "swiftbrush-source-x0":
            generator_calls.append(1)
            if lora:
                assert all(p.requires_grad == ("lora_" in name) for name, p in model.named_parameters())
            else:
                assert all(p.requires_grad for p in model.parameters())
        return original_epsilon(model, *args, **kwargs)

    monkeypatch.setattr(sd_swiftbrush, "epsilon", recorded)
    config["output_dir"] = str(tmp_path/"reference")
    training.train(config)
    # Exactly one generator forward per microbatch; F reuses its detached output.
    assert len(generator_calls) == 3*2
    checkpoint = torch.load(tmp_path/"reference/last.pt", weights_only=False)
    assert checkpoint["student"] and all(("lora_" in key) == lora for key in checkpoint["student"])
    assert checkpoint["fake"] and all("to_q" in key or "to_v" in key for key in checkpoint["fake"])
    assert checkpoint["ema"]["optimization_step"] == 3
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == 1e-6
    records = [json.loads(line) for line in (tmp_path/"reference/train.jsonl").read_text().splitlines()]
    assert records[0]["loss"] == records[0]["grad_norm"] == 0
    assert records[1]["grad_norm"] > 0
    assert records[0]["ema_decay"] == 0 and records[1]["ema_decay"] == pytest.approx(2/11)
    changed = copy.deepcopy(config)
    changed["output_dir"] = str(tmp_path/"resumed")
    original_save = sd_swiftbrush.save_source_state

    def interrupt(*args, **kwargs):
        original_save(*args, **kwargs)
        raise InterruptedError("Simulated interruption after full-generator checkpoint")

    monkeypatch.setattr(sd_swiftbrush, "save_source_state", interrupt)
    with pytest.raises(InterruptedError):
        training.train(changed)
    monkeypatch.setattr(sd_swiftbrush, "save_source_state", original_save)
    training.train(changed, resume=True)
    weights = "adapter.safetensors" if lora else "diffusion_pytorch_model.safetensors"
    a = load_file(str(tmp_path/"reference/stage_1"/weights))
    b = load_file(str(tmp_path/"resumed/stage_1"/weights))
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    resumed = torch.load(tmp_path/"resumed/last.pt", weights_only=False)
    for name in checkpoint["fake"]:
        torch.testing.assert_close(checkpoint["fake"][name], resumed["fake"][name], rtol=0, atol=0)
    for a_ema, b_ema in zip(checkpoint["ema"]["shadow_params"], resumed["ema"]["shadow_params"], strict=True):
        torch.testing.assert_close(a_ema, b_ema, rtol=0, atol=0)
    assert checkpoint["python_rng"] == resumed["python_rng"]
    bundle = TinyBundle(config, torch.device("cpu"))
    metadata = sd_models.restore_export(bundle, tmp_path/"reference/stage_1")
    assert metadata["lora_rank"] == (2 if lora else 0)
    assert metadata["student_parameterization"] == parameterization
    noise = torch.randn(1, 4, 64, 64)
    embeds = bundle.embeddings(["a lake"])
    direct = sd_swiftbrush.source_original(noise, sd_models.epsilon(bundle.unet, noise,
        999, embeds, None, 1, config, bundle.device))
    actual = sd_models.generate_sd_latents(bundle, bundle.unet, torch.tensor([999]),
        ["a lake"], 123, noise=noise)
    torch.testing.assert_close(actual, direct, rtol=0, atol=0)
    exported_state = sd_models.adapter_state(bundle.unet) if lora else bundle.unet.state_dict()
    for name, tensor in exported_state.items():
        torch.testing.assert_close(tensor, a[name], rtol=0, atol=0)
    bundle.config = load_config("configs/sd15.yaml")
    with pytest.raises(ValueError, match="parameterization|source recipe"):
        sd_models.restore_export(bundle, tmp_path/"reference/stage_1")
