"""Offline conditional-model tests exercise adapters, frozen weights, and both resume paths."""

import copy
from types import SimpleNamespace

from diffusers import UNet2DConditionModel
import pytest
from safetensors.torch import load_file
import torch

from few_step_diffusion.config import load_config
from few_step_diffusion import sd_models, sd_training, training


def tiny_unet(config, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(996)
        model = UNet2DConditionModel(sample_size=64, in_channels=4, out_channels=4,
            layers_per_block=1, block_out_channels=(8, 16), norm_num_groups=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=8, attention_head_dim=2)
    return model.to(device).eval().requires_grad_(False)


class TinyBundle:
    def __init__(self, config, device):
        self.config, self.device = config, device
        self.unet = tiny_unet(config, device)
        self.alphas = torch.linspace(.99, .01, 1000).to(device)

    def embeddings(self, prompts):
        return torch.stack([torch.randn(4, 8, generator=torch.Generator().manual_seed(
            sum(prompt.encode()))) for prompt in prompts]).to(self.device)

    def encode(self, images):
        return torch.cat((images, images.mean(1, keepdim=True)), dim=1)*.2

    def decode(self, latents):
        return latents[:, :3].tanh()


class TinyStream:
    def __init__(self, config, images=True):
        self.seed, self.images = config["seed"], images
        self.records = [{"image_id": 1, "caption": "a quiet city street"}]

    def batch(self, microstep, device):
        generator = torch.Generator().manual_seed(self.seed*1000003+microstep)
        images = torch.randn(1, 3, 64, 64, generator=generator).to(device) if self.images else None
        return images, ["a quiet city street"]


@pytest.mark.parametrize("mode,interrupt_at", [("distill", 3), ("vsd", 1), ("consistency", 1), ("dmd", 1), ("instaflow", 1), ("instaflow", 2), ("instaflow_vp", 2)])
def test_sd_adapter_training_resumes_including_previous_stage_and_fake_scorer(
        tmp_path, monkeypatch, mode, interrupt_at):
    config = load_config("configs/sd15_proposed.yaml")
    if mode == "instaflow_vp":
        mode = "instaflow"
        config["train"]["instaflow_prediction_type"] = "epsilon-converted-velocity"
    config.update(device="cpu", precision="fp32", progress_bar=False)
    config["model"].update(lora_rank=2, gradient_checkpointing=False)
    config["data"].update(batch_size=1, train_limit=2)
    one_step = mode in ("vsd", "consistency", "dmd", "instaflow")
    config["train"].update(mode=mode, initial_steps=4, stages=[1] if one_step else [2, 1],
        updates_per_stage=2, grad_accumulation=1, save_every=1, log_every=1, warmup_steps=0)
    if mode == "dmd":
        config["train"].update(dmd_regression_weight=.25, dmd_pair_count=1)
    if mode == "instaflow":
        config["train"].update(instaflow_pair_count=1, updates_per_stage=4)
    config["wandb"]["mode"] = "disabled"
    monkeypatch.setattr(sd_training, "SDBundle", TinyBundle)
    monkeypatch.setattr(sd_training, "new_unet", tiny_unet)
    monkeypatch.setattr(sd_training, "CaptionStream", TinyStream)
    monkeypatch.setattr(sd_training, "preview", lambda *args: None)
    config["output_dir"] = str(tmp_path / "reference")
    training.train(config)
    resumed_config = copy.deepcopy(config)
    resumed_config["output_dir"] = str(tmp_path / "resumed")
    original = sd_training.save_state

    def interrupt(*args, **kwargs):
        original(*args, **kwargs)
        if args[7] == interrupt_at:
            raise InterruptedError("Simulated interruption after adapter checkpoint")

    monkeypatch.setattr(sd_training, "save_state", interrupt)
    with pytest.raises(InterruptedError):
        training.train(resumed_config)
    monkeypatch.setattr(sd_training, "save_state", original)
    training.train(resumed_config, resume=True)
    expected = load_file(str(tmp_path / "reference/stage_1/adapter.safetensors"))
    actual = load_file(str(tmp_path / "resumed/stage_1/adapter.safetensors"))
    assert expected and all("lora_" in name for name in expected)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    a = torch.load(tmp_path / "reference/last.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    for key in ("student", "fake") if mode in ("vsd", "dmd") else ("student",):
        for name in a[key]:
            torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
    assert a["global_step"] == b["global_step"]
    assert a["auxiliary_updates"] == b["auxiliary_updates"]


def test_cfg_batches_both_conditions_and_student_uses_one_branch():
    class Recorder:
        def __call__(self, inputs, t, encoder_hidden_states):
            self.batch_size = len(inputs)
            return SimpleNamespace(sample=encoder_hidden_states[:, 0, 0, None, None, None].expand_as(inputs))
    model = Recorder()
    x = torch.zeros(2, 4, 8, 8)
    positive, negative = torch.full((2, 4, 8), 2.), torch.full((2, 4, 8), -1.)
    config = {"precision": "fp32"}
    guided = sd_models.epsilon(model, x, torch.tensor([999, 500]), positive, negative,
                               7.5, config, torch.device("cpu"))
    assert model.batch_size == 4
    torch.testing.assert_close(guided, torch.full_like(x, 21.5))
    student = sd_models.epsilon(model, x, 999, positive, None, 1, config, torch.device("cpu"))
    assert model.batch_size == 2
    torch.testing.assert_close(student, torch.full_like(x, 2))


def test_vsd_logs_components_over_the_entire_accumulated_batch(tmp_path, monkeypatch):
    import json
    config = load_config("configs/sd15.yaml")
    config.update(device="cpu", precision="fp32", progress_bar=False, output_dir=str(tmp_path/"run"))
    config["train"].update(mode="vsd", stages=[1], updates_per_stage=2, grad_accumulation=2,
                           log_every=1, save_every=2)
    config["model"].update(lora_rank=2, gradient_checkpointing=False)
    config["wandb"]["mode"] = "disabled"
    monkeypatch.setattr(sd_training, "SDBundle", TinyBundle)
    monkeypatch.setattr(sd_training, "CaptionStream", TinyStream)
    monkeypatch.setattr(sd_training, "new_unet", tiny_unet)
    monkeypatch.setattr(sd_training, "preview", lambda *args: None)
    training.train(config)
    records = [json.loads(line) for line in (tmp_path/"run/train.jsonl").read_text().splitlines()]
    assert records[-1]["loss"] > 0
    for record in records:
        assert record["loss"] == record["vsd_loss"]


def test_decoded_rgb_edge_loss_reaches_adapter_and_freezes_backbone():
    config = load_config("configs/sd15_proposed.yaml")
    config.update(device="cpu", precision="fp32")
    config["model"].update(lora_rank=2, gradient_checkpointing=False)
    bundle = TinyBundle(config, torch.device("cpu"))
    student = sd_models.add_lora(tiny_unet(config, bundle.device), config)
    x, t = torch.randn(1, 4, 64, 64), torch.tensor([300])
    prediction = sd_models.epsilon(student, x, t, bundle.embeddings(["a lake"]), None,
                                   1, config, bundle.device)
    loss, parts = sd_training.sd_distillation_loss(bundle, prediction, torch.zeros_like(prediction), t)
    loss.backward()
    assert parts["rgb_edge_mse"] > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in student.named_parameters() if "lora_" in n)
    assert all(p.grad is None for n, p in student.named_parameters() if "lora_" not in n)
