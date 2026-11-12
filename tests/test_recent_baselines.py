"""Source-formula, gradient, accumulation and exact-resume checks for recent baselines."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from few_step_diffusion.config import load_config
from few_step_diffusion.recent_objectives import (
    DensityRatioWeights, SD15Discriminator, SOURCE_COMMIT, discriminator_loss,
    distribution_surrogate, generator_adversarial_loss, phase_for_update, validate_recent,
)
from few_step_diffusion.runtime import sha256

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/"third_party/fastgen_reference"


def test_pinned_source_integrity():
    source = json.loads((REFERENCE/"source.json").read_text())
    assert source["commit"] == SOURCE_COMMIT
    for item in source["files"]:
        assert sha256(REFERENCE/item["path"]) == item["sha256"]


def source_functions():
    """Execute the actual pinned pure loss functions, without installing FastGen."""
    tree = ast.parse((REFERENCE/"fastgen/methods/common_loss.py").read_text())
    names = {"mean_with_loss_scale", "mse_loss", "variational_score_distillation_loss",
             "gan_loss_generator", "gan_loss_discriminator"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch, "F": F, "Optional": Optional,
                 "assert_same": lambda **kw: None,
                 "to_high_precision": lambda *v: v[0].float() if len(v) == 1 else tuple(x.float() for x in v)}
    exec(compile(ast.Module(body=body, type_ignores=[]), "pinned_fastgen_losses", "exec"), namespace)
    return namespace


def test_source_loss_and_gradient_parity():
    ref = source_functions()
    torch.manual_seed(123)
    x = torch.randn(16, 4, 8, 8, requires_grad=True)
    teacher, fake = torch.randn_like(x), torch.randn_like(x)
    h = torch.linspace(.2, 1.8, 16, requires_grad=True)
    a = distribution_surrogate(x, teacher, fake, h)
    b = ref["variational_score_distillation_loss"](x, teacher, fake, h.detach())
    torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(a, x, retain_graph=True)[0]
    torch.testing.assert_close(ga, torch.autograd.grad(b, x)[0])
    assert torch.autograd.grad(a, h, allow_unused=True)[0] is None
    # Global weighting must survive splitting the batch into microbatches.
    micro = sum(distribution_surrogate(x[i:i+2], teacher[i:i+2], fake[i:i+2], h[i:i+2])
                for i in range(0, 16, 2))/8
    torch.testing.assert_close(torch.autograd.grad(micro, x)[0], ga)
    real, fake_logits = torch.randn(16, 1), torch.randn(16, 1)
    torch.testing.assert_close(discriminator_loss(real, fake_logits), ref["gan_loss_discriminator"](real, fake_logits))
    torch.testing.assert_close(generator_adversarial_loss(fake_logits), ref["gan_loss_generator"](fake_logits))


def test_density_weights_against_pinned_js_and_resume():
    tree = ast.parse((REFERENCE/"fastgen/methods/distribution_matching/f_distill.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FdistillModel")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_f_div_weighting_h")
    ns = {"torch": torch, "world_size": lambda: 1, "assert_same": lambda **kw: None,
          "to_high_precision": lambda *v: tuple(x.float() for x in v)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "pinned_js", "exec"), ns)
    settings = SimpleNamespace(ratio_lower=.1, ratio_upper=20., ratio_normalization=True, ratio_ema_rate=0.)
    source = SimpleNamespace(config=SimpleNamespace(f_distill=settings), bin_num=10,
        discriminator=SimpleNamespace(bins=torch.ones(10)),
        net=SimpleNamespace(noise_scheduler=SimpleNamespace(min_t=0., max_t=1.)),
        f_div_weighting_function=lambda r: 1-1/(1+r))
    logits = torch.linspace(-3, 3, 16).view(-1, 1)
    t = torch.tensor([15]*8+[515]*8)
    ours = DensityRatioWeights()
    expected = ns["_get_f_div_weighting_h"](source, logits, t.float()/1000)
    torch.testing.assert_close(ours(logits, t), expected)
    torch.testing.assert_close(ours.bins, source.discriminator.bins)
    assert expected.std() > .2  # Catch accidental per-singleton normalization.
    state = deepcopy(ours.state_dict())
    ours(logits, t, update=False)
    torch.testing.assert_close(ours.bins, state["bins"])
    resumed = DensityRatioWeights()
    resumed.load_state_dict(state)
    torch.testing.assert_close(resumed(logits+1, t), ours(logits+1, t))


def test_real_diffusers_bottleneck_input_gradient():
    from diffusers import UNet2DConditionModel
    from few_step_diffusion.sd_recent_training import teacher_features
    model = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=("CrossAttnDownBlock2D",), up_block_types=("CrossAttnUpBlock2D",),
        block_out_channels=(16,), layers_per_block=1, norm_num_groups=4,
        cross_attention_dim=16, attention_head_dim=4).eval().requires_grad_(False)
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    t = torch.tensor([5, 7])
    conditions = torch.randn(2, 4, 16)
    values = []
    hook = model.mid_block.register_forward_hook(lambda m, a, o: values.append(o))
    model(x, t, encoder_hidden_states=conditions)
    hook.remove()
    actual = teacher_features(model, x, t, conditions, {"precision": "fp32"}, torch.device("cpu"))
    torch.testing.assert_close(actual, values[0])
    disc = SD15Discriminator(16)
    generator_adversarial_loss(disc(actual)).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    assert not model.mid_block._forward_hooks


@pytest.mark.parametrize("method", ["dmd2", "f_distill"])
def test_exact_budget_and_data_phases(method):
    config = load_config(ROOT/f"configs/sd15_{method}_36k.yaml")
    assert phase_for_update(config, 17999) == (0, 17999)
    assert phase_for_update(config, 18000) == (1, 0)
    assert phase_for_update(config, 35999) == (1, 17999)
    assert config["train"]["updates_per_stage"] == 36000
    assert config["recent_baseline"]["auxiliary_updates"] == 324000
    assert config["eval"]["num_samples"] == config["eval"]["num_real"] == 5000
    with pytest.raises(ValueError):
        phase_for_update(config, 36000)
    bad = deepcopy(config)
    bad["train"]["gan_weight"] = 0
    with pytest.raises(ValueError):
        validate_recent(bad)


@pytest.mark.parametrize("method", ["dmd2", "f_distill"])
def test_training_save_resume_exact(monkeypatch, tmp_path, method):
    """Exercise the real loop and all three optimizers; only models/data are tiny."""
    import few_step_diffusion.sd_recent_training as train

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 8, 1)
            self.mid_block = nn.Identity()
            self.up = nn.Conv2d(8, 4, 1)
            self.config = SimpleNamespace(block_out_channels=[8])

        def forward(self, x, t, encoder_hidden_states):
            y = self.mid_block(F.avg_pool2d(self.conv(x), 8))
            return SimpleNamespace(sample=F.interpolate(self.up(y), size=x.shape[-2:]))

    class Bundle:
        def __init__(self, config, device):
            self.unet = pretrained()
            self.alphas = torch.linspace(.99, .1, 1000)

        def embeddings(self, prompts):
            return torch.ones(len(prompts), 2, 2)

        def encode(self, images):
            return images

    def pretrained():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            return TinyUNet()

    records = [{"id": i} for i in range(4)]
    streams = [SimpleNamespace(records=records[:2], config={"data": {"sampling": "replacement"}}),
               SimpleNamespace(records=records[2:], config={"data": {"sampling": "shuffled_epochs"}})]
    monkeypatch.setattr(train, "make_streams", lambda c: streams)
    monkeypatch.setattr(train, "caption_records", lambda c: records)
    monkeypatch.setattr(train, "microbatch", lambda s, i, d, real=False:
                        (torch.randn(2, 4, 64, 64) if real else None, ["a", "b"]))
    monkeypatch.setattr(train, "SDBundle", Bundle)
    monkeypatch.setattr(train, "new_unet", lambda c, d: pretrained())
    monkeypatch.setattr(train, "add_lora", lambda m, c: m.train())
    monkeypatch.setattr(train, "adapter_state", lambda m: {k: v.detach().clone() for k, v in m.state_dict().items()})
    monkeypatch.setattr(train, "load_adapter", lambda m, s: m.load_state_dict(s))
    monkeypatch.setattr(train, "update_adapter_ema", lambda ema, m, decay:
                        [ema[k].lerp_(v.detach(), 1-decay) for k, v in m.state_dict().items()])
    monkeypatch.setattr(train, "export_adapter", lambda *a: None)
    monkeypatch.setattr(train, "preview", lambda *a: None)
    monkeypatch.setattr(train, "checkpoint_evaluation", lambda *a: None)
    cfg = load_config(ROOT/f"configs/sd15_{method}_36k.yaml")
    cfg["precision"] = "fp32"
    cfg["progress_bar"] = False
    cfg["model"]["gradient_checkpointing"] = False
    cfg["train"].update(updates_per_stage=3, grad_accumulation=2, save_every=1)
    cfg["recent_baseline"]["phase1_updates"] = 1
    tracker = SimpleNamespace(log=lambda *a: None, summarize=lambda *a: None)
    full, split = tmp_path/"full", tmp_path/"split"
    for folder in (full, split):
        folder.mkdir()
        torch.manual_seed(5)
        train.train_recent(cfg, None, torch.device("cpu"), folder, folder/"last.pt", {"hardware": {}},
                           tracker, stop_after=1 if folder == split else None)
    saved = torch.load(split/"last.pt", weights_only=False)
    assert saved["global_step"] == 1 and saved["auxiliary_updates"] == 9
    assert not (split/"run.json").exists()
    train.train_recent(cfg, saved, torch.device("cpu"), split, split/"last.pt", {"hardware": {}}, tracker)
    a, b = [torch.load(p/"last.pt", weights_only=False) for p in (full, split)]
    assert a["global_step"] == b["global_step"] == 3
    assert b["auxiliary_updates"] == 27
    for key in ("student", "ema", "fake", "discriminator", "density_ratio"):
        for name in a[key]:
            torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
    for left, right in zip(a["optimizers"], b["optimizers"], strict=True):
        for index in left["state"]:
            for name in left["state"][index]:
                torch.testing.assert_close(left["state"][index][name], right["state"][index][name], rtol=0, atol=0)
