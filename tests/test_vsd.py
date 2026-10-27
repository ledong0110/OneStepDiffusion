import json

import torch

from few_step_diffusion.config import load_config
from few_step_diffusion import training
from few_step_diffusion.vsd import vsd_surrogate


def test_vsd_gradient_is_stopped_score_difference():
    images = torch.randn(2, 3, 4, 4, requires_grad=True)
    real = torch.randn_like(images, requires_grad=True)
    fake = torch.randn_like(images, requires_grad=True)
    sigma = torch.tensor([0.3, 0.8]).reshape(2, 1, 1, 1)
    loss = vsd_surrogate(images, real, fake, sigma)
    loss.backward()
    expected = sigma.square() * (real.detach()-fake.detach()) / images.numel()
    torch.testing.assert_close(images.grad, expected)
    assert real.grad is None and fake.grad is None


def test_vsd_uses_no_training_images_and_accounts_for_auxiliary_updates(tmp_path, monkeypatch):
    config = load_config("configs/smoke.yaml")
    config["train"].update(mode="vsd", stages=[1], updates_per_stage=2)
    config["output_dir"] = str(tmp_path / "vsd")

    def forbidden_data(*args, **kwargs):
        raise AssertionError("VSD must not construct a real-image minibatch stream")

    monkeypatch.setattr(training, "BatchStream", forbidden_data)
    training.train(config)
    result = json.loads((tmp_path / "vsd/run.json").read_text())
    assert result["global_step"] == result["auxiliary_updates"] == 2
    assert result["training_uses_real_images"] is False
