import copy

import pytest
import torch
from safetensors.torch import load_file

from few_step_diffusion.config import load_config
from few_step_diffusion import training


@pytest.mark.parametrize("mode", ["distill", "vsd"])
def test_interrupted_training_resumes_to_same_weights(tmp_path, monkeypatch, mode):
    config = load_config("configs/smoke.yaml", ["train.initial_steps=4", "train.stages=[2,1]",
                        "train.updates_per_stage=3", "train.preview_every=99"])
    if mode == "vsd":
        config["train"].update(mode="vsd", stages=[1], fake_updates_per_step=2)
    config["output_dir"] = str(tmp_path / "reference")
    training.train(config)
    interrupted = copy.deepcopy(config)
    interrupted["output_dir"] = str(tmp_path / "resumed")
    original_save = training.save_checkpoint

    def interrupt_once(*args, **kwargs):
        original_save(*args, **kwargs)
        raise InterruptedError("simulated interruption after a saved optimizer update")

    monkeypatch.setattr(training, "save_checkpoint", interrupt_once)
    with pytest.raises(InterruptedError):
        training.train(interrupted)
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    training.train(interrupted, resume=True)
    original = load_file(str(tmp_path / "reference/stage_1/diffusion_pytorch_model.safetensors"))
    resumed = load_file(str(tmp_path / "resumed/stage_1/diffusion_pytorch_model.safetensors"))
    for name in original:
        torch.testing.assert_close(original[name], resumed[name], rtol=0, atol=0)
    if mode == "vsd":
        a = torch.load(tmp_path / "reference/last.pt", weights_only=False)
        b = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
        for name in a["fake"]:
            torch.testing.assert_close(a["fake"][name], b["fake"][name], rtol=0, atol=0)
