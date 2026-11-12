from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from few_step_diffusion.recent_performance import (
    RealEncoder, configure_optimizers, grouped_microbatch, micro_factor, options,
)


def test_unknown_optimization_is_rejected(monkeypatch):
    monkeypatch.setenv("FSD_RECENT_OPTIMIZATIONS", "change_loss")
    with pytest.raises(ValueError):
        options()


def test_default_execution_keeps_original_optimizer(monkeypatch):
    monkeypatch.delenv("FSD_RECENT_OPTIMIZATIONS", raising=False)
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.AdamW([parameter])
    configure_optimizers([optimizer], options(), torch.device("cpu"))
    assert optimizer.param_groups[0]["fused"] is None


def test_fused_resume_preserves_optimizer_moments_and_step():
    a = torch.nn.Parameter(torch.tensor([1., -2., 3.]))
    original = torch.optim.AdamW([a], lr=1e-5, weight_decay=.01)
    a.grad = torch.tensor([.1, .2, -.3])
    original.step()
    b = torch.nn.Parameter(a.detach().clone())
    resumed = torch.optim.AdamW([b], lr=1e-5, weight_decay=.01)
    resumed.load_state_dict(deepcopy(original.state_dict()))
    configure_optimizers([resumed], {"fused_adamw"}, torch.device("cpu"))
    for gradient in (torch.tensor([.3, -.1, .5]), torch.tensor([-.2, -.1, .2])):
        a.grad, b.grad = gradient.clone(), gradient.clone()
        original.step()
        resumed.step()
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(original.state[a][key], resumed.state[b][key])
    assert resumed.state[b]["step"].item() == 3


def test_helper_sampling_matches_original_posterior_and_rng():
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    posterior = DiagonalGaussianDistribution(torch.randn(2, 8, 8, 8))
    encoder = RealEncoder.__new__(RealEncoder)
    encoder.bundle = SimpleNamespace(vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=.18215)))
    encoder.main, encoder.enabled = torch.device("cpu"), True
    torch.manual_seed(77)
    expected = posterior.sample()*.18215
    expected_rng = torch.get_rng_state()
    torch.manual_seed(77)
    actual = encoder.finish(None, posterior)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("sampling", ["replacement", "shuffled_epochs"])
def test_grouping_preserves_caption_order_and_effective_batch(sampling):
    from few_step_diffusion.sd_data import training_indices
    config = {"seed": 42, "data": {"batch_size": 2, "sampling": sampling}}
    def loader(stream, index, device, real=False):
        indices = training_indices(config, 53, index)
        return indices[:, None].float() if real else None, [str(i) for i in indices.tolist()]
    first = (7*9+3)*8
    original = [loader(None, first+i, "cpu", real=True) for i in range(8)]
    expected_images = torch.cat([part[0] for part in original])
    expected_prompts = [p for _, prompts in original for p in prompts]
    for factor in (1, 2, 4, 8):
        groups = [grouped_microbatch(loader, None, first+i, "cpu", True, factor)
                  for i in range(0, 8, factor)]
        torch.testing.assert_close(torch.cat([part[0] for part in groups]), expected_images)
        assert [p for _, prompts in groups for p in prompts] == expected_prompts
        assert sum(len(prompts) for _, prompts in groups) == 16


def test_invalid_grouping_is_rejected(monkeypatch):
    monkeypatch.setenv("FSD_RECENT_MICRO_FACTOR", "3")
    with pytest.raises(ValueError):
        micro_factor(8)
