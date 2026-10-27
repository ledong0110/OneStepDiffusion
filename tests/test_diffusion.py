import pytest
import torch
from diffusers import DDPMScheduler

from few_step_diffusion.diffusion import (coefficients, collapsed_x0_target, ddim_step,
                                         make_grid, nested_grid)


def test_ddim_matches_oracle_forward_process():
    alphas = DDPMScheduler().alphas_cumprod
    x0, noise = torch.randn(3, 3, 8, 8), torch.randn(3, 3, 8, 8)
    t, s = torch.tensor([999, 500, 20]), torch.tensor([600, 100, -1])
    a, sigma = coefficients(alphas, t)
    a_s, sigma_s = coefficients(alphas, s)
    result = ddim_step(a*x0 + sigma*noise, noise, t, s, alphas)
    torch.testing.assert_close(result, a_s*x0 + sigma_s*noise, atol=2e-5, rtol=1e-4)


def test_collapsed_target_reconstructs_two_step_endpoint():
    alphas = DDPMScheduler().alphas_cumprod
    x = torch.randn(3, 3, 8, 8)
    t, mid, end = torch.tensor([999, 500, 100]), torch.tensor([800, 200, 0]), torch.tensor([600, 0, -1])
    middle = ddim_step(x, torch.randn_like(x), t, mid, alphas)
    endpoint = ddim_step(middle, torch.randn_like(x), mid, end, alphas)
    target = collapsed_x0_target(x, endpoint, t, end, alphas)
    a, sigma = coefficients(alphas, t)
    target_epsilon = (x-a*target)/sigma
    reconstructed = ddim_step(x, target_epsilon, t, end, alphas)
    torch.testing.assert_close(endpoint, reconstructed, atol=5e-5, rtol=1e-4)


def test_nested_grid_stays_on_teacher_endpoints():
    root = make_grid(1000, 64)
    for steps in (32, 16, 8, 4, 2, 1):
        teacher = nested_grid(root, 2*steps)
        assert torch.equal(nested_grid(root, steps), teacher[::2])
    assert nested_grid(root, 1).item() == 999
    with pytest.raises(ValueError):
        nested_grid(root, 7)
