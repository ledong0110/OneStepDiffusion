"""Explicit VP/DDIM mathematics shared by training and sampling.

The -1 sentinel is the exactly clean boundary (alpha=1, sigma=0).
Every stage uses a nested subset of the original grid; rebuilding a generic
scheduler at each stage would silently change the trained transition targets.
"""

import torch
import torch.nn.functional as F


def make_grid(total_timesteps, steps, device="cpu"):
    if not 1 <= steps <= total_timesteps:
        raise ValueError("Invalid number of inference steps")
    if steps == 1:
        return torch.tensor([total_timesteps - 1], device=device, dtype=torch.long)
    return torch.linspace(total_timesteps - 1, 0, steps, device=device).round().long()


def nested_grid(root_grid, steps):
    if steps < 1 or len(root_grid) % steps:
        raise ValueError("Sampling steps must divide the original grid size")
    return root_grid[::len(root_grid) // steps]


def coefficients(alphas_cumprod, t, ndim=4):
    t = torch.as_tensor(t, device=alphas_cumprod.device, dtype=torch.long)
    a2 = alphas_cumprod[t.clamp(min=0)].float()
    a2 = torch.where(t < 0, torch.ones_like(a2), a2)
    shape = (-1,) + (1,) * (ndim - 1)
    return a2.sqrt().reshape(shape), (1 - a2).sqrt().reshape(shape)


def predict_x0(x, epsilon, t, alphas):
    alpha, sigma = coefficients(alphas, t, x.ndim)
    return (x.float() - sigma * epsilon.float()) / alpha


def ddim_step(x, epsilon, t, s, alphas):
    x0 = predict_x0(x, epsilon, t, alphas)
    alpha_s, sigma_s = coefficients(alphas, s, x.ndim)
    return alpha_s * x0 + sigma_s * epsilon.float()


def collapsed_x0_target(x_t, x_s, t, s, alphas):
    """Invert one deterministic DDIM step to reproduce a two-step endpoint."""
    alpha_t, sigma_t = coefficients(alphas, t, x_t.ndim)
    alpha_s, sigma_s = coefficients(alphas, s, x_t.ndim)
    ratio = sigma_s / sigma_t
    denominator = alpha_s - ratio * alpha_t
    if torch.any(denominator <= 0):
        raise ValueError("Distillation target requires a strictly decreasing noise interval")
    return (x_s.float() - ratio * x_t.float()) / denominator


def edge_error(prediction, target):
    """Mean squared differences of horizontal and vertical image gradients."""
    dx = (prediction[..., 1:] - prediction[..., :-1]) - (
        target[..., 1:] - target[..., :-1])
    dy = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :])
    return 0.5 * (dx.square().flatten(1).mean(1) + dy.square().flatten(1).mean(1))


def distillation_loss(prediction, target, t, alphas, gamma, edge_weight, edge_schedule):
    alpha, sigma = coefficients(alphas, t)
    snr = (alpha / sigma.clamp_min(1e-8)).square().flatten()
    weight = snr.clamp(max=gamma)
    mse = F.mse_loss(prediction, target, reduction="none").flatten(1).mean(1)
    edge = edge_error(prediction, target)
    # The proposed gate gives clean intervals greater edge supervision.
    gate = alpha.square().flatten() if edge_schedule == "noise_aware" else torch.ones_like(snr)
    loss = (weight * (mse + edge_weight * gate * edge)).mean()
    return loss, {"x0_mse": mse.detach().mean().item(),
                  "edge_mse": edge.detach().mean().item()}
