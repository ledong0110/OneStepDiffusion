"""Distinct objectives for VP consistency distillation and distribution matching.

These use SD latents/LoRA and are adaptations of the cited papers, not replicas
of their original EDM parameterization or full-model/LPIPS training recipes.
"""

import torch

from .diffusion import coefficients, predict_x0

CONSISTENCY_SAMPLER = "SD15-consistency-boundary-v1"
FLOW_SAMPLER = "SD15-rectified-flow-euler-v1"
PRECONDITIONED_FLOW_SAMPLER = "SD15-epsilon-converted-flow-euler-v1"


def consistency_output(x, epsilon, t, alphas, sigma_data=.5):
    """Boundary-conditioned clean prediction with f(x, clean)=x exactly.

    Rescale VP input to additive-noise coordinates y=x/alpha, rho=sigma/alpha.
    Blend identity and the epsilon-derived x0 estimate. This dimensionless output
    blend preserves SD's pretrained epsilon backbone; it is not EDM preconditioning.
    """
    alpha, sigma = coefficients(alphas, t)
    rho = sigma/alpha
    skip = sigma_data**2/(rho.square()+sigma_data**2)
    output = rho/(rho.square()+sigma_data**2).sqrt()
    prediction = predict_x0(x, epsilon, t, alphas)
    return skip*(x.float()/alpha)+output*prediction


def dmd_surrogate(generated, noisy, real_epsilon, fake_epsilon, t, alphas):
    """Stopped DMD gradient, normalized by mean absolute real-denoiser residual."""
    real_x0 = predict_x0(noisy.detach(), real_epsilon.detach(), t, alphas)
    fake_x0 = predict_x0(noisy.detach(), fake_epsilon.detach(), t, alphas)
    residual = (generated.detach()-real_x0).abs().flatten(1).mean(1).reshape(-1, 1, 1, 1)
    gradient = ((fake_x0-real_x0)/residual.clamp_min(1e-6)).detach()
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite normalized DMD gradient")
    target = (generated.float()-gradient).detach()
    return .5*(generated.float()-target).square().mean()


def rectified_flow_batch(noise, clean, progress):
    """Noise-to-data straight path, velocity target, and SD-compatible time labels."""
    progress = progress.reshape(-1, 1, 1, 1)
    return ((1-progress)*noise+progress*clean, clean-noise,
            ((1-progress.flatten())*999).round().long())


def flow_vp_inputs(x, progress, alphas):
    """Normalize the straight path and match its SNR to the pretrained VP grid.

    For u=a/(a+s), the input scale is 1/(a+s). At the pure-noise endpoint
    the nearest supported VP time is used; SD1.5 has nonzero terminal signal.
    """
    u = progress.float().reshape(-1)
    scale = (u.square()+(1-u).square()).sqrt()
    target_alpha = u/scale
    t = (target_alpha[:, None]-alphas.sqrt()[None, :]).abs().argmin(1)
    return x.float()/scale.reshape(-1, 1, 1, 1), t


def epsilon_to_flow_velocity(normalized_x, predicted_noise, t, alphas):
    """Use the pretrained clean estimate minus noise as the RF velocity."""
    return predict_x0(normalized_x, predicted_noise, t, alphas)-predicted_noise.float()


