import torch

from few_step_diffusion.distillation_objectives import consistency_output, dmd_surrogate
from few_step_diffusion.diffusion import coefficients, predict_x0


def test_rectified_flow_target_integrates_from_noise_to_clean_in_one_call():
    from few_step_diffusion.distillation_objectives import rectified_flow_batch
    noise, clean = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    x, velocity, t = rectified_flow_batch(noise, clean, torch.tensor([0., .5, 1.]))
    torch.testing.assert_close(x[0], noise[0])
    torch.testing.assert_close(x[1], (noise[1]+clean[1])/2)
    torch.testing.assert_close(x[2], clean[2])
    torch.testing.assert_close(noise+velocity, clean)
    assert t.tolist() == [999, 500, 0]


def test_vp_flow_conversion_recovers_the_oracle_velocity_and_endpoint_is_finite():
    from few_step_diffusion.distillation_objectives import flow_vp_inputs, epsilon_to_flow_velocity
    alphas = torch.linspace(.999, .005, 1000)
    t = torch.tensor([10, 500, 999])
    a, s = coefficients(alphas, t)
    u = (a/(a+s)).flatten()
    noise, clean = torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    inputs, matched_t = flow_vp_inputs(u[:, None, None, None]*clean+(1-u[:, None, None, None])*noise, u, alphas)
    assert torch.equal(matched_t, t)
    velocity = epsilon_to_flow_velocity(inputs, noise, matched_t, alphas)
    torch.testing.assert_close(velocity, clean-noise, rtol=1e-4, atol=1e-5)
    endpoint, end_t = flow_vp_inputs(noise, torch.zeros(3), alphas)
    assert (end_t == 999).all()
    assert torch.isfinite(epsilon_to_flow_velocity(endpoint, noise, end_t, alphas)).all()


def test_consistency_boundary_is_identity_and_uses_epsilon_backbone():
    alphas = torch.linspace(.99, .01, 1000)
    images = torch.randn(2, 4, 8, 8, requires_grad=True)
    eps = torch.randn_like(images, requires_grad=True)
    boundary = consistency_output(images, eps, torch.tensor([-1, -1]), alphas)
    torch.testing.assert_close(boundary, images, rtol=0, atol=0)
    prediction = consistency_output(images, eps, torch.tensor([999, 100]), alphas)
    assert not torch.allclose(prediction, predict_x0(images, eps, torch.tensor([999, 100]), alphas))
    prediction.square().mean().backward()
    assert eps.grad is not None and eps.grad.abs().sum() > 0


def test_dmd_stopped_gradient_has_correct_sign_and_per_sample_normalization():
    alphas = torch.linspace(.99, .01, 1000)
    t = torch.tensor([100, 900])
    generated = torch.randn(2, 4, 8, 8, requires_grad=True)
    noise = torch.randn_like(generated)
    alpha, sigma = coefficients(alphas, t)
    noisy = alpha*generated.detach()+sigma*noise
    real = torch.randn_like(generated, requires_grad=True)
    fake = torch.randn_like(generated, requires_grad=True)
    loss = dmd_surrogate(generated, noisy, real, fake, t, alphas)
    loss.backward()
    real_x0 = predict_x0(noisy, real.detach(), t, alphas)
    fake_x0 = predict_x0(noisy, fake.detach(), t, alphas)
    residual = (generated.detach()-real_x0).abs().mean((1, 2, 3), keepdim=True).clamp_min(1e-6)
    expected = (fake_x0-real_x0)/residual/generated.numel()
    torch.testing.assert_close(generated.grad, expected)
    assert real.grad is None and fake.grad is None
