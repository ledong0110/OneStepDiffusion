import copy
import json

import pytest
import torch

from few_step_diffusion.dmd_noise_agreement import (
    MODES, POLICY, PAIR_MODES, PAIR_POLICY, combine_score_probes, normalized_score_direction, paired_score_surrogate,
)
from few_step_diffusion.distillation_objectives import dmd_surrogate


def test_each_probe_matches_existing_dmd_surrogate_gradient_and_detaches_scorers():
    generated = torch.randn(2, 4, 8, 8, requires_grad=True)
    noisy = torch.randn_like(generated, requires_grad=True)
    teacher = torch.randn_like(generated, requires_grad=True)
    fake = torch.randn_like(generated, requires_grad=True)
    t, alphas = torch.tensor([2, 6]), torch.linspace(.98, .1, 10)
    direction = normalized_score_direction(generated, noisy, teacher, fake, t, alphas)
    old_loss = dmd_surrogate(generated, noisy, teacher, fake, t, alphas)
    old_gradient, = torch.autograd.grad(old_loss, generated)
    loss, _ = paired_score_surrogate(generated, [direction, direction], "paired_mean", 2)
    loss.backward()
    torch.testing.assert_close(generated.grad, old_gradient)
    torch.testing.assert_close(generated.grad, direction/generated.numel())
    assert teacher.grad is None and fake.grad is None and noisy.grad is None


def test_spatial_filter_preserves_sample_norm_and_is_not_just_scalar_shrinkage():
    first = torch.tensor([1., 1., 1., 1.]).reshape(1, 1, 2, 2).requires_grad_()
    second = torch.tensor([1., -.5, 1., -.5]).reshape(1, 1, 2, 2).requires_grad_()
    mean, agreement, _ = combine_score_probes(first, second, "paired_mean", 1)
    filtered, _, stats = combine_score_probes(first, second, "agreement", 1)
    shifted, shifted_raw_agreement, _ = combine_score_probes(first, second, "shifted_agreement", 1)
    torch.testing.assert_close(agreement, torch.tensor([1., 0., 1., 0.]).reshape(1, 1, 2, 2))
    torch.testing.assert_close(shifted_raw_agreement, agreement)
    torch.testing.assert_close(filtered.norm(), mean.norm())
    torch.testing.assert_close(shifted.norm(), mean.norm())
    assert filtered[0, 0, 0, 0] > mean[0, 0, 0, 0]
    assert filtered[0, 0, 0, 1] < mean[0, 0, 0, 1]
    assert shifted[0, 0, 0, 1] > mean[0, 0, 0, 1]
    assert stats["score_filter_relative_change"] > 0 and not filtered.requires_grad
    generated = torch.zeros_like(first, requires_grad=True)
    plain_loss, _ = paired_score_surrogate(generated, [first, second], "paired_mean", 1)
    filtered_loss, _ = paired_score_surrogate(generated, [first, second], "agreement", 1)
    torch.testing.assert_close(filtered_loss, plain_loss)
    plain_gradient, = torch.autograd.grad(plain_loss, generated)
    filtered_gradient, = torch.autograd.grad(filtered_loss, generated)
    torch.testing.assert_close(plain_gradient, mean/generated.numel())
    torch.testing.assert_close(filtered_gradient, filtered/generated.numel())
    assert not torch.allclose(plain_gradient, filtered_gradient)


@pytest.mark.parametrize("mode", list(MODES))
def test_agreement_handles_identical_opposed_and_zero_probes_without_nan(mode):
    first = torch.randn(2, 4, 16, 16)
    unchanged, _, _ = combine_score_probes(first, first, mode, 8)
    torch.testing.assert_close(unchanged, first)
    zero, _, _ = combine_score_probes(first, -first, mode, 8)
    assert torch.count_nonzero(zero) == 0
    zero, _, _ = combine_score_probes(torch.zeros_like(first), torch.zeros_like(first), mode, 8)
    assert torch.isfinite(zero).all() and torch.count_nonzero(zero) == 0
    second = torch.randn_like(first)
    forward, _, _ = combine_score_probes(first, second, mode, 8)
    reverse, _, _ = combine_score_probes(second, first, mode, 8)
    torch.testing.assert_close(forward, reverse)
    torch.testing.assert_close(forward.flatten(1).norm(dim=1), (.5*(first+second)).flatten(1).norm(dim=1))


def test_score_filter_rejects_bad_shapes_patch_sizes_and_nonfinite_values():
    first = torch.ones(1, 4, 8, 8)
    with pytest.raises(ValueError, match="patches"):
        combine_score_probes(first, first, "agreement", 3)
    with pytest.raises(ValueError, match="matching BCHW"):
        combine_score_probes(first, first[:, :1], "agreement")
    with pytest.raises(FloatingPointError):
        combine_score_probes(first, first*float("nan"), "agreement")


def test_bounded_weights_limit_latent_rotation_and_bound_is_attainable():
    # Independent bound: cos(angle) >= 2*sqrt(2)/3 for weights in [1,2].
    # This construction gives paired mean (sqrt(2),1) and weights (1,2).
    first = torch.tensor([2*2**.5, 1.]).reshape(1, 1, 1, 2)
    second = torch.tensor([0., 1.]).reshape(1, 1, 1, 2)
    filtered, _, _ = combine_score_probes(first, second, "agreement", 1)
    mean = .5*(first+second)
    cosine = (filtered*mean).sum()/(filtered.norm()*mean.norm())
    assert cosine.item() == pytest.approx(2*2**.5/3, abs=1e-6)
    generator = torch.Generator().manual_seed(718)
    first = torch.randn(32, 4, 16, 16, generator=generator)
    second = torch.randn(first.shape, generator=generator)
    mean = .5*(first+second)
    for mode in MODES:
        filtered, _, _ = combine_score_probes(first, second, mode, 4)
        cosine = (filtered*mean).flatten(1).sum(1)/(filtered.flatten(1).norm(dim=1)*mean.flatten(1).norm(dim=1))
        assert torch.all(cosine >= 2*2**.5/3-1e-6)


@pytest.mark.parametrize("noise_pairs", [1, 2])
def test_result_import_checks_actual_mode_probe_count_and_budget(tmp_path, noise_pairs):
    from few_step_diffusion.config import load_config
    from few_step_diffusion.reporting import collect_results
    from test_sd_reporting import sd_result
    records = []
    policy = PAIR_POLICY if noise_pairs == 2 else POLICY
    for mode, method in (PAIR_MODES if noise_pairs == 2 else MODES).items():
        record = sd_result()
        config = load_config("configs/dmd.yaml")
        config["name"] = method
        config["train"].update(dmd_noise_mode=mode, dmd_agreement_patch_size=8,
                              dmd_regression_type="lpips_vgg", dmd_diversity_weight=0.)
        if noise_pairs == 2:
            config["train"]["dmd_noise_pairs"] = 2
        record.update(method=method, resolved_config=config, training_updates=500,
            cumulative_training_updates=36500, dmd_noise_mode=mode, dmd_score_probes=2*noise_pairs,
            dmd_noise_pairs=noise_pairs, dmd_score_estimator_policy=policy, dmd_agreement_patch_size=8,
            dmd_regression_type="lpips_vgg", dmd_diversity_weight=0.,
            refinement_policy=policy, refinement_updates=500,
            refinement_parent_checkpoint_sha256="c"*64, teacher_pair_protocol_hash="d"*64,
            teacher_pair_weights_sha256="e"*64,
            dmd_refinement={"policy": policy, "parent_training_updates": 36000,
                "parent_checkpoint_sha256": "c"*64, "additional_updates": 500,
                "training_prompt_hash": record["distillation_prompt_hash"]})
        (tmp_path/f"{method}.result.json").write_text(json.dumps(record))
        records.append(record)
    imported, skipped = collect_results(tmp_path, "schemas/result.schema.json")
    assert len(imported) == 3 and not skipped
    original = records[0]
    path = tmp_path/f"{original['method']}.result.json"
    for field, invalid in [("dmd_score_probes", 1), ("dmd_noise_mode", "paired_mean"),
                           ("cumulative_training_updates", 36000), ("dmd_agreement_patch_size", 4),
                           ("dmd_noise_pairs", 3)]:
        changed = copy.deepcopy(original)
        changed[field] = invalid
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="noise-agreement"):
            collect_results(tmp_path, "schemas/result.schema.json")
    path.write_text(json.dumps(original))


def test_independent_pair_means_avoid_reweighting_already_cancelled_linear_noise():
    signal = torch.ones(1, 1, 1, 2)
    first_noise = torch.tensor([0., 2.]).reshape_as(signal)
    second_noise = torch.tensor([3., 0.]).reshape_as(signal)
    probes = [signal+first_noise, signal-first_noise, signal+second_noise, signal-second_noise]
    generated = torch.zeros_like(signal, requires_grad=True)
    old_loss, _ = paired_score_surrogate(generated, probes[:2], "agreement", 1)
    old_gradient, = torch.autograd.grad(old_loss, generated)
    assert not torch.allclose(old_gradient, signal/generated.numel())
    for mode in MODES:
        loss, _ = paired_score_surrogate(generated, probes, mode, 1, noise_pairs=2)
        actual, = torch.autograd.grad(loss, generated)
        torch.testing.assert_close(actual, signal/generated.numel())
    random = [torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(i)) for i in range(4)]
    generated = torch.zeros_like(random[0], requires_grad=True)
    loss, _ = paired_score_surrogate(generated, random, "paired_mean", 4, noise_pairs=2)
    actual, = torch.autograd.grad(loss, generated)
    torch.testing.assert_close(actual, torch.stack(random).mean(0)/generated.numel())
    with pytest.raises(ValueError, match="antithetic pairs"):
        paired_score_surrogate(generated, random[:2], "agreement", noise_pairs=2)
