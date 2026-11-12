"""Experimental spatial agreement filtering of paired DMD score estimates.

DMD, LPIPS, antithetic sampling and gradient normalization are prior techniques.
The candidate tests a particular local agreement map and its norm-preserving
application to the DMD latent update. It is biased, is not a calibrated measure
of uncertainty, and makes no claim of novel theory or improved image quality.
"""

import torch
import torch.nn.functional as F

from .diffusion import predict_x0

POLICY = "dmd-paired-noise-spatial-agreement-v1"
PAIR_POLICY = "dmd-independent-pair-spatial-agreement-v1"
POLICIES = (POLICY, PAIR_POLICY)
MODES = {"agreement": "dmd_noise_agreement", "paired_mean": "dmd_noise_mean",
         "shifted_agreement": "dmd_noise_shifted"}
PAIR_MODES = {"agreement": "dmd_pair_agreement", "paired_mean": "dmd_pair_mean",
              "shifted_agreement": "dmd_pair_shifted"}


def validate_score_mode(config):
    tr = config["train"]
    mode = tr.get("dmd_noise_mode")
    if mode is None:
        return
    patch = tr.get("dmd_agreement_patch_size")
    pairs = tr.get("dmd_noise_pairs", 1)
    names = PAIR_MODES if pairs == 2 else MODES
    if (not isinstance(pairs, int) or isinstance(pairs, bool) or pairs not in (1, 2) or
            mode not in names or config["name"] != names[mode] or tr["mode"] != "dmd" or
            tr.get("dmd_gradient_mode") is not None or tr.get("dmd_regression_type") != "lpips_vgg" or
            tr.get("dmd_diversity_weight", 0) != 0 or tr.get("edge_weight", 0) != 0 or
            not isinstance(patch, int) or isinstance(patch, bool) or patch < 1 or 64 % patch):
        raise ValueError("Noise-agreement DMD requires its own mode, LPIPS, valid patches and no extra losses")


def refinement_config(parent, parent_run, output, mode, checkpoint_sha, updates=500,
                      learning_rate=1e-5, warmup_steps=25, patch_size=8, *, noise_pairs=1):
    from .config import validate
    from .dmd_conflict import refinement_config as base_config
    if mode not in MODES or parent["train"].get("dmd_noise_mode") or parent["train"].get("dmd_gradient_mode"):
        raise ValueError("Noise-agreement pilots must fork the fixed ordinary DMD+LPIPS parent")
    result = base_config(parent, parent_run, output, "sum", checkpoint_sha,
                         updates, learning_rate, warmup_steps)
    if not isinstance(noise_pairs, int) or isinstance(noise_pairs, bool) or noise_pairs not in (1, 2):
        raise ValueError("Use one or two independent antithetic noise pairs")
    result["name"] = (PAIR_MODES if noise_pairs == 2 else MODES)[mode]
    result["train"].pop("dmd_gradient_mode")
    result["train"].update(dmd_noise_mode=mode, dmd_agreement_patch_size=patch_size)
    if noise_pairs == 2:
        result["train"]["dmd_noise_pairs"] = noise_pairs
    result["dmd_refinement"].update(policy=PAIR_POLICY if noise_pairs == 2 else POLICY)
    result["wandb"]["group"] = "sd15-dmd-pair-agreement" if noise_pairs == 2 else "sd15-dmd-noise-agreement"
    validate(result)
    return result


def validate_result(record):
    lineage = record.get("dmd_refinement", {})
    mode = record.get("dmd_noise_mode")
    resolved = record["resolved_config"]
    pairs = resolved["train"].get("dmd_noise_pairs", 1)
    policy = PAIR_POLICY if pairs == 2 else POLICY
    names = PAIR_MODES if pairs == 2 else MODES
    if (names.get(mode) != record["method"] or resolved["name"] != record["method"] or
            resolved["train"].get("dmd_noise_mode") != mode or record.get("refinement_policy") != policy or
            lineage.get("policy") != policy or record.get("dmd_score_estimator_policy") != policy or
            record.get("dmd_score_probes") != 2*pairs or record.get("dmd_noise_pairs", 1) != pairs or
            record.get("dmd_gradient_mode") is not None or
            record.get("dmd_agreement_patch_size") != record["resolved_config"]["train"].get("dmd_agreement_patch_size") or
            record.get("dmd_regression_type") != "lpips_vgg" or record.get("dmd_diversity_weight") != 0 or
            record["sampling_steps"] != 1 or record.get("refinement_updates") != record["training_updates"] or
            lineage.get("additional_updates") != record["training_updates"] or
            lineage.get("training_prompt_hash") != record["distillation_prompt_hash"] or
            record.get("cumulative_training_updates") != lineage.get("parent_training_updates", -1)+record["training_updates"] or
            record.get("refinement_parent_checkpoint_sha256") != lineage.get("parent_checkpoint_sha256") or
            not record.get("refinement_parent_checkpoint_sha256") or not record.get("teacher_pair_protocol_hash") or
            not record.get("teacher_pair_weights_sha256")):
        raise ValueError("Invalid DMD noise-agreement refinement result")
    validate_score_mode(record["resolved_config"])


@torch.no_grad()
def normalized_score_direction(generated, noisy, real_epsilon, fake_epsilon, t, alphas):
    """Use exactly the per-probe normalization of the existing DMD surrogate."""
    real = predict_x0(noisy.detach(), real_epsilon.detach(), t, alphas)
    fake = predict_x0(noisy.detach(), fake_epsilon.detach(), t, alphas)
    denominator = (generated.detach()-real).abs().flatten(1).mean(1).reshape(-1, 1, 1, 1)
    direction = (fake-real)/denominator.clamp_min(1e-6)
    if not torch.isfinite(direction).all():
        raise FloatingPointError("Nonfinite paired DMD score estimate")
    return direction.float()


@torch.no_grad()
def combine_score_probes(first, second, mode, patch_size=8):
    """Return a detached latent direction, raw spatial agreement, and diagnostics.

    Agreement is clipped normalized cross-energy within non-overlapping spatial
    patches and all latent channels. Weight=1+agreement stays in [1,2], then a
    per-image rescaling preserves the paired mean's Euclidean norm. Thus the
    largest possible normalization factor is bounded; no hard zero mask is used.
    """
    if mode not in MODES or first.shape != second.shape or first.ndim != 4:
        raise ValueError("Paired DMD requires a recognized mode and matching BCHW estimates")
    if (not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size < 1 or
            first.shape[-2] % patch_size or first.shape[-1] % patch_size):
        raise ValueError("Agreement patches must divide the latent height and width")
    first, second = first.detach().float(), second.detach().float()
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise FloatingPointError("Nonfinite paired DMD score estimate")
    mean = .5*(first+second)
    cross = F.avg_pool2d((first*second).mean(1, keepdim=True), patch_size)
    energy = F.avg_pool2d((first.square()+second.square()).mean(1, keepdim=True), patch_size)
    tiny = torch.finfo(torch.float32).tiny
    agreement = (2*cross/energy.clamp_min(tiny)).clamp(0, 1)
    weights = 1+agreement
    if mode == "shifted_agreement":
        weights = weights.roll((weights.shape[-2]//2, weights.shape[-1]//2), (-2, -1))
    if mode == "paired_mean":
        update = mean
    else:
        weights = weights.repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1)
        weighted = weights*mean
        norm = mean.flatten(1).norm(dim=1).reshape(-1, 1, 1, 1)
        weighted_norm = weighted.flatten(1).norm(dim=1).reshape(-1, 1, 1, 1)
        update = weighted*(norm/weighted_norm.clamp_min(tiny))
    if not torch.isfinite(update).all():
        raise FloatingPointError("Nonfinite filtered DMD score estimate")
    denominator = first.flatten(1).norm(dim=1)*second.flatten(1).norm(dim=1)
    cosine = (first*second).flatten(1).sum(1)/denominator.clamp_min(tiny)
    relative_change = (update-mean).flatten(1).norm(dim=1)/mean.flatten(1).norm(dim=1).clamp_min(tiny)
    statistics = {"score_probe_cosine": cosine.clamp(-1, 1).mean().item(),
        "score_spatial_agreement": agreement.mean().item(),
        "score_agreeing_patch_fraction": (agreement > 0).float().mean().item(),
        "score_filter_relative_change": relative_change.mean().item(),
        "score_paired_mean_norm": mean.flatten(1).norm(dim=1).mean().item(),
        "score_filtered_norm": update.flatten(1).norm(dim=1).mean().item()}
    return update, agreement, statistics


def paired_score_surrogate(generated, directions, mode, patch_size=8, *, noise_pairs=1):
    if noise_pairs not in (1, 2) or len(directions) != 2*noise_pairs:
        raise ValueError("Score probes must match one or two independent antithetic pairs")
    estimates = (directions if noise_pairs == 1 else
                 [.5*(directions[0]+directions[1]), .5*(directions[2]+directions[3])])
    update, agreement, statistics = combine_score_probes(*estimates, mode, patch_size)
    if generated.shape != update.shape:
        raise ValueError("DMD score direction and generated latent shapes differ")
    target = (generated.float()-update).detach()
    return .5*(generated.float()-target).square().mean(), statistics
