"""Direct one-step DMD with perceptual regression and same-prompt diversity targets.

This is a project extension inspired by DMD, RDD and CoDMD, not a claim of a
new distribution divergence or a reproduction of those published recipes.
"""

from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from .runtime import sha256

STUDY_POLICY = "dmd-perceptual-diversity-v1"
VARIANTS = {"dmd_grouped": ("latent_mse", 0.),
            "dmd_perceptual": ("lpips_vgg", 0.),
            "dmd_diversity": ("latent_mse", .1),
            "dmd_perceptual_diversity": ("lpips_vgg", .1)}


def apply_variant(config, variant, cache_dir):
    if variant not in VARIANTS or config["train"]["mode"] != "dmd":
        raise ValueError("DMD diversity study requires a recognized variant and DMD parent")
    result = deepcopy(config)
    regression, diversity = VARIANTS[variant]
    effective_batch = result["data"]["batch_size"]*result["train"]["grad_accumulation"]
    if effective_batch % 2:
        raise ValueError("Grouped DMD requires an even effective batch size")
    result["name"] = variant
    result["data"]["batch_size"] = 2
    result["train"].update(grad_accumulation=effective_batch//2, edge_weight=0., dmd_pair_seeds=2,
        dmd_regression_type=regression, dmd_diversity_weight=diversity,
        dmd_pair_cache_dir=str(Path(cache_dir).resolve()))
    result["wandb"]["group"] = "sd15-dmd-perceptual-diversity"
    result["dmd_study"] = {"policy": STUDY_POLICY, "variant": variant,
                            "pair_cache_dir": str(Path(cache_dir).resolve())}
    return result


def load_lpips(net, device):
    import lpips
    # Training uses VGG; evaluation uses AlexNet to avoid reusing the training metric.
    return lpips.LPIPS(net=net, version="0.1", verbose=False).to(device).eval().requires_grad_(False)


def lpips_identity(net):
    import lpips
    calibration = Path(lpips.__file__).parent/"weights"/"v0.1"/f"{net}.pth"
    backbone = {"vgg": "vgg16-397923af.pth", "alex": "alexnet-owt-7be5be79.pth"}[net]
    return {"implementation": "lpips-0.1.4", "net": net, "version": "0.1",
            "calibration_sha256": sha256(calibration), "backbone": backbone,
            "input_range": "decoded-RGB-clamp[-1,1]"}


def regression_and_diversity(bundle, metric, generated, target, config):
    tr = config["train"]
    perceptual = tr.get("dmd_regression_type", "latent_mse") == "lpips_vgg"
    diversity = tr.get("dmd_diversity_weight", 0.) > 0
    regression = (generated-target).square().mean()
    zero = generated.new_zeros(())
    if not (perceptual or diversity):
        return regression, zero, {}
    if diversity and (len(generated) < 2 or len(generated) % 2):
        raise ValueError("Diversity loss requires adjacent same-prompt image pairs")
    with torch.no_grad():
        target_rgb = torch.cat([bundle.decode(row).clamp(-1, 1) for row in target.split(1)])
        target_distance = (metric(target_rgb[0::2], target_rgb[1::2]).flatten()
                           if diversity else zero)

    # Separate decoder checkpoints permit one image's VAE backward at a time.
    # An outer checkpoint over both decodes would rebuild both large VAE graphs.
    rgb = torch.cat([checkpoint(lambda row: bundle.decode(row).clamp(-1, 1), row,
                                use_reentrant=False) for row in generated.split(1)])

    def rgb_losses(rgb):
        perceptual_loss = metric(rgb, target_rgb).mean() if perceptual else zero
        distance = metric(rgb[0::2], rgb[1::2]).flatten() if diversity else zero
        diversity_loss = (distance-target_distance).square().mean() if diversity else zero
        return perceptual_loss, diversity_loss, distance.mean()

    perceptual_loss, diversity_loss, distance = checkpoint(rgb_losses, rgb, use_reentrant=False)
    parts = {"dmd_diversity_loss": diversity_loss.detach().item()}
    if diversity:
        parts.update(student_pair_lpips_vgg=distance.detach().item(),
                     teacher_pair_lpips_vgg=target_distance.detach().mean().item())
    if perceptual:
        regression = perceptual_loss
    return regression, diversity_loss, parts


def study_metadata(config, pairs, metric):
    if "dmd_study" not in config:
        return {}
    return {"dmd_study_policy": STUDY_POLICY,
        "dmd_study_parent_config_hash": config["continuation"]["parent_config_hash"],
        "dmd_regression_type": config["train"]["dmd_regression_type"],
        "dmd_diversity_weight": config["train"]["dmd_diversity_weight"],
        "dmd_pair_seeds": 2, "teacher_pair_unique_prompts": len(pairs.rows)//2,
        "teacher_pair_protocol_hash": pairs.metadata["protocol_hash"],
        "teacher_pair_weights_sha256": pairs.metadata["weights_sha256"],
        "perceptual_metric": lpips_identity("vgg") if metric is not None else None,
        "adaptation": "DMD-inspired direct one-step LoRA; shared same-prompt teacher pairs; "
                      "factorial latent-MSE/LPIPS-VGG regression and teacher-calibrated pair-distance matching"}
