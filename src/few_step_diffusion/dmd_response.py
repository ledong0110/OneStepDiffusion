"""Direction-only finite-response experiments on cached same-caption pairs.

Cosine alignment, relational distillation, and noise-response matching have
precedents. This is an experimental DMD extension, not a claim to invent them.
Independent cached noises define finite displacements, not local Jacobians.
"""

import json
import math
from pathlib import Path

import torch

POLICY = "dmd-finite-response-alignment-v1"
SETTINGS = {f"{mode}_{round(fraction*100)}": (mode, fraction)
            for mode in ("angular", "magnitude", "vector") for fraction in (.1, .3, .5)}
MODES = {key: f"dmd_response_{key}" for key in SETTINGS}


def response_pair_metadata(config, metadata):
    """Recover pair-count fields from the recorded teacher-bank protocol.

    Early response exports retained the complete bank protocol but omitted its
    flattened pair-count fields. Never infer these from a method name alone.
    """
    if not config["train"].get("dmd_response_mode"):
        return {}
    validate_response_mode(config)
    protocol = metadata.get("protocol", {})
    count, unique = protocol.get("count"), protocol.get("unique_prompts")
    if (metadata.get("method") != config["name"] or
            protocol.get("seeds_per_prompt") != 2 or
            protocol.get("grouping") != "adjacent-images-share-caption" or
            not isinstance(unique, int) or unique < 1 or count != 2*unique or
            count != config["train"]["dmd_pair_count"] or
            metadata.get("teacher_pair_count") != count):
        raise ValueError("Response export pair metadata differs from the recorded teacher-bank protocol")
    fields = {"dmd_pair_seeds": 2, "teacher_pair_unique_prompts": unique}
    if any(key in metadata and metadata[key] != value for key, value in fields.items()):
        raise ValueError("Response export has conflicting paired-seed metadata")
    return fields


def validate_response_mode(config):
    tr = config["train"]
    mode = tr.get("dmd_response_mode")
    if mode is None:
        if tr.get("dmd_response_weight", 0):
            raise ValueError("A response weight requires a response mode")
        return
    key = tr.get("dmd_response_variant")
    weight = tr.get("dmd_response_weight", 0)
    independent = config.get("independent_training", {}).get("policy") == "independent-sd15-response-36k-v1"
    expected_name = "response_scratch" if independent else MODES.get(key)
    if (key not in SETTINGS or SETTINGS[key] != (mode, tr.get("dmd_response_gradient_fraction")) or
            config["name"] != expected_name or not isinstance(weight, (int, float)) or
            not math.isfinite(weight) or weight <= 0 or tr["mode"] != "dmd" or
            tr.get("dmd_pair_seeds") != 2 or tr.get("dmd_regression_type") != "lpips_vgg" or
            tr.get("dmd_gradient_mode") or tr.get("dmd_noise_mode") or
            tr.get("dmd_diversity_weight", 0) or tr.get("edge_weight", 0)):
        raise ValueError("Response alignment requires calibrated paired DMD+LPIPS and no other experimental losses")


def calibrated_weight(path, digest, parent_sha, mode, fraction):
    from .runtime import sha256
    if sha256(path) != digest:
        raise ValueError("Response gradient calibration changed")
    calibration = json.loads(Path(path).read_text())
    if (calibration.get("policy") != "frozen-teacher-finite-response-diagnostic-v1" or
            calibration.get("parent_checkpoint_sha256") != parent_sha or
            calibration.get("training_only") is not True or
            calibration.get("weights_unchanged") is not True or calibration.get("optimizer_updates") != 0):
        raise ValueError("Response calibration must use the unchanged training-only reference")
    return calibration["gradient_calibration"]["coefficients"][mode]["weights"][str(fraction)]


def refinement_config(parent, parent_run, output, variant, checkpoint_sha, updates=500,
                      learning_rate=1e-5, warmup_steps=25, *, calibration_file, calibration_sha256):
    from .config import validate
    from .dmd_conflict import refinement_config as base_config
    if (variant not in SETTINGS or any(parent["train"].get(key) for key in
            ("dmd_response_mode", "dmd_gradient_mode", "dmd_noise_mode"))):
        raise ValueError("Response pilots must fork the fixed ordinary DMD+LPIPS parent")
    mode, fraction = SETTINGS[variant]
    weight = calibrated_weight(calibration_file, calibration_sha256, checkpoint_sha, mode, fraction)
    result = base_config(parent, parent_run, output, "sum", checkpoint_sha,
                         updates, learning_rate, warmup_steps)
    result["name"] = MODES[variant]
    result["train"].pop("dmd_gradient_mode")
    result["train"].update(dmd_response_variant=variant, dmd_response_mode=mode,
        dmd_response_weight=weight, dmd_response_gradient_fraction=fraction)
    result["dmd_refinement"].update(policy=POLICY, calibration_file=str(calibration_file),
                                   calibration_sha256=calibration_sha256)
    result["wandb"]["group"] = "sd15-dmd-response"
    validate(result)
    return result


def validate_result(record):
    config = record["resolved_config"]
    tr, lineage = config["train"], record.get("dmd_refinement", {})
    validate_response_mode(config)
    fields = ("dmd_response_mode", "dmd_response_variant", "dmd_response_weight",
              "dmd_response_gradient_fraction")
    if (record["method"] != config["name"] or record["method"] not in MODES.values() or
            any(record.get(key) != tr.get(key) for key in fields) or
            record.get("refinement_policy") != POLICY or lineage.get("policy") != POLICY or
            not lineage.get("calibration_sha256") or
            record.get("dmd_regression_type") != "lpips_vgg" or record.get("dmd_diversity_weight") != 0 or
            record.get("dmd_pair_seeds") != 2 or record["sampling_steps"] != 1 or
            record.get("refinement_updates") != record["training_updates"] or
            lineage.get("additional_updates") != record["training_updates"] or
            lineage.get("training_prompt_hash") != record["distillation_prompt_hash"] or
            record.get("cumulative_training_updates") != lineage.get("parent_training_updates", -1)+record["training_updates"] or
            record.get("refinement_parent_checkpoint_sha256") != lineage.get("parent_checkpoint_sha256") or
            not record.get("refinement_parent_checkpoint_sha256") or not record.get("teacher_pair_protocol_hash") or
            not record.get("teacher_pair_weights_sha256")):
        raise ValueError("Invalid DMD response-alignment result")


def response_loss(student, teacher, mode="angular"):
    """Compare responses without adding a new pointwise reconstruction target.

    For nonzero displacements, the angular gradient has zero pairwise sum and
    is perpendicular to the student displacement. Those are latent-space,
    first-order properties of this term alone, not guarantees about an Adam
    parameter update, image quality, or the combined DMD objective.
    """
    if (mode not in ("angular", "magnitude", "vector") or student.ndim != 4 or
            student.shape != teacher.shape or len(student) < 2 or len(student) % 2):
        raise ValueError("Response alignment needs matching adjacent same-caption image pairs")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise FloatingPointError("Nonfinite response-alignment input")
    s = student.float().reshape(-1, 2, *student.shape[1:])
    t = teacher.detach().float().reshape_as(s)
    ds = (s[:, 1]-s[:, 0]).flatten(1)
    dt = (t[:, 1]-t[:, 0]).flatten(1)
    ns, nt = ds.norm(dim=1), dt.norm(dim=1)
    valid = nt > 1e-6
    if not valid.any():
        return student.sum()*0, {"response_valid_pairs": 0,
            "response_cosine": 0., "response_norm_ratio": 0.}
    ds, dt, ns, nt = ds[valid], dt[valid], ns[valid], nt[valid]
    cosine = ((ds*dt).sum(1)/(ns.clamp_min(1e-6)*nt)).clamp(-1, 1)
    ratio = ns/nt
    if mode == "angular":
        loss = (1-cosine).mean()
    elif mode == "magnitude":
        loss = (ratio-1).square().mean()
    else:
        loss = ((ds-dt).square().sum(1)/nt.square()).mean()
    return loss, {"response_valid_pairs": int(valid.sum()),
        "response_cosine": cosine.detach().mean().item(),
        "response_norm_ratio": ratio.detach().mean().item()}
