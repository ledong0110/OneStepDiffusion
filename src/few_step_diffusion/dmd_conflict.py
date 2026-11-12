"""Asymmetric PCGrad-style DMD/LPIPS gradient combination and matched refinement.

Projection is applied once to the full accumulated parameter gradients. This is
an experimental application of gradient surgery, not a new optimizer or a claim
of improved sample quality. All variants calculate the same two gradients.
"""

from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path

import torch

from .runtime import atomic_json, sha256

POLICY = "dmd-lpips-conflict-refinement-v1"
MODES = {"sum": "dmd_conflict_control", "project": "dmd_conflict",
         "norm_match": "dmd_conflict_norm_control", "block_project": "dmd_block_conflict",
         "block_norm_match": "dmd_block_norm_control"}


def validate_gradient_mode(config):
    tr = config["train"]
    mode = tr.get("dmd_gradient_mode")
    if mode is None:
        return
    if (mode not in MODES or tr["mode"] != "dmd" or
            tr.get("dmd_regression_type") != "lpips_vgg" or
            tr.get("dmd_diversity_weight", 0) != 0 or tr.get("edge_weight", 0) != 0):
        raise ValueError("DMD gradient combination requires a recognized mode, LPIPS and no extra losses")


class GradientAccumulator:
    """Accumulate two independent backward passes without changing their scale."""

    def __init__(self, parameters, names=None):
        self.parameters = list(parameters)
        self.score = [torch.zeros_like(p, dtype=torch.float32) for p in self.parameters]
        self.perceptual = [torch.zeros_like(p, dtype=torch.float32) for p in self.parameters]
        self.used = [False for _ in self.parameters]
        self.groups = None
        if names is not None:
            groups = defaultdict(list)
            for i, (name, _) in enumerate(zip(names, self.parameters, strict=True)):
                if ".lora_" not in name:
                    raise ValueError("Module projection requires named LoRA parameters")
                groups[name.split(".lora_", 1)[0]].append(i)
            self.groups = list(groups.values())

    def _assign_blocks(self, mode, weight, total_perceptual_sq):
        if not self.groups:
            raise ValueError("Module projection requires named LoRA parameter groups")
        conflicts = 0
        projected_sq = total_perceptual_sq.new_zeros(())
        for indices in self.groups:
            dot = sum((self.score[i]*self.perceptual[i]).sum() for i in indices)
            a2 = sum(self.score[i].square().sum() for i in indices)
            b2 = sum(self.perceptual[i].square().sum() for i in indices)
            conflict = bool(dot < 0) and bool(a2 > 0) and bool(b2 > 0)
            cosine = (dot/(a2*b2).sqrt().clamp_min(torch.finfo(torch.float32).tiny)).clamp(-1, 1)
            scale = (1-cosine.square()).clamp_min(0).sqrt() if conflict else dot.new_tensor(1.)
            projected_sq += b2*scale.square()
            conflicts += int(conflict)
            for i in indices:
                p, a, b = self.parameters[i], self.score[i], self.perceptual[i]
                if not self.used[i]:
                    p.grad = None
                    continue
                if mode == "block_project" and conflict:
                    adjusted = b-(dot/a2.sqrt())*(a/a2.sqrt())
                elif mode == "block_norm_match" and conflict:
                    adjusted = b*scale
                else:
                    adjusted = b
                p.grad = (a+weight*adjusted).to(p.dtype)
        ratio = (projected_sq/total_perceptual_sq).clamp(0, 1).sqrt() if total_perceptual_sq > 0 else projected_sq.new_tensor(1.)
        return {"gradient_block_count": len(self.groups),
                "gradient_block_conflict_fraction": conflicts/len(self.groups),
                "lpips_projected_norm_ratio": ratio.item()}

    def add(self, score_loss, perceptual_loss, accumulation):
        for loss, buffers in ((score_loss, self.score), (perceptual_loss, self.perceptual)):
            gradients = torch.autograd.grad(loss/accumulation, self.parameters, allow_unused=True)
            for i, gradient in enumerate(gradients):
                if gradient is not None:
                    buffers[i].add_(gradient.detach().float())
                    self.used[i] = True

    @torch.no_grad()
    def assign(self, mode, weight):
        if mode not in MODES or not math.isfinite(weight) or weight < 0:
            raise ValueError("Invalid gradient combination")
        dot = sum((a*b).sum() for a, b in zip(self.score, self.perceptual, strict=True))
        score_sq = sum(a.square().sum() for a in self.score)
        perceptual_sq = sum(b.square().sum() for b in self.perceptual)
        if not torch.isfinite(torch.stack([dot, score_sq, perceptual_sq])).all():
            raise FloatingPointError("Nonfinite DMD/LPIPS component gradients")
        denominator = score_sq.sqrt()*perceptual_sq.sqrt()
        cosine = (dot/denominator.clamp_min(torch.finfo(torch.float32).tiny)).clamp(-1, 1)
        conflict = bool(dot < 0) and bool(score_sq > 0) and bool(perceptual_sq > 0)
        scale = (1-cosine.square()).clamp_min(0).sqrt() if conflict else dot.new_tensor(1.)
        score_norm = score_sq.sqrt()
        statistics = {"gradient_cosine": cosine.item(), "gradient_conflict": int(conflict),
                      "dmd_gradient_norm": score_norm.item(),
                      "lpips_gradient_norm": perceptual_sq.sqrt().item(),
                      "lpips_projected_norm_ratio": scale.item(), "gradient_mode": mode}
        if mode in ("block_project", "block_norm_match"):
            return {**statistics, **self._assign_blocks(mode, weight, perceptual_sq)}
        for i, (p, a, b) in enumerate(zip(self.parameters, self.score, self.perceptual, strict=True)):
            if not self.used[i]:
                p.grad = None
                continue
            if mode == "project" and conflict:
                adjusted = b - (dot/score_norm)*(a/score_norm)
            elif mode == "norm_match" and conflict:
                adjusted = b*scale
            else:
                adjusted = b
            p.grad = (a+weight*adjusted).to(p.dtype)
        return statistics


def refinement_config(parent, parent_run, output, mode, checkpoint_sha, updates=500,
                      learning_rate=1e-5, warmup_steps=25):
    from .config import config_fingerprint, validate
    tr = parent["train"]
    if (tr["mode"] != "dmd" or tr.get("dmd_regression_type") != "lpips_vgg" or
            tr.get("dmd_diversity_weight", 0) != 0 or tr.get("edge_weight", 0) != 0 or
            tr["stages"] != [1] or mode not in MODES):
        raise ValueError("Refinement requires a completed one-step DMD+LPIPS control")
    if updates < 1 or not math.isfinite(learning_rate) or learning_rate <= 0 or warmup_steps < 0:
        raise ValueError("Invalid refinement budget or learning rate")
    result = deepcopy(parent)
    for field in ("continuation", "dmd_study", "dmd_weight_sweep", "dmd_refinement"):
        result.pop(field, None)
    result.update(name=MODES[mode], output_dir=str(output))
    result["train"].update(updates_per_stage=updates, learning_rate=learning_rate,
        warmup_steps=warmup_steps, dmd_gradient_mode=mode, save_every=50,
        preview_every=250, log_every=10)
    parent_offset = parent.get("dmd_refinement", {}).get("data_update_offset", 0)
    result["dmd_refinement"] = {"policy": POLICY, "parent_run": str(Path(parent_run).resolve()),
        "parent_config_hash": config_fingerprint(parent), "parent_checkpoint_sha256": checkpoint_sha,
        "additional_updates": updates, "data_update_offset": parent_offset+tr["updates_per_stage"]}
    result["eval"].update(steps=[1], comparison_steps=1, retain_generated_images=False)
    result["wandb"]["group"] = "sd15-dmd-conflict-refinement"
    validate(result)
    return result


def initialize_refinement(config, output):
    from .config import config_fingerprint
    from .sd_data import caption_records, manifest_hash
    spec = config["dmd_refinement"]
    parent = Path(spec["parent_run"])
    from .dmd_noise_agreement import POLICIES as NOISE_POLICIES
    from .dmd_response import POLICY as RESPONSE_POLICY
    if spec["policy"] not in (POLICY, *NOISE_POLICIES, RESPONSE_POLICY) or parent.resolve() == output.resolve():
        raise ValueError("Refinement requires its own output folder and a recognized policy")
    run = json.loads((parent/"run.json").read_text())
    if run.get("status") != "complete" or config_fingerprint(run["config"]) != spec["parent_config_hash"]:
        raise ValueError("Refinement parent is incomplete or its configuration changed")
    digest = sha256(parent/"last.pt")
    if digest != spec["parent_checkpoint_sha256"]:
        raise ValueError("Pinned refinement parent checkpoint changed")
    state = torch.load(parent/"last.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((parent/"stage_1/metadata.json").read_text())
    if (config_fingerprint(state["config"]) != spec["parent_config_hash"] or
            state["stage_index"] != 0 or state["stage_step"] != run["config"]["train"]["updates_per_stage"] or
            state["global_step"] != metadata["global_step"] or state["global_step"] != run["global_step"]):
        raise ValueError("Refinement parent checkpoint/export is not complete or consistent")
    if spec["policy"] == RESPONSE_POLICY:
        from .dmd_response import refinement_config as response_config
        expected = response_config(state["config"], parent, config["output_dir"],
            config["train"]["dmd_response_variant"], digest, spec["additional_updates"],
            config["train"]["learning_rate"], config["train"]["warmup_steps"],
            calibration_file=spec["calibration_file"], calibration_sha256=spec["calibration_sha256"])
    elif spec["policy"] in NOISE_POLICIES:
        from .dmd_noise_agreement import refinement_config as noise_config
        expected = noise_config(state["config"], parent, config["output_dir"],
            config["train"]["dmd_noise_mode"], digest, spec["additional_updates"],
            config["train"]["learning_rate"], config["train"]["warmup_steps"],
            config["train"]["dmd_agreement_patch_size"], noise_pairs=config["train"].get("dmd_noise_pairs", 1))
    else:
        expected = refinement_config(state["config"], parent, config["output_dir"],
            config["train"]["dmd_gradient_mode"], digest, spec["additional_updates"],
            config["train"]["learning_rate"], config["train"]["warmup_steps"])
    if config_fingerprint(expected) != config_fingerprint(config):
        raise ValueError("Refinement changes settings outside the matched gradient/budget policy")
    rows = caption_records(config)
    if manifest_hash(rows) != metadata["training_prompt_hash"]:
        raise ValueError("Refinement must use the same verified training caption pool")
    preserved = ["student", "ema", "optimizer", "fake", "fake_optimizer", "torch_rng", "cuda_rng"]
    if any(key not in state for key in preserved):
        raise ValueError("Parent lacks full generator/scorer/optimizer/RNG state")
    lineage = {**spec, "parent_export_sha256": sha256(parent/"stage_1/adapter.safetensors"),
        "parent_training_updates": metadata.get("cumulative_training_updates", state["global_step"]),
        "parent_training_seconds": metadata.get("cumulative_training_seconds", state["elapsed_seconds"]),
        "parent_auxiliary_updates": metadata.get("cumulative_auxiliary_updates", state["auxiliary_updates"]),
        "training_prompt_hash": manifest_hash(rows), "training_prompt_count": len(rows),
        "preserved_state": preserved, "data_policy": "same-training-pool-continued-stream",
        "parent_history": {key: metadata[key] for key in ("continuation", "dmd_refinement") if key in metadata}}
    atomic_json(output/"refinement.json", lineage)
    state.update(config=config, stage_index=0, stage_step=0, global_step=0,
                 elapsed_seconds=0., auxiliary_updates=0)
    return state


def refinement_metadata(config, output, updates, elapsed, auxiliary_updates):
    if "dmd_refinement" not in config:
        return {}
    lineage = json.loads((output/"refinement.json").read_text())
    modes = ({key: config["train"][key] for key in ("dmd_response_mode", "dmd_response_variant",
              "dmd_response_weight", "dmd_response_gradient_fraction")}
             if config["train"].get("dmd_response_mode") else
             {"dmd_noise_mode": config["train"]["dmd_noise_mode"],
              "dmd_agreement_patch_size": config["train"]["dmd_agreement_patch_size"],
              "dmd_score_estimator_policy": lineage["policy"],
              "dmd_score_probes": 2*config["train"].get("dmd_noise_pairs", 1)}
             if config["train"].get("dmd_noise_mode") else
             {"dmd_gradient_mode": config["train"]["dmd_gradient_mode"]})
    if config["train"].get("dmd_noise_pairs"):
        modes["dmd_noise_pairs"] = config["train"]["dmd_noise_pairs"]
    return {"dmd_refinement": lineage, **modes,
        "refinement_parent_checkpoint_sha256": lineage["parent_checkpoint_sha256"],
        "refinement_policy": lineage["policy"], "refinement_updates": updates,
        "cumulative_training_updates": lineage["parent_training_updates"]+updates,
        "cumulative_training_seconds": lineage["parent_training_seconds"]+elapsed,
        "cumulative_auxiliary_updates": lineage["parent_auxiliary_updates"]+auxiliary_updates}


def validate_refinement_result(record):
    lineage = record.get("dmd_refinement", {})
    mode = record.get("dmd_gradient_mode")
    if (MODES.get(mode) != record["method"] or record.get("refinement_policy") != POLICY or
            lineage.get("policy") != POLICY or record.get("dmd_regression_type") != "lpips_vgg" or
            record.get("dmd_diversity_weight") != 0 or record["sampling_steps"] != 1 or
            record.get("refinement_updates") != record["training_updates"] or
            lineage.get("additional_updates") != record["training_updates"] or
            lineage.get("training_prompt_hash") != record["distillation_prompt_hash"] or
            record.get("cumulative_training_updates") !=
            lineage.get("parent_training_updates", -1)+record["training_updates"] or
            record.get("refinement_parent_checkpoint_sha256") != lineage.get("parent_checkpoint_sha256") or
            not record.get("refinement_parent_checkpoint_sha256") or
            not record.get("teacher_pair_protocol_hash") or not record.get("teacher_pair_weights_sha256")):
        raise ValueError("Invalid matched DMD conflict refinement result")
