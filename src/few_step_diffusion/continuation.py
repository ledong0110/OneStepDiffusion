"""Fork a completed one-step training state without modifying its parent run."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import torch

from .config import config_fingerprint, validate
from .runtime import atomic_json, sha256
from .sd_data import caption_records, manifest_hash


POLICY = "fresh-coco-one-step-v1"


def continuation_config(parent, parent_run, output, offset=50000, count=68287, updates=18000):
    config = deepcopy(parent)
    config["output_dir"] = str(output)
    config["data"].update(train_offset=offset, train_limit=count, sampling="shuffled_epochs")
    config["train"]["updates_per_stage"] = updates
    config["eval"].update(steps=[1], comparison_steps=1, retain_generated_images=False)
    config["wandb"]["group"] = ("sd15-fresh68287-swiftbrush-cfg4.5" if
        parent["train"].get("vsd_recipe") == "swiftbrush-source-v1" else "sd15-fresh68287-one-step")
    config["continuation"] = {"policy": POLICY, "parent_run": str(Path(parent_run).resolve()),
        "parent_config_hash": config_fingerprint(parent), "additional_updates": updates,
        "instaflow_endpoint_only": parent["train"]["mode"] == "instaflow"}
    validate(config)
    return config


def copy_verified(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256(source)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError(f"Frozen continuation teacher differs: {destination}")
    else:
        temporary = destination.with_suffix(".copy.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    return digest


def initialize_continuation(config, output):
    specification = config["continuation"]
    if specification["policy"] != POLICY:
        raise ValueError("Unknown continuation policy")
    parent = Path(specification["parent_run"])
    if parent.resolve() == output.resolve():
        raise ValueError("Continuation must use a new output folder")
    run = json.loads((parent/"run.json").read_text())
    if run.get("status") != "complete":
        raise ValueError("Wait for the parent training run to finish")
    if config_fingerprint(run["config"]) != specification["parent_config_hash"]:
        raise ValueError("Parent configuration changed after the continuation was planned")
    # Loading local training state retains G/F optimizers, EMA age and RNG state.
    checkpoint = parent/"last.pt"
    digest = sha256(checkpoint)
    if specification.get("parent_checkpoint_sha256", digest) != digest:
        raise ValueError("Pinned parent checkpoint changed after this study was scheduled")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if config_fingerprint(state["config"]) != specification["parent_config_hash"]:
        raise ValueError("Parent checkpoint configuration differs from the completed run")
    metadata = json.loads((parent/"stage_1/metadata.json").read_text())
    stages = state["config"]["train"]["stages"]
    if (stages[-1] != 1 or state["stage_index"] != len(stages)-1 or
            state["stage_step"] != state["config"]["train"]["updates_per_stage"] or
            metadata["global_step"] != state["global_step"] or
            run["global_step"] != state["global_step"]):
        raise ValueError("Parent is not a completed, exported one-step checkpoint")
    expected = continuation_config(state["config"], parent, config["output_dir"],
        config["data"]["train_offset"], config["data"]["train_limit"],
        specification["additional_updates"])
    if "dmd_study" in config:
        from .dmd_diversity import apply_variant
        expected = apply_variant(expected, config["dmd_study"]["variant"],
                                 config["dmd_study"]["pair_cache_dir"])
    if "parent_checkpoint_sha256" in specification:
        expected["continuation"]["parent_checkpoint_sha256"] = specification["parent_checkpoint_sha256"]
    if config_fingerprint(expected) != config_fingerprint(config):
        raise ValueError("Continuation changes settings outside the supported data/budget changes")
    old, fresh = caption_records(state["config"]), caption_records(config)
    if metadata["training_prompt_hash"] != manifest_hash(old):
        raise ValueError("Parent export's training manifest differs from its configuration")
    old_ids, fresh_ids = {row["image_id"] for row in old}, {row["image_id"] for row in fresh}
    if old_ids & fresh_ids:
        raise ValueError("Continuation training images overlap the parent training pool")
    lineage = {"policy": POLICY, "parent_run": str(parent.resolve()),
        "parent_checkpoint_sha256": digest, "parent_config_hash": specification["parent_config_hash"],
        "parent_export_sha256": sha256(parent/"stage_1/adapter.safetensors"),
        "parent_training_updates": metadata.get("cumulative_training_updates", state["global_step"]),
        "parent_training_seconds": metadata.get("cumulative_training_seconds", state["elapsed_seconds"]),
        "parent_auxiliary_updates": metadata.get("cumulative_auxiliary_updates", state.get("auxiliary_updates", 0)),
        "parent_training_prompt_count": len(old), "parent_training_prompt_hash": manifest_hash(old),
        "fresh_training_prompt_count": len(fresh), "fresh_training_prompt_hash": manifest_hash(fresh),
        "combined_unique_training_images": len(old_ids | fresh_ids), "training_image_overlap": 0,
        "additional_updates": specification["additional_updates"], "sampling_steps": 1,
        "preserved_state": ["student", "ema", "optimizer", "torch_rng", "cuda_rng"]}
    if state["config"]["train"]["mode"] == "distill":
        lineage["frozen_two_step_teacher_sha256"] = copy_verified(
            parent/"stage_2/adapter.safetensors", output/"stage_2/adapter.safetensors")
    if specification["instaflow_endpoint_only"]:
        lineage["frozen_reflow_teacher_sha256"] = copy_verified(
            parent/"reflow_adapter.safetensors", output/"reflow_adapter.safetensors")
    if "fake" in state:
        lineage["preserved_state"].extend(["fake", "fake_optimizer"])
    existing = output/"continuation.json"
    if existing.exists() and json.loads(existing.read_text()) != lineage:
        raise ValueError("Continuation lineage changed; use a new output directory")
    atomic_json(existing, lineage)
    state.update(config=deepcopy(config), stage_step=0, global_step=0,
                 elapsed_seconds=0.0, auxiliary_updates=0)
    return state


def continuation_metadata(config, output, updates, elapsed, auxiliary_updates=0):
    if "continuation" not in config:
        return {}
    lineage = json.loads((Path(output)/"continuation.json").read_text())
    return {"continuation": lineage, "continuation_only_steps": 1,
        "continuation_policy": POLICY,
        "parent_training_prompt_hash": lineage["parent_training_prompt_hash"],
        "parent_training_updates": lineage["parent_training_updates"],
        "cumulative_training_updates": lineage["parent_training_updates"]+updates,
        "cumulative_training_seconds": lineage["parent_training_seconds"]+elapsed,
        "cumulative_auxiliary_updates": lineage["parent_auxiliary_updates"]+auxiliary_updates}
