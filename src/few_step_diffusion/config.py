"""Small YAML configuration loader with inheritance and strict dotted overrides."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml


def image_size(config):
    return config["data"].get("image_size", 32)


def dataset_name(config):
    return config["data"].get("dataset", "CIFAR-10")


def is_stable_diffusion(config):
    return config["model"].get("family") == "stable-diffusion"


def config_fingerprint(config):
    return hashlib.sha256(json.dumps(experiment_config(config), sort_keys=True).encode()).hexdigest()


def experiment_config(config):
    """Display/tracking settings may change when resuming an experiment."""
    return {key: value for key, value in config.items() if key not in ("wandb", "progress_bar", "study_seeds")}


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, overrides=(), _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Circular config inheritance: {path}")
    seen.add(path)
    config = yaml.safe_load(path.read_text()) or {}
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, _seen=seen), config)
    for override in overrides:
        key, separator, raw = override.partition("=")
        if not separator:
            raise ValueError(f"Override must be key=value: {override}")
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown config key: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown config key: {key}")
        target[parts[-1]] = yaml.safe_load(raw)
    validate(config)
    return config


def validate(c):
    for key in ("name", "seed", "output_dir", "device", "precision", "model", "data", "train", "eval"):
        if key not in c:
            raise ValueError(f"Missing config key: {key}")
    if c["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if not isinstance(image_size(c), int) or image_size(c) not in (32, 64, 128, 256, 512):
        raise ValueError("data.image_size must be 32, 64, 128, 256, or 512")
    if not c["data"]["fake"]:
        supported = {"CIFAR-10": 32, "CelebA-HQ": 256, "COCO-2017": 512}
        if dataset_name(c) not in supported or image_size(c) != supported[dataset_name(c)]:
            raise ValueError("Choose CIFAR-10/32, CelebA-HQ/256, or COCO-2017/512")
    if not isinstance(c.get("progress_bar", True), bool):
        raise ValueError("progress_bar must be true or false")
    tracking = c.get("wandb", {})
    if tracking.get("mode", "auto") not in ("auto", "online", "offline", "disabled"):
        raise ValueError("wandb.mode must be auto, online, offline, or disabled")
    if not isinstance(tracking.get("project", "few-step-diffusion"), str):
        raise ValueError("wandb.project must be a string")
    for section in (c, c["eval"]):
        if not isinstance(section["seed"], int) or section["seed"] < 0:
            raise ValueError("Training and evaluation seeds must be nonnegative integers")
    tr = c["train"]
    recipe = tr.get("vsd_recipe", "adapted")
    if recipe not in ("adapted", "swiftbrush-source-v1"):
        raise ValueError("Unknown vsd_recipe")
    if recipe == "swiftbrush-source-v1":
        parameterization = tr.get("student_parameterization", "full-unet")
        if parameterization not in ("full-unet", "lora"):
            raise ValueError("Invalid source-recipe student_parameterization")
        if parameterization == "lora":
            if (not isinstance(tr.get("fake_lora_rank"), int) or tr["fake_lora_rank"] < 1 or
                    tr.get("fake_lora_alpha", 0) <= 0 or not tr.get("fake_lora_targets")):
                raise ValueError("A LoRA source-recipe student needs explicit auxiliary LoRA settings")
        if not is_stable_diffusion(c) or tr["mode"] != "vsd":
            raise ValueError("The SwiftBrush source recipe requires Stable Diffusion VSD")
        if tr.get("fake_updates_per_step") != 1 or tr["warmup_steps"] != 0:
            raise ValueError("The source recipe uses one scorer update and constant learning rates")
        if tr["edge_weight"] != 0:
            raise ValueError("The SwiftBrush source recipe has no edge regularizer")
        if (tr.get("vsd_t_min"), tr.get("vsd_t_max")) != (.02, .98):
            raise ValueError("The source recipe samples generator score timesteps in [20,980]")
        if c["model"].get("lora_alpha", 0) <= 0 or not c["model"].get("lora_targets"):
            raise ValueError("The source recipe needs explicit auxiliary LoRA alpha and targets")
    if is_stable_diffusion(c):
        if dataset_name(c) != "COCO-2017" or image_size(c) != 512 or c["data"]["fake"]:
            raise ValueError("Stable Diffusion requires the real COCO-2017 512px profile")
        if tr["mode"] == "teacher":
            raise ValueError("Stable Diffusion uses a pretrained teacher; choose distill or vsd")
        if c["eval"]["reference_split"] != "validation":
            raise ValueError("Stable Diffusion evaluation uses the COCO validation split")
        if not isinstance(c["model"].get("lora_rank"), int) or c["model"]["lora_rank"] < 1:
            raise ValueError("model.lora_rank must be a positive integer")
        if c["model"].get("lora_alpha", c["model"]["lora_rank"]) <= 0:
            raise ValueError("model.lora_alpha must be positive")
        if c["model"].get("guidance_scale", 0) <= 1:
            raise ValueError("The guided teacher requires guidance_scale > 1")
        if not 1 <= c["data"].get("train_limit", 0) <= 118287:
            raise ValueError("data.train_limit must be in [1,118287]")
        offset = c["data"].get("train_offset", 0)
        if not isinstance(offset, int) or offset < 0 or offset+c["data"]["train_limit"] > 118287:
            raise ValueError("COCO train_offset + train_limit must fit the training split")
        if c["data"].get("sampling", "replacement") not in ("replacement", "shuffled_epochs"):
            raise ValueError("Unknown caption sampling policy")
        if c["eval"]["num_samples"] > 5000 or c["eval"]["num_real"] > 5000:
            raise ValueError("COCO validation has 5000 image/caption pairs")
    if tr["mode"] not in ("teacher", "distill", "vsd", "consistency", "dmd", "instaflow"):
        raise ValueError("Unknown train.mode")
    if tr["mode"] in ("consistency", "dmd", "instaflow") and not is_stable_diffusion(c):
        raise ValueError("Consistency, DMD, and InstaFlow comparisons require the SD15 profile")
    if tr["mode"] == "instaflow":
        if tr.get("instaflow_prediction_type", "rectified-flow-velocity") not in ("rectified-flow-velocity", "epsilon-converted-velocity"):
            raise ValueError("Invalid instaflow_prediction_type")
        if not isinstance(tr.get("instaflow_pair_count"), int) or tr["instaflow_pair_count"] < 1:
            raise ValueError("instaflow_pair_count must be a positive integer")
        if tr["updates_per_stage"] < 2:
            raise ValueError("InstaFlow needs at least two updates for reflow and distillation")
    if tr["mode"] == "consistency" and tr.get("consistency_sigma_data", .5) <= 0:
        raise ValueError("consistency_sigma_data must be positive")
    if tr["mode"] == "dmd":
        if tr.get("dmd_regression_weight", 0) <= 0:
            raise ValueError("DMD requires a positive paired-regression weight")
        if not isinstance(tr.get("dmd_pair_count"), int) or tr["dmd_pair_count"] < 1:
            raise ValueError("dmd_pair_count must be a positive integer")
        if tr.get("dmd_regression_type", "latent_mse") not in ("latent_mse", "lpips_vgg"):
            raise ValueError("Unknown DMD regression type")
        if tr.get("dmd_diversity_weight", 0) < 0 or tr.get("dmd_pair_seeds", 1) not in (1, 2):
            raise ValueError("Invalid DMD diversity settings")
        if tr.get("dmd_pair_seeds", 1) == 2 and (c["data"]["batch_size"] % 2 or tr["dmd_pair_count"] % 2):
            raise ValueError("Grouped DMD needs an even microbatch and even pair-image count")
        if tr.get("dmd_diversity_weight", 0) and tr.get("dmd_pair_seeds") != 2:
            raise ValueError("DMD diversity requires same-prompt teacher pairs")
        if (tr.get("dmd_regression_type", "latent_mse") != "latent_mse" or tr.get("dmd_diversity_weight", 0)) and tr["edge_weight"]:
            raise ValueError("The DMD perceptual/diversity study excludes edge supervision")
    if tr["edge_schedule"] not in ("constant", "noise_aware"):
        raise ValueError("edge_schedule must be constant or noise_aware")
    previous = tr["initial_steps"]
    if not isinstance(previous, int) or not 2 <= previous <= 1000:
        raise ValueError("initial_steps must be an integer in [2,1000]")
    if not tr["stages"]:
        raise ValueError("train.stages must not be empty")
    for steps in tr["stages"]:
        if tr["mode"] in ("vsd", "consistency", "dmd", "instaflow"):
            if tr["stages"] != [1]:
                raise ValueError("One-step methods use train.stages=[1]")
            break
        if not isinstance(steps, int) or steps < 1 or previous != 2 * steps:
            raise ValueError("Each progressive stage must halve the preceding step count")
        previous = steps
    for section, keys in (("data", ("batch_size",)), ("train", (
        "updates_per_stage", "grad_accumulation", "save_every", "log_every", "preview_every"
    )), ("eval", ("num_samples", "batch_size", "num_real", "benchmark_batch_size",
                 "benchmark_repeats"))):
        for key in keys:
            value = c[section][key]
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{section}.{key} must be a positive integer")
    if not 0 <= tr["ema_decay"] < 1 or tr["edge_weight"] < 0 or tr["min_snr_gamma"] <= 0:
        raise ValueError("Invalid EMA decay or loss weight")
    if tr["learning_rate"] <= 0 or tr["warmup_steps"] < 0 or tr["max_grad_norm"] <= 0:
        raise ValueError("Invalid optimizer settings")
    if c["eval"]["reference_split"] not in ("train", "test", "validation"):
        raise ValueError("reference_split must be train, test, or validation")
    comparison = c["eval"].get("comparison_steps")
    if comparison is not None:
        if not isinstance(comparison, int) or comparison < 1 or comparison not in tr["stages"]:
            raise ValueError("eval.comparison_steps must be a trained stage")
    if tr["mode"] in ("vsd", "dmd"):
        if not 0 < tr.get("vsd_t_min", 0.02) < tr.get("vsd_t_max", 0.98) < 1:
            raise ValueError("VSD requires 0 < vsd_t_min < vsd_t_max < 1")
        if tr.get("fake_learning_rate", 0.0001) <= 0:
            raise ValueError("fake_learning_rate must be positive")
        if not isinstance(tr.get("fake_updates_per_step", 1), int) or tr.get("fake_updates_per_step", 1) < 1:
            raise ValueError("fake_updates_per_step must be a positive integer")
