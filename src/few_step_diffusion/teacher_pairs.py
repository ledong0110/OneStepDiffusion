"""Reproducible teacher couplings for DMD and two-stage rectified-flow training."""

import hashlib
import json
from pathlib import Path
import time

from safetensors.torch import load_file, save_file
import torch
from tqdm.auto import tqdm

from .runtime import atomic_json, sha256
from .sd_data import manifest_hash, training_indices
from .sd_models import generate_sd_latents


class TeacherPairs:
    def __init__(self, config, bundle, grid, output, records):
        self.config = config
        key = "instaflow_pair_count" if config["train"]["mode"] == "instaflow" else "dmd_pair_count"
        self.seeds_per_prompt = config["train"].get("dmd_pair_seeds", 1) if key == "dmd_pair_count" else 1
        unique = min(config["train"][key]//self.seeds_per_prompt, len(records))
        self.rows = [row for row in records[:unique] for _ in range(self.seeds_per_prompt)]
        count = len(self.rows)
        path = Path(output)/"teacher_pairs.safetensors"
        metadata_path = Path(output)/"teacher_pairs.json"
        protocol = {"teacher": config["model"]["source"], "revision": config["model"]["revision"],
                    "guidance_scale": config["model"]["guidance_scale"], "seed": config["seed"],
                    "timesteps": grid.cpu().tolist(), "prompt_hash": manifest_hash(self.rows),
                    "precision": config["precision"], "count": count}
        if self.seeds_per_prompt > 1:
            protocol.update(seeds_per_prompt=self.seeds_per_prompt, unique_prompts=unique,
                            grouping="adjacent-images-share-caption")
        fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        if path.exists() and metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text())
            if self.metadata["protocol_hash"] != fingerprint or self.metadata["weights_sha256"] != sha256(path):
                raise ValueError("Cached teacher pairs differ from the configured protocol")
            self.tensors = load_file(str(path))
        else:
            started = time.perf_counter()
            generator = torch.Generator().manual_seed(config["seed"]*1000003+714623)
            noise = torch.randn((count, 4, 64, 64), generator=generator)
            targets = []
            cached_generation_seconds = 0.
            batch_size = config["eval"]["batch_size"]
            for start in tqdm(range(0, count, batch_size), desc="Teacher noise/output pairs",
                              disable=not config.get("progress_bar", True)):
                prompts = [row["caption"] for row in self.rows[start:start+batch_size]]
                chunks = Path(output)/"teacher_pair_chunks"
                chunk = chunks/f"{start:06d}.safetensors"
                chunk_info = chunk.with_suffix(".json")
                if self.seeds_per_prompt == 2 and chunk.exists() and chunk_info.exists():
                    info = json.loads(chunk_info.read_text())
                    if info["protocol_hash"] != fingerprint or info["sha256"] != sha256(chunk):
                        raise ValueError("Partial teacher pair cache differs from its protocol")
                    target = load_file(str(chunk))["latents"]
                    if target.shape != noise[start:start+batch_size].shape:
                        raise ValueError("Partial teacher pair cache has an invalid shape")
                    cached_generation_seconds += info["generation_seconds"]
                else:
                    part_started = time.perf_counter()
                    target = generate_sd_latents(bundle, bundle.unet, grid, prompts, config["seed"],
                        config["model"]["guidance_scale"], noise=noise[start:start+batch_size].to(bundle.device)).cpu()
                    seconds = time.perf_counter()-part_started
                    if self.seeds_per_prompt == 2:
                        chunks.mkdir(parents=True, exist_ok=True)
                        temporary_chunk = chunk.with_suffix(".tmp")
                        save_file({"latents": target.contiguous()}, str(temporary_chunk))
                        temporary_chunk.replace(chunk)
                        atomic_json(chunk_info, {"protocol_hash": fingerprint, "sha256": sha256(chunk),
                                                "generation_seconds": seconds})
                        cached_generation_seconds += seconds
                targets.append(target)
            self.tensors = {"noise": noise, "latents": torch.cat(targets)}
            if not torch.isfinite(self.tensors["latents"]).all():
                raise FloatingPointError("Nonfinite cached teacher latents")
            temporary = path.with_suffix(".tmp")
            save_file(self.tensors, str(temporary))
            temporary.replace(path)
            self.metadata = {"protocol": protocol, "protocol_hash": fingerprint,
                             "weights_sha256": sha256(path), "teacher_pair_count": count,
                             "teacher_pair_generation_seconds": cached_generation_seconds if self.seeds_per_prompt == 2 else time.perf_counter()-started,
                             "teacher_pair_nfe": count*len(grid)*2}
            atomic_json(metadata_path, self.metadata)

    def batch(self, microstep, device):
        if getattr(self, "seeds_per_prompt", 1) == 2:
            group_config = {**self.config, "data": {**self.config["data"],
                "batch_size": self.config["data"]["batch_size"]//2}}
            groups = training_indices(group_config, len(self.rows)//2, microstep, self, seed_offset=927901)
            indices = (groups[:, None]*2+torch.arange(2)).flatten()
        else:
            indices = training_indices(self.config, len(self.rows), microstep, self, seed_offset=927901)
        return (self.tensors["noise"][indices].to(device), self.tensors["latents"][indices].to(device),
                [self.rows[i]["caption"] for i in indices.tolist()])


class ContinuationFlowPairs(TeacherPairs):
    """Fresh endpoint couplings from the parent's immutable reflow teacher.

    Continuing endpoint distillation needs no new raw-SD pair generation and does
    not restart the already completed uniform-time reflow optimization phase.
    """

    def __init__(self, config, bundle, student, grid, output, records):
        from .sd_models import using_adapter

        self.config, self.rows = config, records
        output = Path(output)
        frozen = output/"reflow_adapter.safetensors"
        protocol = {"kind": "continuation-frozen-reflow-endpoints-v1",
            "adapter_sha256": sha256(frozen), "prompt_hash": manifest_hash(records),
            "count": len(records), "seed": config["seed"], "timesteps": grid.cpu().tolist(),
            "precision": config["precision"], "guidance_scale": 1,
            "prediction_type": config["train"]["instaflow_prediction_type"]}
        path, metadata_path = output/"reflow_pairs.safetensors", output/"reflow_pairs.json"
        if path.exists() and metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text())
            if self.metadata["protocol"] != protocol or self.metadata["weights_sha256"] != sha256(path):
                raise ValueError("Continuation reflow cache differs from its immutable teacher/prompt pool")
            self.tensors = load_file(str(path))
            return
        started = time.perf_counter()
        noise = torch.randn((len(records), 4, 64, 64),
            generator=torch.Generator().manual_seed(config["seed"]*1000003+714623))
        targets, batch = [], config["eval"]["batch_size"]
        with using_adapter(student, load_file(str(frozen), device=str(bundle.device))):
            for start in tqdm(range(0, len(records), batch), desc="Fresh reflow endpoint pairs",
                              disable=not config.get("progress_bar", True)):
                prompts = [row["caption"] for row in records[start:start+batch]]
                targets.append(generate_sd_latents(bundle, student, grid, prompts, config["seed"],
                    1, noise=noise[start:start+batch].to(bundle.device)).cpu())
        self.tensors = {"noise": noise, "latents": torch.cat(targets)}
        if not torch.isfinite(self.tensors["latents"]).all():
            raise FloatingPointError("Nonfinite continuation reflow endpoints")
        temporary = path.with_suffix(".tmp")
        save_file(self.tensors, str(temporary))
        temporary.replace(path)
        self.metadata = {"protocol": protocol, "weights_sha256": sha256(path),
            "teacher_pair_count": len(records), "teacher_pair_nfe": 0,
            "reflow_pair_nfe": len(records)*len(grid),
            "reflow_pair_generation_seconds": time.perf_counter()-started}
        atomic_json(metadata_path, self.metadata)


def reflow_pairs(config, bundle, student, ema, grid, output, initial):
    """Freeze phase-one EMA once; regenerate its coupling with multi-step Euler.

    The immutable adapter and cache make a resume after the phase boundary exact.
    This is a small LoRA adaptation of InstaFlow, not the released full-model recipe.
    """
    from .sd_models import using_adapter

    output = Path(output)
    frozen = output/"reflow_adapter.safetensors"
    if not frozen.exists():
        temporary = frozen.with_suffix(".tmp")
        save_file({k: v.cpu().contiguous() for k, v in ema.items()}, str(temporary))
        temporary.replace(frozen)
    protocol = {"initial_protocol_hash": initial.metadata["protocol_hash"],
                "adapter_sha256": sha256(frozen), "flow_steps": len(grid),
                "time_mapping": "nearest-VP-SNR" if config["train"].get("instaflow_prediction_type") == "epsilon-converted-velocity" else "round(999*(1-progress))", "guidance_scale": 1}
    path, metadata_path = output/"reflow_pairs.safetensors", output/"reflow_pairs.json"
    if path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["protocol"] != protocol or metadata["weights_sha256"] != sha256(path):
            raise ValueError("Reflow pair cache differs from its frozen flow teacher")
        tensors = load_file(str(path))
    else:
        started = time.perf_counter()
        targets = []
        batch = config["eval"]["batch_size"]
        with using_adapter(student, load_file(str(frozen), device=str(bundle.device))):
            for start in tqdm(range(0, len(initial.rows), batch), desc="Reflow teacher pairs",
                              disable=not config.get("progress_bar", True)):
                prompts = [r["caption"] for r in initial.rows[start:start+batch]]
                targets.append(generate_sd_latents(bundle, student, grid, prompts, config["seed"],
                    1, noise=initial.tensors["noise"][start:start+batch].to(bundle.device)).cpu())
        tensors = {"noise": initial.tensors["noise"], "latents": torch.cat(targets)}
        if not torch.isfinite(tensors["latents"]).all():
            raise FloatingPointError("Nonfinite reflow teacher latents")
        temporary = path.with_suffix(".tmp")
        save_file(tensors, str(temporary))
        temporary.replace(path)
        metadata = {"protocol": protocol, "weights_sha256": sha256(path),
                    "reflow_pair_nfe": len(initial.rows)*len(grid),
                    "reflow_pair_generation_seconds": time.perf_counter()-started}
        atomic_json(metadata_path, metadata)
    # Keep the exact prompt/noise minibatch protocol from the initial coupling.
    import copy
    pairs = copy.copy(initial)
    pairs.tensors, pairs.metadata = tensors, metadata
    return pairs
