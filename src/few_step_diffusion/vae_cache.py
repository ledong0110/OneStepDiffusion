"""Frozen FP32 VAE posterior statistics; posterior noise stays fresh per update."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .sd_data import caption_records, manifest_hash, pixels


def cache_protocol(config):
    full = deepcopy(config)
    full["data"].update(train_limit=118287, train_offset=0)
    records = caption_records(full)
    return {"policy": "coco-fp32-vae-posterior-v1", "model_source": config["model"]["source"],
            "model_revision": config["model"]["revision"], "training_manifest_hash": manifest_hash(records),
            "pixel_transform_sha256": hashlib.sha256(inspect.getsource(pixels).encode()).hexdigest(),
            "torch_version": torch.__version__, "dtype": "float32", "image_size": 512,
            "count": len(records), "shape": [len(records), 8, 64, 64]}, records


@dataclass
class PosteriorBatch:
    mean: torch.Tensor
    std: torch.Tensor


class PosteriorCache:
    def __init__(self, folder, config):
        self.folder = Path(folder)
        metadata = json.loads((self.folder/"metadata.json").read_text())
        expected, records = cache_protocol(config)
        if metadata.get("status") != "complete" or metadata.get("protocol") != expected:
            raise ValueError("VAE posterior cache is incomplete or belongs to another model/data transform")
        ids = np.load(self.folder/"image_ids.npy", allow_pickle=False)
        if ids.tolist() != [r["image_id"] for r in records]:
            raise ValueError("VAE cache image order differs from the declared training pool")
        self.indices = {int(value): i for i, value in enumerate(ids)}
        self.stats = np.load(self.folder/"posterior.npy", mmap_mode="r", allow_pickle=False)
        if list(self.stats.shape) != expected["shape"] or self.stats.dtype != np.dtype("float32"):
            raise ValueError("VAE posterior cache has the wrong shape or precision")
        self.protocol_identity = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()

    def batch(self, rows):
        values = torch.from_numpy(np.array(self.stats[[self.indices[r["image_id"]] for r in rows]], copy=True))
        if not torch.isfinite(values).all() or (values[:, 4:] < 0).any():
            raise ValueError("Invalid cached VAE posterior statistics")
        return PosteriorBatch(values[:, :4], values[:, 4:])
