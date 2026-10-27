"""Pinned CIFAR-10/CelebA-HQ acquisition and reproducible minibatches."""

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm
from .config import dataset_name, image_size


def dataset(config, split="train"):
    if config["data"]["fake"]:
        g = torch.Generator().manual_seed(987 if split == "train" else 988)
        size = image_size(config)
        return torch.randint(0, 256, (64, 3, size, size), generator=g, dtype=torch.uint8)
    if dataset_name(config) == "CelebA-HQ":
        return celeb_dataset(config, split)
    root = Path(config["data"]["root"])
    # Reuse a conventional torchvision download if one is already available.
    if (root / "cifar-10-batches-py").exists():
        ds = CIFAR10(root, train=split == "train", download=False)
        return torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    cache = root / f"cifar10_{split}.npy"
    if not cache.exists():
        if not config["data"]["download"]:
            raise FileNotFoundError(f"Dataset missing: {cache}; run fsd prepare")
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        # The official UofT mirror avoids the heavily throttled Toronto archive.
        parquet = hf_hub_download(
            repo_id="uoft-cs/cifar10", repo_type="dataset",
            revision="0b2714987fa478483af9968de7c934580d0bb9a2",
            filename=f"plain_text/{split}-00000-of-00001.parquet",
        )
        table = pq.read_table(parquet, columns=["img"])
        images = np.stack([np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB"))
                           for row in table["img"].to_pylist()])
        expected = 50000 if split == "train" else 10000
        if images.shape != (expected, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError("Unexpected CIFAR-10 mirror contents")
        root.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        with temporary.open("wb") as f:
            np.save(f, images, allow_pickle=False)
        temporary.replace(cache)
    return torch.from_numpy(np.load(cache, allow_pickle=False)).permute(0, 3, 1, 2).contiguous()


def celeb_dataset(config, split):
    if split not in ("train", "validation"):
        raise ValueError("The pinned CelebA-HQ mirror provides train and validation splits")
    root = Path(config["data"]["root"])
    revision = config["data"]["revision"]
    cache = root / f"celebahq_{split}_{revision[:12]}.npy"
    expected = 28000 if split == "train" else 2000
    size = image_size(config)
    if not cache.exists():
        if not config["data"]["download"]:
            raise FileNotFoundError(f"Dataset missing: {cache}; run fsd prepare")
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        root.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        images = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
                                           shape=(expected, size, size, 3))
        count, shards = 0, 6 if split == "train" else 1
        with tqdm(total=expected, desc=f"Decode CelebA-HQ {split}", unit="image",
                  disable=not config.get("progress_bar", True)) as progress:
            for shard in range(shards):
                parquet = hf_hub_download(repo_id=config["data"]["repo"], repo_type="dataset",
                    revision=revision, filename=f"data/{split}-{shard:05d}-of-{shards:05d}.parquet")
                for batch in pq.ParquetFile(parquet).iter_batches(batch_size=64, columns=["image"]):
                    for row in batch.column(0).to_pylist():
                        with Image.open(io.BytesIO(row["bytes"])) as image:
                            if image.size != (size, size) or count >= expected:
                                raise ValueError("Unexpected CelebA-HQ mirror image shape or row count")
                            images[count] = np.asarray(image.convert("RGB"))
                        count += 1
                    progress.update(len(batch))
        if count != expected:
            raise ValueError(f"Expected {expected} CelebA-HQ images, got {count}")
        images.flush()
        del images
        temporary.replace(cache)
    # Copy-on-write mmap avoids loading the entire 5.5 GB train tensor into RAM.
    images = np.load(cache, mmap_mode="c", allow_pickle=False)
    if images.shape != (expected, size, size, 3) or images.dtype != np.uint8:
        raise ValueError("Invalid cached CelebA-HQ dataset")
    return torch.from_numpy(images).permute(0, 3, 1, 2)


class BatchStream:
    """Uniform sampling with replacement, indexed by optimizer microstep.

    Data order and flips depend on (seed, microstep), so resuming does not require
    replaying a DataLoader. Only the official training split is used here.
    """

    def __init__(self, config):
        self.images = dataset(config, "train")
        self.batch_size = config["data"]["batch_size"]
        self.seed = config["seed"]

    def batch(self, microstep, device):
        g = torch.Generator().manual_seed(self.seed * 1000003 + microstep)
        indices = torch.randint(len(self.images), (self.batch_size,), generator=g)
        x = self.images[indices].float() / 127.5 - 1
        flip = torch.rand(self.batch_size, generator=g) < 0.5
        x[flip] = x[flip].flip(-1)
        return x.to(device)


def prepare(config, metrics=False):
    from .config import is_stable_diffusion
    if is_stable_diffusion(config):
        from .sd_data import prepare_sd
        return prepare_sd(config, metrics)
    from .models import load_model

    for split in (("train", "validation") if dataset_name(config) == "CelebA-HQ" else ("train", "test")):
        images = dataset(config, split)
        print(f"{split}: {len(images)} images, shape={tuple(images.shape[1:])}", flush=True)
    model, scheduler = load_model(config)
    print(f"Teacher ready: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    if metrics:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

        FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"])
        print("FID / Inception Score weights cached", flush=True)
    Path(config["data"]["root"]).mkdir(parents=True, exist_ok=True)
    print(f"Noise scheduler: {scheduler.config.num_train_timesteps} timesteps", flush=True)
