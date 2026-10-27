"""Official COCO captions and a deterministic image subset; no full train archive needed."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageOps
import requests
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .runtime import atomic_json, sha256

# HTTPS virtual-hosted S3 endpoint for the official images.cocodataset.org bucket.
ORIGIN = "https://s3.amazonaws.com/images.cocodataset.org"
ANNOTATIONS = "annotations/annotations_trainval2017.zip"


class RemoteZip(io.RawIOBase):
    """Seekable HTTP range reader: fetch caption entries instead of the 241 MB archive."""

    def __init__(self, url):
        self.url, self.position = url, 0
        response = requests.head(url, timeout=(20, 60))
        response.raise_for_status()
        self.length = int(response.headers["Content-Length"])

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.length)+offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        end = self.length-1 if size < 0 else min(self.length-1, self.position+size-1)
        if end < self.position:
            return b""
        start = self.position
        response = requests.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 180))
        response.raise_for_status()
        if (response.status_code != 206 or
                response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.length}" or
                len(response.content) != end-start+1):
            raise IOError("COCO server did not honor a verified byte-range request")
        self.position += len(response.content)
        return response.content


def download(url, path):
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    last_error = None
    for _ in range(3):
        try:
            with requests.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                expected = int(response.headers.get("Content-Length", 0))
                count = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024*1024):
                        handle.write(chunk)
                        count += len(chunk)
                if expected and count != expected:
                    raise IOError(f"Incomplete download: {url}")
            temporary.replace(path)
            return path
        except (requests.RequestException, OSError) as exc:
            last_error = exc
    raise RuntimeError(f"Download failed after three attempts: {url}") from last_error


def caption_records(config, split="train", limit=None):
    if split not in ("train", "validation"):
        raise ValueError("COCO caption protocol uses train and validation")
    root = Path(config["data"]["root"])
    official_split = "train2017" if split == "train" else "val2017"
    annotations = root / "annotations" / f"captions_{official_split}.json"
    if not annotations.exists():
        if not config["data"]["download"]:
            raise FileNotFoundError(f"Run fsd prepare to download {annotations}")
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        archive = root / ANNOTATIONS
        source = archive if archive.exists() else RemoteZip(f"{ORIGIN}/{ANNOTATIONS}")
        # Extract named caption files only; never trust arbitrary archive paths.
        with zipfile.ZipFile(source) as zipped:
            for name in ("captions_train2017.json", "captions_val2017.json"):
                destination = root / "annotations" / name
                temporary = destination.with_suffix(".tmp")
                temporary.write_bytes(zipped.read("annotations/" + name))
                temporary.replace(destination)
        atomic_json(root / "annotations/provenance.json", {
            "url": f"{ORIGIN}/{ANNOTATIONS}",
            "caption_sha256": {name: sha256(root/"annotations"/name) for name in
                               ("captions_train2017.json", "captions_val2017.json")},
            "caption_policy": "smallest annotation ID per image; ascending image ID",
        })
    contents = json.loads(annotations.read_text())
    captions = {}
    for item in sorted(contents["annotations"], key=lambda item: item["id"]):
        captions.setdefault(item["image_id"], (item["id"], item["caption"]))
    records = [{"image_id": item["id"], "file_name": item["file_name"],
                "caption_id": captions[item["id"]][0], "caption": captions[item["id"]][1],
                "split": official_split}
               for item in sorted(contents["images"], key=lambda item: item["id"])]
    if limit is None:
        limit = config["data"]["train_limit"] if split == "train" else 5000
    offset = config["data"].get("train_offset", 0) if split == "train" else 0
    if not isinstance(offset, int) or offset < 0 or not 1 <= limit <= len(records)-offset:
        raise ValueError(f"Requested {limit} captions from a split of size {len(records)}")
    records = records[offset:offset+limit]
    manifest = {"policy": "coco2017-first-caption-by-id-v1", "records": records}
    suffix = f"_offset{offset}" if offset else ""
    atomic_json(root / f"{official_split}_{limit}{suffix}.manifest.json", manifest)
    return records


def manifest_hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def image_path(config, record):
    path = Path(config["data"]["root"]) / record["split"] / record["file_name"]
    if not path.exists():
        if not config["data"]["download"]:
            raise FileNotFoundError(f"Missing COCO image: {path}")
        download(f"{ORIGIN}/{record['split']}/{record['file_name']}", path)
    return path


def prepare_images(config, records):
    with ThreadPoolExecutor(max_workers=config["data"].get("download_workers", 12)) as pool:
        paths = pool.map(lambda row: image_path(config, row), records)
        for _ in tqdm(paths, total=len(records), desc="Cache COCO images", unit="image",
                      disable=not config.get("progress_bar", True)):
            pass


def pixels(config, record):
    with Image.open(image_path(config, record)) as image:
        image = ImageOps.fit(image.convert("RGB"), (512, 512), method=Image.Resampling.BICUBIC)
        return torch.from_numpy(np.array(image)).permute(2, 0, 1)


class COCOReferences(Dataset):
    def __init__(self, config, count):
        self.config = config
        self.records = caption_records(config, "validation", count)
        prepare_images(config, self.records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return pixels(self.config, self.records[index])


class CaptionStream:
    """Microstep-indexed selection gives reproducible prompts and image pairs on resume.

    Captions can specify left/right, so image flips are deliberately disabled.
    VSD never downloads or reads the training images.
    """

    def __init__(self, config, images=True):
        self.config = config
        self.records = caption_records(config)
        self.images = images
        self._epoch, self._order = None, None
        if images:
            prepare_images(config, self.records)

    def batch(self, microstep, device):
        indices = self.indices(microstep)
        rows = [self.records[index] for index in indices.tolist()]
        images = (torch.stack([pixels(self.config, row) for row in rows]).to(device).float()/127.5-1
                  if self.images else None)
        return images, [row["caption"] for row in rows]

    def indices(self, microstep):
        return training_indices(self.config, len(self.records), microstep, self)


def training_indices(config, count, microstep, cache=None, seed_offset=0):
    """Default sampling stays unchanged; continuation traverses shuffled full epochs."""
    batch = config["data"]["batch_size"]
    seed = config["seed"]*1000003+seed_offset
    if config["data"].get("sampling", "replacement") == "replacement":
        generator = torch.Generator().manual_seed(seed+microstep)
        return torch.randint(count, (batch,), generator=generator)
    indices = []
    for position in range(microstep*batch, (microstep+1)*batch):
        epoch, index = divmod(position, count)
        if cache is None or getattr(cache, "_epoch", None) != epoch:
            order = torch.randperm(count, generator=torch.Generator().manual_seed(seed+epoch))
            if cache is not None:
                cache._epoch, cache._order = epoch, order
        else:
            order = cache._order
        indices.append(order[index].item())
    return torch.tensor(indices)


def prepare_sd(config, metrics=False):
    from .sd_models import SDBundle

    train_records = caption_records(config)
    if config["train"]["mode"] not in ("vsd", "dmd", "instaflow"):
        prepare_images(config, train_records)
    prepare_images(config, caption_records(config, "validation", config["eval"]["num_real"]))
    bundle = SDBundle(config, torch.device("cpu"))
    print(f"SD 1.5 ready: {sum(p.numel() for p in bundle.unet.parameters()):,} U-Net parameters")
    print(f"COCO: {len(train_records)} training captions, {config['eval']['num_real']} reference images")
    if metrics:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"])
        from .sd_evaluation import load_clip
        load_clip(torch.device("cpu"))
