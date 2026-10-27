import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import uuid
from pathlib import Path

import numpy as np
import torch


@contextlib.contextmanager
def experiment_lock(output):
    """Serialize train/evaluate owners of one output folder across processes."""
    import fcntl
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/".experiment.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Run already active; waiting for its training/evaluation: {output}", flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "output_dir": str(output.resolve())})+"\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # Avoid excessive thread contention on large shared servers.
    torch.set_num_threads(int(os.environ.get("FSD_CPU_THREADS", "4")))


def device_for(config):
    device = torch.device(config["device"])
    if device.type == "cpu" and config["precision"] != "fp32":
        raise ValueError("CPU runs require --set precision=fp32")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. For CPU use --set device=cpu --set precision=fp32")
    if device.type == "cuda" and config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16; use --set precision=fp32")
    return device


def autocast(config, device):
    if config["precision"] == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def provenance(config, device):
    packages = {}
    for name in ("torch", "torchvision", "diffusers", "numpy", "torch-fidelity", "transformers", "peft"):
        packages[name] = importlib.metadata.version(name)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    return {"packages": packages, "torch_cuda": torch.version.cuda, "git_commit": commit,
            "git_dirty": dirty, "device": str(device), "precision": config["precision"],
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_threads": torch.get_num_threads(), "tf32": False}
