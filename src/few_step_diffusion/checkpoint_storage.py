"""Optional machine-local storage for new runs' checkpoints only."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def storage_root():
    configured = os.environ.get("FSD_CHECKPOINT_ROOT")
    local = ROOT/".checkpoint-storage.json"
    if configured is None and local.exists():
        configured = json.loads(local.read_text())["checkpoint_root"]
    return Path(configured).expanduser().absolute() if configured else None


def prepare_output_storage(output):
    """Configure future checkpoint writes; never relocate an existing run."""
    root = storage_root()
    output = Path(output).absolute()
    if root is None or (output/"initialization.json").exists() or (output/"last.pt").exists():
        return
    target = root/output.relative_to(ROOT)
    target.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("last.pt", "checkpoints", "stage_1"):
        path, destination = output/name, target/name
        if path.is_symlink():
            if path.resolve() != destination.resolve():
                raise ValueError(f"Checkpoint link has a different destination: {path}")
            continue
        if path.exists():
            continue
        if name != "last.pt":
            destination.mkdir(parents=True, exist_ok=True)
        path.symlink_to(destination, target_is_directory=name != "last.pt")


def save_destination(path):
    """Resolve a checkpoint link before atomic replace, preserving the alias."""
    path = Path(path)
    destination = path.resolve() if path.is_symlink() else path
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def checkpoint_free_gib(fallback=ROOT):
    import shutil
    root = storage_root()
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(root or fallback).free/2**30
