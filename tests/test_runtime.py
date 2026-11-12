from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from few_step_diffusion.runtime import atomic_json, experiment_lock


@pytest.mark.parametrize(("configured_device", "allocation_device"), [("cuda", 0), ("cuda:2", 2)])
def test_gpu_memory_cap_requires_free_margin_and_records_allocator_limit(monkeypatch, configured_device, allocation_device):
    from few_step_diffusion.runtime import device_for
    monkeypatch.setenv("FSD_MAX_GPU_MEMORY_GIB", "15")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (18*2**30, 140*2**30))
    calls = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda limit, device: calls.append((limit, device)))
    assert device_for({"device": configured_device, "precision": "fp32"}).type == "cuda"
    assert calls == [(15/140, allocation_device)]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (15*2**30, 140*2**30))
    with pytest.raises(RuntimeError, match="allocation cap"):
        device_for({"device": configured_device, "precision": "fp32"})
    assert len(calls) == 1


def test_cpu_does_not_apply_cuda_allocation_cap(monkeypatch):
    from few_step_diffusion.runtime import device_for
    monkeypatch.setenv("FSD_MAX_GPU_MEMORY_GIB", "15")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: pytest.fail("CPU must not query GPU memory"))
    assert device_for({"device": "cpu", "precision": "fp32"}).type == "cpu"


def test_simultaneous_caption_manifest_writes_are_atomic(tmp_path):
    path = tmp_path/"shared_caption_manifest.json"
    values = [{"writer": i, "records": list(range(1000))} for i in range(32)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: atomic_json(path, value), values))
    assert json.loads(path.read_text()) in values
    assert list(tmp_path.iterdir()) == [path]


def test_experiment_lock_serializes_independent_workers_and_releases(tmp_path):
    script = ("import sys\nfrom few_step_diffusion.runtime import experiment_lock\n"
              "with experiment_lock(sys.argv[1]):\n    print('acquired', flush=True)\n")
    child = None
    try:
        with experiment_lock(tmp_path):
            child = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert child.stdout.readline().startswith("Run already active;")
            assert child.poll() is None
        remaining, _ = child.communicate(timeout=20)
        assert child.returncode == 0 and "acquired" in remaining
        with experiment_lock(tmp_path):
            assert Path(tmp_path/".experiment.lock").exists()
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
