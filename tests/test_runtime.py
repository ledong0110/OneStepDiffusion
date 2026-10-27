from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from few_step_diffusion.runtime import atomic_json, experiment_lock


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
