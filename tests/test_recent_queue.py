"""Resource guards, process incarnation checks and final-result gates."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
spec = importlib.util.spec_from_file_location("recent_queue", ROOT/"scripts/launch_recent_baselines.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def test_resource_gate_and_gpu_scope():
    plan = {"gpus": [0, 1, 2], "minimum_free_gpu_gib": 64, "minimum_free_disk_gib": 70}
    assert queue.eligible_gpus(plan, {0: 70, 1: 12, 2: 90, 3: 140}, {2}, 80) == [0]
    assert queue.eligible_gpus(plan, {0: 140}, set(), 11) == []
    assert queue.eligible_gpus(plan, {0: 63}, set(), 100) == []


def test_surviving_child_blocks_duplicate_worker(monkeypatch):
    monkeypatch.setattr(queue, "process_matches", lambda r: r["token"] == "current")
    assert queue.live_worker({"worker": {"token": "reused-pid"}, "child": {"token": "current"}})
    assert not queue.live_worker({"worker": {"token": "reused-pid"}})
    assert not queue.live_worker({})


def test_plan_refuses_changed_code_or_config(monkeypatch, tmp_path):
    from few_step_diffusion.runtime import sha256
    from few_step_diffusion.config import config_fingerprint, load_config
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    path = tmp_path/"code.py"
    path.write_text("x = 1\n")
    config = load_config(ROOT/"configs/sd15_dmd2_36k.yaml")
    monkeypatch.setattr(queue, "load_config", lambda p: config)
    plan = {"source_hashes": {"code.py": sha256(path)},
            "jobs": [{"config": "config.yaml", "config_hash": config_fingerprint(config)}]}
    queue.verify_plan(plan)
    config["train"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="config"):
        queue.verify_plan(plan)
    path.write_text("x = 2\n")
    with pytest.raises(ValueError, match="implementation"):
        queue.verify_plan(plan)


def test_report_omits_unfinished_jobs(tmp_path):
    from few_step_diffusion.recent_reporting import write_recent_results
    from few_step_diffusion.runtime import atomic_json
    out = tmp_path/"report"
    out.mkdir()
    assert write_recent_results(tmp_path, out) == []
    atomic_json(tmp_path/"configs/recent_baselines_36k.json", {"jobs": [{"method": "dmd2", "result_dir": "results"}]})
    assert write_recent_results(tmp_path, out) == []
    assert "tabular" not in (out/"recent_baselines.tex").read_text()


@pytest.mark.parametrize("key,value", [("num_samples", 8), ("nfe", 2),
    ("cumulative_training_updates", 3600), ("auxiliary_updates", 36000),
    ("teacher_guidance_scale", 7.5), ("source_commit", "unknown"),
    ("auxiliary_parameterization", "lora"), ("gan_weight", 0.)])
def test_report_rejects_incomplete_or_relabelled_results(monkeypatch, tmp_path, key, value):
    import few_step_diffusion.recent_reporting as reporting
    from few_step_diffusion.config import config_fingerprint, load_config
    from few_step_diffusion.recent_objectives import SOURCE_COMMIT
    config = load_config(ROOT/"configs/sd15_dmd2_36k.yaml")
    result = {"method": "dmd2", "seed": 42, "status": "complete",
        "num_samples": 5000, "num_real": 5000, "reference_split": "validation",
        "sampling_steps": 1, "nfe": 1, "lora_rank": 32, "effective_batch_size": 16,
        "cumulative_training_updates": 36000, "training_updates": 36000,
        "auxiliary_updates": 324000, "discriminator_updates": 324000,
        "source_commit": SOURCE_COMMIT, "teacher_guidance_scale": 1.75,
        "guidance_scale": 1., "precision": "bf16", "student_update_frequency": 10,
        "gan_weight": .001, "training_uses_real_images": True,
        "auxiliary_parameterization": "full-unet", "config_hash": config_fingerprint(config)}
    bad = deepcopy(result)
    bad[key] = value
    monkeypatch.setattr(reporting, "load_config", lambda p: config)
    monkeypatch.setattr(reporting, "collect_results", lambda *a, **kw: ([bad], []))
    with pytest.raises(ValueError, match=key):
        reporting.validate_result(tmp_path, tmp_path/"result.json",
            {"method": "dmd2", "config": "config.yaml", "config_hash": config_fingerprint(config)})
