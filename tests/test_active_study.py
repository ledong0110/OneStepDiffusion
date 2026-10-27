import json
from pathlib import Path
import runpy

import pytest


def test_active_snapshot_tracks_adopted_preprocessing_and_selected_training(tmp_path):
    module = runpy.run_path("scripts/monitor_active_study.py")
    snapshot = module["snapshot"]
    run = tmp_path/"instaflow"
    run.mkdir()
    owner = tmp_path/"owner.json"
    owner.write_text(json.dumps({"workers": {"3": {"method": "instaflow", "status": "running",
        "phase": "continue_one_step_instaflow", "child_pid": 123}}}))
    job = {"method": "instaflow", "run_dir": str(run), "result_dir": str(tmp_path/"results"),
           "gpu": 3, "owner_status": str(owner), "owner_key": "3"}
    status = snapshot(job)
    assert status["status"] == "running"
    assert status["phase"] == "preparing_fresh_reflow_pairs"
    assert status["cumulative_updates"] == 18000
    (run/"train.jsonl").write_text(json.dumps({"global_step": 50})+"\n")
    assert snapshot(job)["cumulative_updates"] == 18050
    with pytest.raises(ValueError, match="outside the active study"):
        snapshot({**job, "method": "edge_weight_high"})


def test_active_scope_excludes_edges_and_parallel_worker_rejects_edge_config():
    scope = json.loads(Path("configs/active_study.json").read_text())
    assert len(scope["jobs"]) == 9
    assert {job["method"] for job in scope["jobs"]}.isdisjoint(scope["excluded_methods"])
    worker = runpy.run_path("scripts/run_selected_experiment.py")
    assert "proposed" not in worker["ALLOWED"]
    assert "baseline" in worker["ALLOWED"]
    assert all(method in worker["ALLOWED"] for method in ("dmd_grouped", "dmd_perceptual",
                                                         "dmd_diversity", "dmd_perceptual_diversity"))
