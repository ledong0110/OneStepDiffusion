"""Parallel launch must respect memory margins and surviving process ownership."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import launch_independent_job as job


def test_measured_memory_budget_keeps_cuda_margin():
    assert job.enough_memory(15.01, 13.5)
    assert job.enough_memory(14.5, 13.5)
    assert not job.enough_memory(14.49, 13.5)
    assert not job.enough_memory(15.01, 15)


def test_surviving_child_blocks_another_parallel_launch(monkeypatch):
    monkeypatch.setattr(job, "process_matches", lambda record: record["incarnation"] == "live")
    assert job.live({"worker": {"incarnation": "old"}, "child": {"incarnation": "live"}})
    assert job.live({"worker": {"incarnation": "live"}})
    assert not job.live({"worker": {"incarnation": "old"}})
    assert not job.live({})
