import importlib.util
from pathlib import Path
import shutil
import sys

import pytest

from few_step_diffusion.config import load_config


@pytest.mark.parametrize("preset,total_updates,count,batch", [("fair_pilot", 600, 128, 4),
                                                             ("scale50k", 18000, 5000, 16)])
def test_fair_pilot_matches_all_student_steps_update_budgets_and_data(tmp_path, monkeypatch, preset, total_updates, count, batch):
    path = Path("scripts/run_experiments.py").resolve()
    spec = importlib.util.spec_from_file_location("experiment_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    shutil.copytree("configs", tmp_path/"configs")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(sys, "argv", [str(path), "--preset", preset, "--ablations", "--gpu", "3"])
    training, evaluation = [], []

    def capture(command, **kwargs):
        config = load_config(command[command.index("--config")+1])
        if "train" in command:
            training.append(config)
        else:
            evaluation.append((config, int(command[command.index("--steps")+1])))

    monkeypatch.setattr(runner.subprocess, "run", capture)
    runner.main()
    assert len(training) == 9
    assert {c["name"] for c in training} == {"baseline", "proposed", "edge_constant", "swiftbrush_vsd",
                                            "consistency", "dmd", "instaflow", "edge_weight_low", "edge_weight_high"}
    budgets = {c["train"]["updates_per_stage"]*len(c["train"]["stages"]) for c in training}
    assert budgets == {total_updates}
    assert all(c["train"]["stages"][-1] == 1 for c in training)
    assert {c["seed"] for c in training} == {42}
    assert {c["data"]["batch_size"]*c["train"]["grad_accumulation"] for c in training} == {batch}
    assert {c["eval"]["num_samples"] for c in training} == {count}
    if preset == "scale50k":
        assert {c["data"]["train_limit"] for c in training} == {50000}
        flow = next(c for c in training if c["name"] == "instaflow")
        assert flow["train"]["instaflow_pair_count"] == 50000
        assert flow["train"]["instaflow_prediction_type"] == "epsilon-converted-velocity"
    shared = {(c["model"]["source"], c["model"]["revision"], c["model"]["lora_rank"],
               c["data"]["root"], c["data"]["train_limit"], c["eval"]["num_samples"],
               c["eval"]["num_real"], c["eval"]["seed"], c["precision"]) for c in training}
    assert len(shared) == 1
    students = [(c, steps) for c, steps in evaluation if "teacher_seed" not in c["output_dir"]]
    teachers = [steps for c, steps in evaluation if "teacher_seed" in c["output_dir"]]
    assert len(students) == 9 and {steps for _, steps in students} == {1}
    assert teachers == [64, 1]
    weights = {c["name"]: c["train"]["edge_weight"] for c in training}
    assert weights["edge_weight_low"] == .03 and weights["edge_weight_high"] == .3
