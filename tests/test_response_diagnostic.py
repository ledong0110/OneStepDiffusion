import copy
import json
from pathlib import Path

import pytest
import yaml

from few_step_diffusion.config import config_fingerprint, load_config
from few_step_diffusion.runtime import sha256


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import analyze_teacher_responses as diagnostic
    from few_step_diffusion import reporting
    study, key = "dmd_response_strength", "angular_30"
    run = tmp_path/"runs"/key
    (run/"stage_1").mkdir(parents=True)
    (tmp_path/"configs").mkdir()
    (tmp_path/"results").mkdir()
    adapter = run/"stage_1/adapter.safetensors"
    adapter.write_bytes(b"unit-test model identity only")
    config = load_config("configs/dmd_perceptual.yaml")
    config.update(name="dmd_response_angular_30", output_dir=str(run))
    config["train"].update(updates_per_stage=500, dmd_response_mode="angular",
        dmd_response_variant=key, dmd_response_weight=.18, dmd_response_gradient_fraction=.3)
    (tmp_path/"configs/arm.yaml").write_text(yaml.safe_dump(config))
    job = {"key": key, "method": config["name"], "config": "configs/arm.yaml",
           "config_hash": config_fingerprint(config), "run_dir": f"runs/{key}", "result_dir": "results"}
    plan = {"jobs": [job], "seed": 42, "additional_updates": 500,
            "cumulative_training_updates": 36500, "parent_checkpoint_sha256": "c"*64}
    (tmp_path/"configs"/f"{study}.json").write_text(json.dumps(plan))
    finished = {"status": "complete", "global_step": 500, "config": config}
    (run/"run.json").write_text(json.dumps(finished))
    result = {"method": config["name"], "config_hash": job["config_hash"], "num_samples": 5000,
              "num_real": 5000, "cumulative_training_updates": 36500,
              "model_identity": sha256(adapter), "refinement_parent_checkpoint_sha256": "c"*64}
    result_file = tmp_path/"results"/f"{config['name']}_seed42_1steps.result.json"
    result_file.write_text("{}")
    # Full schema/loss validation has its own importer tests. Here test binding
    # the validated result to the completed run and exact adapter file.
    def collect(folder, schema, *, result_files):
        assert result_files == [result_file]
        return [copy.deepcopy(result)], []
    monkeypatch.setattr(reporting, "collect_results", collect)
    return diagnostic, tmp_path, study, key, run, result


def test_completed_diagnostic_uses_exact_evaluated_export(source_fixture):
    diagnostic, root, study, key, run, _ = source_fixture
    actual_run, config, plan = diagnostic.completed_source(root, study, key)
    assert actual_run == run and config["name"] == plan["jobs"][0]["method"]
    (run/"stage_1/adapter.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="differs from the completed full evaluation"):
        diagnostic.completed_source(root, study, key)


@pytest.mark.parametrize("field,value", [("num_samples", 1000), ("num_real", 1000),
    ("cumulative_training_updates", 36000), ("config_hash", "x"*64),
    ("refinement_parent_checkpoint_sha256", "x"*64)])
def test_diagnostic_rejects_partial_or_mismatched_results(source_fixture, field, value):
    diagnostic, root, study, key, _, result = source_fixture
    result[field] = value
    with pytest.raises(ValueError, match="differs from the completed full evaluation"):
        diagnostic.completed_source(root, study, key)


def test_diagnostic_refuses_live_or_missing_training(source_fixture):
    diagnostic, root, study, key, run, _ = source_fixture
    completed = json.loads((run/"run.json").read_text())
    completed["global_step"] = 499
    (run/"run.json").write_text(json.dumps(completed))
    with pytest.raises(ValueError, match="completed scheduled training"):
        diagnostic.completed_source(root, study, key)
    (run/"run.json").unlink()
    with pytest.raises(ValueError, match="waits for completed"):
        diagnostic.completed_source(root, study, key)
