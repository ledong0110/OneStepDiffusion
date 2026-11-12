import copy
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch
import yaml

from few_step_diffusion.config import config_fingerprint
from few_step_diffusion.dmd_conflict import MODES, POLICY
from few_step_diffusion import dmd_noise_agreement as noise
from few_step_diffusion.runtime import sha256
from test_active_report import result_fixture


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location("conflict_launcher", "scripts/launch_dmd_conflict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_identity_waits_for_exec_argv(launcher, monkeypatch):
    command = ["python", "train.py"]
    records = iter([{"pid": 2, "command": []}, {"pid": 2, "command": command}])
    monkeypatch.setattr(launcher, "process_record", lambda pid: next(records))
    monkeypatch.setattr(launcher.time, "sleep", lambda duration: None)
    assert launcher.started_process_record(SimpleNamespace(pid=2, poll=lambda: None), command)["command"] == command
    assert launcher.started_process_record(SimpleNamespace(pid=2, poll=lambda: 1), command) is None


def test_phase_memory_policy_preserves_training_budget_and_bounds_diagnostics(launcher):
    manifest = {"max_gpu_memory_gib": 15, "min_free_gpu_gib": 16}
    assert launcher.phase_memory_limits(manifest, "training") == (15, 16)
    assert launcher.phase_memory_limits(manifest, "full_5000_image_evaluation") == (15, 16)
    assert launcher.phase_memory_limits(manifest, "diversity_128_prompts_four_seeds") == (10, 11)
    assert launcher.phase_memory_limits(manifest, "matched_qualitative_samples") == (4, 5)
    assert launcher.phase_memory_limits(manifest, "teacher_response_diagnostic") == (8, 9)
    assert manifest == {"max_gpu_memory_gib": 15, "min_free_gpu_gib": 16}


def test_completed_pipeline_refreshes_without_acquiring_a_gpu(launcher, tmp_path, monkeypatch):
    import few_step_diffusion.reporting as reporting
    config = {"name": "synthetic completed response"}
    fingerprint = config_fingerprint(config)
    run = tmp_path/"runs/angular_50"
    (run/"checkpoint_previews/step_000500").mkdir(parents=True)
    (run/"checkpoint_previews/step_000500/samples.json").write_text("{}")
    (run/"run.json").write_text(json.dumps({"status": "complete", "config": config}))
    result_dir = tmp_path/"results"
    result_dir.mkdir()
    (result_dir/"dmd_response_angular_50_seed42_1steps.result.json").write_text("{}")
    runs = tmp_path/"study"
    runs.mkdir()
    plan = tmp_path/"plan.json"
    plan.write_text(json.dumps({"gpus": [0], "max_parallel": 1, "min_free_gpu_gib": 16,
        "max_gpu_memory_gib": 15, "additional_updates": 500,
        "jobs": [{"key": "angular_50", "method": "dmd_response_angular_50", "run_dir": "runs/angular_50",
                  "config": "unused.yaml", "config_hash": fingerprint, "result_dir": "results"}]}))
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "RUNS", runs)
    monkeypatch.setattr(launcher, "PLAN", plan)
    monkeypatch.setattr(launcher, "STUDY", "dmd_response_strength")
    monkeypatch.setattr(launcher, "load_config", lambda path: config)
    def forbidden_gpu_query():
        pytest.fail("A fully completed pipeline must not wait for GPU capacity")
    monkeypatch.setattr(launcher, "gpu_free", forbidden_gpu_query)
    # Result provenance/report correctness are covered by their importer tests;
    # this fixture exercises the scheduler's completed-work control flow.
    monkeypatch.setattr(reporting, "collect_results", lambda *args, **kwargs:
                        ([{"config_hash": fingerprint, "diversity_lpips_alex_mean": .7}], []))
    monkeypatch.setattr(launcher, "refresh_results", lambda *args: None)
    monkeypatch.setattr(launcher.subprocess, "call", lambda *args, **kwargs: 0)
    launcher.supervise()
    state = json.loads((runs/"status.json").read_text())
    assert state["status"] == "complete"
    assert state["jobs"]["angular_50"]["child"] is None


def test_final_response_design_requires_complete_pilot_and_pins_evidence(launcher, tmp_path, monkeypatch):
    import analyze_teacher_responses as diagnostic
    import build_assignment2_report as report
    (tmp_path/"configs").mkdir()
    folder = tmp_path/"runs/sd15/dmd_response_strength"
    folder.mkdir(parents=True)
    (tmp_path/"results").mkdir()
    jobs, results = [], []
    for index, fraction in enumerate((10, 30, 50)):
        key, method = f"angular_{fraction}", f"dmd_response_angular_{fraction}"
        jobs.append({"key": key, "method": method, "result_dir": "results"})
        (tmp_path/"results"/f"{method}_seed42_1steps.result.json").write_text(json.dumps({"synthetic": key}))
        results.append({"method": method, "fid": 29+index, "clip_score_mean": 31.,
                        "diversity_lpips_alex_mean": .7, "model_identity": str(index)*64})
    plan = {"seed": 42, "additional_updates": 500, "jobs": jobs}
    plan_path = tmp_path/"configs/dmd_response_strength.json"
    plan_path.write_text(json.dumps(plan))
    comparison = {"results": [], "fixed_reference": {"model_identity": "r"*64}}
    def refresh(root, study):
        (folder/"comparison.json").write_text(json.dumps(comparison))
    monkeypatch.setattr(launcher, "refresh_results", refresh)
    monkeypatch.setattr(diagnostic, "completed_source", lambda *args: None)
    # Source/checkpoint validation is covered separately. Exercise completion
    # gating, selection, matched prompts and evidence pinning here.
    sample = {"sample_prompts": ["test prompt"], "sample_image_ids": [1], "eval_seed": 12345,
              "evaluation_prompt_hash": "p"*64}
    preview = {"checkpoint_sha256": "s"*64, "records": [copy.deepcopy(sample), copy.deepcopy(sample)]}
    monkeypatch.setattr(report, "completed_preview", lambda *args, **kwargs: copy.deepcopy(preview))
    with pytest.raises(ValueError, match="full quality and diversity"):
        launcher.response_final_design(tmp_path, .3, "test rationale")
    comparison["results"] = results
    preview["records"][1]["sample_prompts"] = ["different prompt"]
    with pytest.raises(ValueError, match="matching prompts"):
        launcher.response_final_design(tmp_path, .3, "test rationale")
    preview["records"][1] = copy.deepcopy(sample)
    selected = launcher.response_final_design(tmp_path, .3, "test rationale")
    # Explicit selection is retained; the helper does not silently substitute
    # the lowest-FID synthetic arm or claim that it establishes novelty.
    assert selected["selected_pilot_method"] == "dmd_response_angular_30"
    assert selected["pilot_plan_sha256"] == sha256(plan_path)
    assert len(selected["pilot_evidence"]) == 3
    for item in selected["pilot_evidence"]:
        assert item["result_sha256"] == sha256(tmp_path/item["result_file"])
    results[1]["diversity_lpips_alex_mean"] = None
    with pytest.raises(ValueError, match="full quality and diversity"):
        launcher.response_final_design(tmp_path, .3, "test rationale")


def test_final_response_prepare_has_equal_budget_and_no_baseline_training(launcher, tmp_path, monkeypatch):
    from few_step_diffusion.config import load_config
    parent = tmp_path/"runs/sd15/dmd_diversity_study/dmd_perceptual_seed42"
    (parent/"stage_1").mkdir(parents=True)
    parent_config = load_config("configs/dmd_perceptual.yaml")
    parent_config["train"]["updates_per_stage"] = 18000
    bank = tmp_path/"teacher_pairs"
    bank.mkdir()
    (bank/"teacher_pairs.safetensors").write_bytes(b"synthetic teacher bank")
    (bank/"teacher_pairs.json").write_text(json.dumps({"weights_sha256": sha256(bank/"teacher_pairs.safetensors")}))
    parent_config["train"]["dmd_pair_cache_dir"] = str(bank)
    (parent/"config.yaml").write_text(yaml.safe_dump(parent_config))
    (parent/"run.json").write_text(json.dumps({"status": "complete", "config": parent_config}))
    (parent/"stage_1/metadata.json").write_text(json.dumps({"cumulative_training_updates": 36000}))
    (parent/"stage_1/adapter.safetensors").write_bytes(b"synthetic fixed model")
    (parent/"last.pt").write_bytes(b"synthetic full parent state")
    parent_hash = sha256(parent/"last.pt")
    reference = tmp_path/"results/sd15/dmd_diversity_study/dmd_perceptual_seed42_1steps.result.json"
    reference.parent.mkdir(parents=True)
    reference.write_text(json.dumps({"synthetic": "fixed reference"}))
    calibration = tmp_path/"artifacts/dmd_teacher_response/gradient_calibration.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text(json.dumps({"policy": "frozen-teacher-finite-response-diagnostic-v1",
        "parent_checkpoint_sha256": parent_hash, "training_only": True, "weights_unchanged": True,
        "optimizer_updates": 0, "gradient_calibration": {"coefficients": {
            key: {"weights": {"0.3": value}} for key,value in
            (("angular", .18), ("magnitude", .75), ("vector", .073))}}}))
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "STUDY", "dmd_response_final")
    monkeypatch.setattr(launcher, "RUNS", tmp_path/"runs/sd15/dmd_response_final")
    monkeypatch.setattr(launcher, "PLAN", tmp_path/"configs/dmd_response_final.json")
    monkeypatch.setattr(launcher, "response_final_design", lambda *args: {"gradient_fraction": .3, "reason": "test"})
    monkeypatch.chdir(tmp_path)
    plan = launcher.prepare([0, 1], 2, 16, updates=1200, response_fraction=.3, selection_reason="test")
    assert plan["additional_updates"] == 1200 and plan["cumulative_training_updates"] == 37200
    assert {job["key"] for job in plan["jobs"]} == {"angular_30", "magnitude_30", "vector_30"}
    for job in plan["jobs"]:
        config = load_config(tmp_path/job["config"])
        assert config["train"]["updates_per_stage"] == 1200
        assert config["train"]["dmd_regression_weight"] == parent_config["train"]["dmd_regression_weight"]
        assert config["train"]["dmd_response_gradient_fraction"] == .3
        assert config["dmd_refinement"]["parent_run"] == str(parent)
        assert config["dmd_refinement"]["data_update_offset"] == 18000
    assert sha256(parent/"last.pt") == parent_hash


@pytest.mark.parametrize("phase", ["matched_qualitative_samples", "teacher_response_diagnostic"])
def test_preview_gpu_argument_follows_reassigned_device_without_mutating_original(launcher, phase):
    command = ["python", "scripts/preview_dmd_refinement.py", "--mode", "block_norm_match", "--gpu", "1"]
    moved = launcher.phase_launch_args(command, phase, 2)
    assert moved[-1] == "2" and command[-1] == "1"
    ordinary = ["python", "-m", "few_step_diffusion.cli", "evaluate", "--steps", "1"]
    assert launcher.phase_launch_args(ordinary, "full_5000_image_evaluation", 2) == ordinary


def test_checkpoint_archive_preserves_full_state_and_rejects_replacement(launcher, tmp_path):
    config = {"name": "test", "train": {"updates_per_stage": 500}}
    state = {"config": config, "global_step": 250, "stage_index": 0, "stage_step": 250,
             "student": torch.tensor([1.]), "optimizer": {"step": 250}}
    torch.save(state, tmp_path/"last.pt")
    fingerprint = config_fingerprint(config)
    archive = launcher.preserve_milestone(tmp_path, fingerprint)
    assert sha256(archive) == sha256(tmp_path/"last.pt")
    restored = torch.load(archive, weights_only=False)
    assert torch.equal(restored["student"], state["student"])
    assert restored["optimizer"] == state["optimizer"]
    assert launcher.preserve_milestone(tmp_path, fingerprint) == archive
    with pytest.raises(ValueError, match="configuration"):
        launcher.preserve_milestone(tmp_path, "different")
    original = sha256(archive)
    state["student"].add_(1)
    torch.save(state, tmp_path/"last.pt")
    with pytest.raises(ValueError, match="different checkpoint"):
        launcher.preserve_milestone(tmp_path, fingerprint)
    assert sha256(archive) == original
    state.update(global_step=300, stage_step=300)
    torch.save(state, tmp_path/"last.pt")
    assert launcher.preserve_milestone(tmp_path, fingerprint) is None


@pytest.mark.parametrize("study", ["dmd_conflict", "dmd_noise_agreement", "dmd_pair_agreement"])
def test_refresher_keeps_fixed_baseline_and_rejects_wrong_protocol(launcher, tmp_path, study):
    is_noise = study in ("dmd_noise_agreement", "dmd_pair_agreement")
    pairs = 2 if study == "dmd_pair_agreement" else 1
    modes = (noise.PAIR_MODES if pairs == 2 else noise.MODES if is_noise else
             {key: MODES[key] for key in ("project", "norm_match")})
    policy = noise.PAIR_POLICY if pairs == 2 else noise.POLICY if is_noise else POLICY
    (tmp_path/"configs").mkdir()
    (tmp_path/"schemas").mkdir()
    shutil.copyfile("schemas/result.schema.json", tmp_path/"schemas/result.schema.json")
    reference = result_fixture("dmd_perceptual")
    reference.update(fid=29.94, inception_score_mean=34.12, clip_score_mean=31.10)
    reference.update(diversity_protocol="synthetic-test-only", diversity_prompt_hash="e"*64,
        diversity_prompt_count=128, diversity_seed_list=[12345,12346,12347,12348],
        diversity_samples_per_prompt=4, diversity_lpips_identity={"net":"alex", "test":True},
        diversity_generation_nfe=512, diversity_lpips_alex_std=0.,
        diversity_pairwise_scores_per_prompt=[.2]*128)
    reference_path = tmp_path/"reference.result.json"
    reference_path.write_text(json.dumps(reference))
    jobs = []
    for mode in modes:
        config = copy.deepcopy(reference["resolved_config"])
        config.update(name=modes[mode], output_dir=f"runs/{mode}")
        config["train"].update(mode="dmd", stages=[1], updates_per_stage=500,
            dmd_gradient_mode=mode, dmd_regression_weight=.25, dmd_pair_count=5000,
            dmd_regression_type="lpips_vgg", dmd_diversity_weight=0.)
        if is_noise:
            config["train"].pop("dmd_gradient_mode")
            config["train"].update(dmd_noise_mode=mode, dmd_agreement_patch_size=8)
            if pairs == 2:
                config["train"]["dmd_noise_pairs"] = 2
        config_path = tmp_path/f"configs/{mode}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        checkpoint = tmp_path/config["output_dir"]/"stage_1/adapter.safetensors"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic test fixture")
        jobs.append({"key": mode, "method": modes[mode], "run_dir": config["output_dir"],
            "config": str(config_path.relative_to(tmp_path)), "config_hash": config_fingerprint(config)})
    manifest = {"policy": policy, "jobs": jobs, "parent_checkpoint_sha256": "f"*64,
        "teacher_pair_weights_sha256": reference["teacher_pair_weights_sha256"],
        "reference_baseline": {"result_file": "reference.result.json",
            "result_sha256": sha256(reference_path), "model_identity": reference["model_identity"]}}
    (tmp_path/"configs"/f"{study}.json").write_text(json.dumps(manifest))
    launcher.refresh_results(tmp_path, study)
    initial = (tmp_path/f"{study.upper()}_RESULTS.md").read_text()
    assert "29.94" in initial and initial.count("pending | pending | pending") == len(modes)
    # A completed, independently validated experimental result has a distinct budget.
    result = copy.deepcopy(reference)
    for key in ("continuation", "continuation_policy", "continuation_only_steps", "dmd_study_policy",
                "dmd_study_parent_config_hash", "parent_training_updates", "parent_training_prompt_hash"):
        result.pop(key, None)
    job = jobs[0]
    result.update(method=job["method"], training_updates=500, cumulative_training_updates=36500,
        dmd_gradient_mode="project", refinement_policy=policy, refinement_updates=500,
        config_hash=job["config_hash"], fid=29.8,
        model_identity=sha256(tmp_path/job["run_dir"]/"stage_1/adapter.safetensors"),
        refinement_parent_checkpoint_sha256="f"*64,
        dmd_refinement={"policy": policy, "parent_training_updates": 36000,
            "additional_updates": 500, "parent_checkpoint_sha256": "f"*64,
            "training_prompt_hash": result["distillation_prompt_hash"]})
    result["resolved_config"] = yaml.safe_load((tmp_path/job["config"]).read_text())
    if is_noise:
        result.pop("dmd_gradient_mode")
        result.update(dmd_noise_mode=job["key"], dmd_score_probes=2*pairs, dmd_noise_pairs=pairs,
                      dmd_score_estimator_policy=policy, dmd_agreement_patch_size=8)
    result_dir = tmp_path/"results/sd15"/study
    result_dir.mkdir(parents=True)
    path = result_dir/f"{job['method']}_seed42_1steps.result.json"
    path.write_text(json.dumps(result))
    launcher.refresh_results(tmp_path, study)
    table = (tmp_path/f"{study.upper()}_RESULTS.md").read_text()
    assert "36,000 | 29.94" in table and "36,500 | 29.80" in table
    assert "500-update budget difference" in table
    result["eval_seed"] += 1
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="reference protocol"):
        launcher.refresh_results(tmp_path, study)


def test_diversity_requires_matching_seeds_and_complete_measurements(launcher):
    record = {"diversity_protocol": "test", "diversity_prompt_hash": "e"*64,
        "diversity_prompt_count": 128, "diversity_seed_list": [1,2,3,4],
        "diversity_samples_per_prompt": 4, "diversity_lpips_identity": {"net":"alex"},
        "diversity_generation_nfe": 512, "diversity_lpips_alex_mean": .2,
        "diversity_lpips_alex_std": 0., "diversity_pairwise_scores_per_prompt": [.2]*128}
    launcher.validate_diversity(record, record)
    changed = copy.deepcopy(record)
    changed["diversity_seed_list"] = [5,6,7,8]
    with pytest.raises(ValueError, match="protocol"):
        launcher.validate_diversity(changed, record)
    changed = copy.deepcopy(record)
    changed["diversity_pairwise_scores_per_prompt"].pop()
    with pytest.raises(ValueError, match="128 per-prompt"):
        launcher.validate_diversity(changed, record)
    changed = copy.deepcopy(record)
    changed["diversity_lpips_alex_mean"] = .3
    with pytest.raises(ValueError, match="128 per-prompt"):
        launcher.validate_diversity(changed, record)
