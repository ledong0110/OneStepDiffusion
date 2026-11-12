import importlib.util
import copy
import json
from pathlib import Path

import pytest

from few_step_diffusion.runtime import sha256
from test_active_report import result_fixture


@pytest.fixture
def assignment_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location("assignment_report", "scripts/build_assignment2_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assignment_uses_perceptual_dmd_and_labels_historical_mse(assignment_module, tmp_path):
    module = assignment_module
    records = {m: result_fixture(m) for m in module.LABELS}
    for record in records.values():
        record["fid"] = 50.
    records["dmd"]["fid"] = 37.02
    records["dmd_perceptual"]["fid"] = 29.94
    records["dmd_perceptual_diversity"]["fid"] = 30.40
    module.write_settings(tmp_path, "11", records)
    settings = (tmp_path/"settings.tex").read_text()
    assert r"\ADMDFID}{29.94}" in settings
    assert r"\ALatentDMDFID}{37.02}" in settings
    assert r"\ADiversityCost}{0.46}" in settings
    table = module.quality_table(records, result_fixture("teacher"),
        methods=(*module.ASSIGNMENT_METHODS, "dmd"), labels=module.ASSIGNMENT_LABELS)
    assert r"DMD + LPIPS & 1 & \textbf{29.94}" in table
    assert "DMD (latent-MSE control) & 1 & 37.02" in table
    assert "Ours" not in table


@pytest.mark.parametrize("study,method,prefix", [
    ("dmd_conflict", "dmd_conflict", "optimization_samples"),
    ("dmd_noise_agreement", "dmd_noise_agreement", "noise_agreement_samples"),
    ("dmd_pair_agreement", "dmd_pair_agreement", "pair_agreement_samples")])
def test_optimization_figure_requires_final_evaluated_model_and_matched_prompts(
        assignment_module, tmp_path, study, method, prefix):
    output = tmp_path/"generated"
    output.mkdir()
    (tmp_path/"configs").mkdir()
    job = {"method": method, "run_dir": "runs/project"}
    (tmp_path/"configs"/f"{study}.json").write_text(json.dumps({"jobs":[job]}))
    def render(comparison):
        return assignment_module.optimization_samples(tmp_path, output, comparison,
                                                      studies=(study,), prefix=prefix)
    comparison = {"fixed_reference": {"model_identity": "9"*64}, "results": []}
    assert render(comparison) is None
    assert not (output/f"{prefix}.pdf").exists()
    folder = tmp_path/"runs/project/checkpoint_previews/step_000500"
    folder.mkdir(parents=True)
    (folder/"checkpoint.pt").write_bytes(b"synthetic checkpoint fixture")
    baseline = result_fixture("dmd_perceptual")
    baseline["source_file"] = "test-reference"
    sample = copy.deepcopy(baseline)
    sample.update(method=method, model_identity="8"*64, global_step=500,
                  cumulative_training_updates=36500, source_file="test-proposal")
    result = {"method":method, "model_identity":"8"*64, "cumulative_training_updates":36500}
    comparison["results"].append(result)
    saved = {"checkpoint_sha256":sha256(folder/"checkpoint.pt"),
        "cumulative_training_updates":36500, "records":[baseline,sample]}
    path = folder/"samples.json"
    path.write_text(json.dumps(saved))
    selection = render(comparison)
    assert len(selection["methods"]) == 2 and (output/f"{prefix}.pdf").exists()
    sample["model_identity"] = "7"*64
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="evaluated models"):
        render(comparison)
    sample["model_identity"] = "8"*64
    sample["sample_prompts"][0] = "DIFFERENT TEST PROMPT"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="sample_prompts"):
        render(comparison)


def test_final_main_selection_requires_completed_arms_and_pinned_native_images(assignment_module, tmp_path):
    module = assignment_module
    baseline = result_fixture("dmd_perceptual")
    baseline["source_file"] = "test-reference"
    old = {**baseline, "method": "dmd_perceptual_diversity"}
    teacher = {**baseline, "method": "teacher", "sampling_steps": 64, "nfe": 128}
    columns = [baseline, old, teacher]
    assert module.select_main_candidate(tmp_path, columns, {}) == (columns, None)

    config_dir = tmp_path/"configs"
    config_dir.mkdir()
    method, control = "dmd_noise_agreement", "dmd_noise_mean"
    job = {"method": method, "run_dir": "runs/agreement", "result_dir": "results"}
    plan = {"jobs": [job, {"method": control}], "seed": 42, "additional_updates": 500}
    (config_dir/"dmd_noise_agreement.json").write_text(json.dumps(plan))
    folder = tmp_path/job["run_dir"]/"checkpoint_previews/step_000500"
    folder.mkdir(parents=True)
    (folder/"checkpoint.pt").write_bytes(b"synthetic selected checkpoint fixture")
    result = {"method": method, "model_identity": "8"*64, "cumulative_training_updates": 36500,
              "diversity_lpips_alex_mean": .25}
    sample = {**copy.deepcopy(baseline), **result, "global_step": 500}
    # A visibly different fixture proves replacement uses the selected native images.
    import base64
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), "red").save(buffer, format="PNG")
    sample["sample_images_png_base64"] = [base64.b64encode(buffer.getvalue()).decode()]*8
    saved = {"checkpoint_sha256": sha256(folder/"checkpoint.pt"),
             "cumulative_training_updates": 36500, "records": [baseline, sample]}
    (tmp_path/"results").mkdir()
    result_path = tmp_path/"results"/f"{method}_seed42_1steps.result.json"
    result_path.write_text(json.dumps(result))
    selection = {"policy": "assignment2-evaluated-main-candidate-v1", "study": "dmd_noise_agreement",
                 "method": method, "model_identity": result["model_identity"],
                 "result_sha256": sha256(result_path), "checkpoint_sha256": saved["checkpoint_sha256"],
                 "reason": "Synthetic test selection, not empirical evidence."}
    (config_dir/"assignment2_selected_method.json").write_text(json.dumps(selection))
    comparison = {"fixed_reference": baseline, "results": [result]}
    studies = {"dmd_noise_agreement": comparison}
    with pytest.raises(ValueError, match="all study arms"):
        module.select_main_candidate(tmp_path, columns, studies)
    comparison["results"].append({**result, "method": control})
    with pytest.raises(ValueError, match="native checkpoint samples"):
        module.select_main_candidate(tmp_path, columns, studies)
    preview_path = folder/"samples.json"
    preview_path.write_text(json.dumps(saved))
    updated, metadata = module.select_main_candidate(tmp_path, columns, studies)
    assert [r["method"] for r in updated] == ["dmd_perceptual", method, "teacher"]
    assert updated[0] == baseline and updated[2] == teacher
    assert updated[1]["sample_images_png_base64"] == sample["sample_images_png_base64"]
    assert updated[1]["sample_images_png_base64"] != old["sample_images_png_base64"]
    assert columns[1] == old  # Selection does not relabel or mutate old outputs.
    module.write_main_settings(tmp_path, metadata)
    assert "36,500" in (tmp_path/"main_selection.tex").read_text()
    # Both actual budgets must be disclosed; no particular adjective is required.
    assert "36,000" in (tmp_path/"main_selection.tex").read_text()
    sample["model_identity"] = "7"*64
    preview_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="evaluated models"):
        module.select_main_candidate(tmp_path, columns, studies)
    sample["model_identity"] = "8"*64
    sample["sample_prompts"][0] = "DIFFERENT TEST PROMPT"
    preview_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="prompts or noise"):
        module.select_main_candidate(tmp_path, columns, studies)
    result_path.write_text(json.dumps({**result, "diversity_lpips_alex_mean": .3}))
    with pytest.raises(ValueError, match="pinned identity"):
        module.select_main_candidate(tmp_path, columns, studies)


def test_independent_main_uses_evaluated_36k_model_and_rejects_wrong_native_sample(
        assignment_module, tmp_path, monkeypatch):
    import hashlib
    import few_step_diffusion.independent_response as independent
    module = assignment_module
    # Training-protocol validation has its own tests; this fixture targets the
    # report's independent result/checkpoint/figure identity boundary.
    monkeypatch.setattr(independent, "validate_result", lambda value: None)
    monkeypatch.setattr(module, "load_result", lambda root, path:
                        {**json.loads(path.read_text()), "source_file": str(path.relative_to(root))})
    run = tmp_path/"runs/new_proposal"
    (run/"stage_1").mkdir(parents=True)
    (run/"qualitative").mkdir()
    (run/"last.pt").write_bytes(b"synthetic completed checkpoint")
    (run/"stage_1/adapter.safetensors").write_bytes(b"synthetic evaluated adapter")
    (run/"run.json").write_text(json.dumps({"status": "complete", "global_step": 36000}))
    result = result_fixture("response_scratch")
    result.update(model_identity=sha256(run/"stage_1/adapter.safetensors"), training_updates=36000)
    folder = tmp_path/"results/sd15/independent36k"
    folder.mkdir(parents=True)
    path = folder/"response_scratch_seed42_1steps.result.json"
    path.write_text(json.dumps(result))
    config_dir = tmp_path/"configs"
    config_dir.mkdir()
    (config_dir/"independent_response_36k.json").write_text(json.dumps({"jobs": [
        {"method": "response_scratch", "run_dir": "runs/new_proposal", "config_hash": result["config_hash"]}]}))
    rows = [{"key": "teddy", "prompt": result["sample_prompts"][4], "label": "Teddy bears",
             "source": "validation", "noise_stream_index": 4, "noise_sha256": "1"*64}]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    baseline = result_fixture("dmd_perceptual")
    baseline.update(sample_prompts=[rows[0]["prompt"]], evaluation_prompt_hash=digest)
    old = {**baseline, "method": "dmd_perceptual_diversity"}
    payload = {"protocol": "matched-main-figure-v1", "image_size": 512, "rows": rows,
        "protocol_hash": digest, "eval_seed": baseline["eval_seed"], "methods": {"response_scratch": {
            "model_identity": result["model_identity"], "source_result_sha256": sha256(path),
            "cumulative_training_updates": 36000, "sampling_steps": 1, "nfe": 1,
            "rows": [{**rows[0], "png_base64": result["sample_images_png_base64"][4]}]}}}
    main_path = run/"qualitative/main_comparison.json"
    main_path.write_text(json.dumps(payload))
    selection = {"policy": "assignment2-evaluated-main-candidate-v1", "study": "independent_response_36k",
        "method": "response_scratch", "reason": "Completed fixed 36k endpoint", "model_identity": result["model_identity"],
        "result_sha256": sha256(path), "checkpoint_sha256": sha256(run/"last.pt"),
        "qualitative_main_sha256": sha256(main_path)}
    selection_path = config_dir/"assignment2_selected_method.json"
    selection_path.write_text(json.dumps(selection))
    columns, selected = module.select_main_candidate(tmp_path, [baseline, old], {})
    assert columns[1]["method"] == "response_scratch"
    assert columns[1]["sample_images_png_base64"] == [result["sample_images_png_base64"][4]]
    assert selected["cumulative_training_updates"] == 36000
    assert selected["independent_control_complete"] is False
    assert old["method"] == "dmd_perceptual_diversity"
    payload["methods"]["response_scratch"]["rows"][0]["png_base64"] = "different native sample"
    main_path.write_text(json.dumps(payload))
    selection["qualitative_main_sha256"] = sha256(main_path)
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="validation image differs"):
        module.select_main_candidate(tmp_path, [baseline, old], {})


def test_independent_summary_reports_equal_updates_and_pending_control(assignment_module, tmp_path):
    record = result_fixture("response_scratch")
    record["resolved_config"]["train"]["dmd_response_weight"] = .2984940099461313
    reference = result_fixture("dmd_perceptual")
    assignment_module.write_selected_response(tmp_path,
        {"study": "independent_response_36k", "method": "response_scratch", "independent_control_complete": False},
        record, reference)
    assert r"\ASelectedUpdates}{36,000}" in (tmp_path/"selected_response_metrics.tex").read_text()
    assert "final full-validation scores are pending" in (tmp_path/"selected_response_metrics.tex").read_text()
    assert "independently trained" in (tmp_path/"selected_response_summary.tex").read_text()
    assert "first update" in (tmp_path/"selected_response_settings.tex").read_text()
    decision = (tmp_path/"selected_response_decision.tex").read_text()
    assert "independently trained 36,000-update model" in decision
    assert not any(old in decision for old in ("pilot", "36,500", "37,000"))
