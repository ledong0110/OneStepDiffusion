"""Protect full-validation budgets and matched-prompt visual selection."""

import base64
import copy
import importlib.util
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from few_step_diffusion.config import config_fingerprint, load_config
from few_step_diffusion.dmd_diversity import VARIANTS
from test_sd_reporting import sd_result


@pytest.fixture
def report_module():
    spec = importlib.util.spec_from_file_location("active_report", "scripts/build_active_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_fixture(method="dmd_perceptual_diversity"):
    record = sd_result()
    config = load_config("configs/sd15_50k.yaml")
    config.update(name=method, output_dir=f"runs/{method}")
    config["data"].update(batch_size=2, train_limit=68287, train_offset=50000)
    config["train"].update(edge_weight=0, grad_accumulation=8, updates_per_stage=18000)
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), "blue").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    record.update(method=method, resolved_config=config, config_hash=config_fingerprint(config),
        precision="bf16", seed=42, effective_batch_size=16, num_samples=5000, num_real=5000,
        training_updates=18000, edge_weight=0, parent_training_updates=18000,
        cumulative_training_updates=36000, parent_training_prompt_hash="e"*64,
        continuation_policy="fresh-coco-one-step-v1", continuation_only_steps=1,
        continuation={"policy": "fresh-coco-one-step-v1", "parent_training_updates": 18000,
            "parent_training_prompt_hash": "e"*64, "parent_training_prompt_count": 50000,
            "fresh_training_prompt_count": 68287, "combined_unique_training_images": 118287,
            "fresh_training_prompt_hash": record["distillation_prompt_hash"], "training_image_overlap": 0,
            "additional_updates": 18000, "parent_checkpoint_sha256": "f"*64},
        sample_images_png_base64=[encoded]*8, sample_prompts=[f"SYNTHETIC TEST FIXTURE {i}" for i in range(8)],
        sample_image_ids=list(range(8)), model_identity="9"*64)
    if method in VARIANTS:
        regression, weight = VARIANTS[method]
        record.update(dmd_study_policy="dmd-perceptual-diversity-v1", dmd_study_parent_config_hash="a"*64,
            dmd_regression_type=regression, dmd_diversity_weight=weight, dmd_pair_seeds=2,
            dmd_regression_weight=.25, teacher_pair_generation_seconds=1,
            teacher_pair_count=5000, teacher_pair_unique_prompts=2500,
            teacher_pair_nfe=5000*record["initial_steps"]*2, paired_regression_batch_size=16,
            teacher_pair_protocol_hash="c"*64, teacher_pair_weights_sha256="d"*64,
            diversity_lpips_alex_mean=.2)
    return record


def test_import_rejects_partial_validation_and_wrong_total_budget(tmp_path, report_module):
    (tmp_path/"schemas").mkdir()
    shutil.copy("schemas/result.schema.json", tmp_path/"schemas/result.schema.json")
    record = result_fixture()
    path = tmp_path/"fixture.result.json"
    path.write_text(json.dumps(record))
    assert report_module.load_result(tmp_path, path)["cumulative_training_updates"] == 36000
    record.update(num_samples=128, num_real=128)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="5,000"):
        report_module.load_result(tmp_path, path)
    record.update(num_samples=5000, num_real=5000, training_updates=17000,
                  cumulative_training_updates=35000)
    record["continuation"]["additional_updates"] = 17000
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="36,000"):
        report_module.load_result(tmp_path, path)


def test_selected_rows_reject_mismatched_caption_and_noise(tmp_path, report_module):
    first = result_fixture()
    second = copy.deepcopy(first)
    second["sample_prompts"][2] = "DIFFERENT PROMPT"
    with pytest.raises(ValueError, match="sample_prompts"):
        report_module.draw_samples([first, second], [2, 5, 6], tmp_path/"bad")
    second = copy.deepcopy(first)
    second["eval_seed"] += 1
    with pytest.raises(ValueError, match="eval_seed"):
        report_module.draw_samples([first, second], [2, 5, 6], tmp_path/"bad")
    assert not (tmp_path/"bad.pdf").exists()


def test_main_comparison_rejects_wrong_model_noise_and_changed_validation_image(tmp_path, report_module):
    import hashlib
    proposal = result_fixture()
    teacher = {**result_fixture(), "method": "teacher", "sampling_steps": 64, "nfe": 128}
    rows = [{"key": "dog", "prompt": "A dog in flowers.", "label": "Dog in flowers",
             "source": "preset", "noise_stream_index": 7, "noise_sha256": "a"*64},
            {"key": "teddy", "prompt": proposal["sample_prompts"][4], "label": "COCO 776",
             "source": "validation", "noise_stream_index": 4, "noise_sha256": "b"*64}]
    payload = {"protocol": "matched-main-figure-v1", "image_size": 512,
        "eval_seed": proposal["eval_seed"], "rows": rows,
        "protocol_hash": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "methods": {r["method"]: {**{k:r[k] for k in
            ("model_identity", "sampling_steps", "nfe", "precision", "cumulative_training_updates")},
            "rows": [{**row, "png_base64": r["sample_images_png_base64"][4]} for row in rows]}
            for r in (proposal, teacher)}}
    path = tmp_path/"main_comparison.json"
    path.write_text(json.dumps(payload))
    comparisons, _ = report_module.load_main_comparison(tmp_path, [proposal, teacher], path)
    assert comparisons[0]["sample_labels"] == ["Dog in flowers", "COCO 776"]
    original = copy.deepcopy(payload)
    payload["methods"][proposal["method"]]["model_identity"] = "0"*64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="model_identity"):
        report_module.load_main_comparison(tmp_path, [proposal, teacher], path)
    payload = copy.deepcopy(original)
    payload["methods"][proposal["method"]]["rows"][0]["noise_sha256"] = "0"*64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="prompts or noise"):
        report_module.load_main_comparison(tmp_path, [proposal, teacher], path)
    payload = copy.deepcopy(original)
    payload["methods"][proposal["method"]]["rows"][1]["png_base64"] = "changed"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="validation sample changed"):
        report_module.load_main_comparison(tmp_path, [proposal, teacher], path)


def test_shared_control_requires_same_anchor_bank_and_initialization(report_module):
    first, second = result_fixture(), result_fixture("dmd_grouped")
    report_module.verify_common([first, second])
    second["teacher_pair_weights_sha256"] = "0"*64
    with pytest.raises(ValueError, match="teacher_pair_weights_sha256"):
        report_module.verify_common([first, second])
    second = result_fixture("dmd_grouped")
    second["continuation"]["parent_checkpoint_sha256"] = "0"*64
    with pytest.raises(ValueError, match="same initialization"):
        report_module.verify_common([first, second])


def test_validation_coverage_checks_all_official_image_ids(tmp_path, report_module):
    import hashlib
    base = tmp_path/"data/coco2017"
    (base/"annotations").mkdir(parents=True)
    rows = [{"image_id": i, "caption": f"SYNTHETIC {i}"} for i in range(5000)]
    (base/"val2017_5000.manifest.json").write_text(json.dumps({"records": rows}))
    (base/"annotations/captions_val2017.json").write_text(json.dumps({"images": [{"id": i} for i in range(5000)]}))
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    record = {"evaluation_prompt_hash": digest, "reference_manifest_hash": digest}
    assert report_module.verify_validation(tmp_path, record)["local_manifest_checked"]
    rows[-1]["image_id"] = 0
    (base/"val2017_5000.manifest.json").write_text(json.dumps({"records": rows}))
    with pytest.raises(ValueError, match="complete official"):
        report_module.verify_validation(tmp_path, record)


def test_primary_table_bolds_displayed_winners_independently_of_ablations(report_module):
    records = {
        "dmd": {"fid": 37.02, "inception_score_mean": 29.38, "clip_score_mean": 30.57},
        "dmd_perceptual_diversity": {"fid": 30.40, "inception_score_mean": 33.76, "clip_score_mean": 31.12},
        "dmd_perceptual": {"fid": 29.94, "inception_score_mean": 34.12, "clip_score_mean": 31.10}}
    teacher = {"fid": 24.10, "inception_score_mean": 34.46, "clip_score_mean": 31.30}
    table = report_module.quality_table(records, teacher)
    assert r"Ours & 1 & \textbf{30.40} & \textbf{33.76} & \textbf{31.12}" in table
    assert "29.94" not in table
    assert r"\textbf{24.10}" not in table


def test_completed_ablation_findings_report_the_diversity_tradeoff(report_module):
    records = {
        "dmd_perceptual_diversity": {"fid": 30.397064, "inception_score_mean": 33.759,
            "clip_score_mean": 31.1205, "diversity_lpips_alex_mean": .712911},
        "dmd": {"fid": 37.021989},
        "dmd_grouped": {"fid": 34.202037, "diversity_lpips_alex_mean": .728254},
        "dmd_perceptual": {"fid": 29.937644, "diversity_lpips_alex_mean": .708632},
        "dmd_diversity": {"fid": 33.977077, "diversity_lpips_alex_mean": .727439}}
    findings = report_module.measured_findings(records)
    assert "Perceptual regression alone obtains FID 29.94" in findings
    assert "changes FID by +0.46" in findings
    assert "from 0.7086 to 0.7129" in findings
    assert "does not improve FID" in findings
    assert "changes FID by -0.22" in findings
    assert "unfinished" not in findings


def test_root_pdf_updates_only_after_a_successful_page_check(tmp_path, report_module, monkeypatch):
    build = tmp_path/"report/build"
    build.mkdir(parents=True)
    source = build/"main.pdf"
    source.write_bytes(b"synthetic PDF test fixture")
    published = tmp_path/"main_report.pdf"
    published.write_bytes(b"previous published report")
    monkeypatch.setattr(report_module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(report_module.subprocess, "check_output", lambda *args, **kwargs: "Pages: 5\n")
    with pytest.raises(ValueError, match="exceeding"):
        report_module.compile_report(tmp_path)
    assert published.read_bytes() == b"previous published report"
    monkeypatch.setattr(report_module.subprocess, "check_output", lambda *args, **kwargs: "Pages: 4\n")
    report_module.compile_report(tmp_path)
    assert published.read_bytes() == source.read_bytes()
    assert not (tmp_path/".main_report.pdf.tmp").exists()


def watcher_inputs(root):
    (root/"configs").mkdir()
    (root/"configs/active_study.json").write_text(json.dumps({"jobs": [
        {"method": "baseline", "result_dir": "results"}]}))
    (root/"configs/report_selection.json").write_text("{}")
    (root/"results").mkdir()
    return root/"results/baseline_seed42_1steps.result.json"


def test_watcher_rebuilds_on_final_result_arrival_and_diversity_update(
        tmp_path, report_module, monkeypatch):
    from few_step_diffusion.runtime import atomic_json
    result = watcher_inputs(tmp_path)
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    builds, compiles, polls = [], [], []

    def build():
        builds.append(result.read_text() if result.exists() else None)
        return {"updated_at": "test", "completed": int(result.exists())}

    def sleep(_):
        polls.append(1)
        if len(polls) == 1:
            result.with_suffix(".tmp").write_text("incomplete evaluation")
        elif len(polls) == 2:
            atomic_json(result, {"status": "complete"})
        elif len(polls) == 4:
            atomic_json(result, {"status": "complete", "diversity_lpips_alex_mean": .7})
        elif len(polls) == 5:
            raise KeyboardInterrupt

    monkeypatch.setattr(report_module, "build", build)
    monkeypatch.setattr(report_module, "compile_report", lambda *args: compiles.append(args))
    monkeypatch.setattr(report_module.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        report_module.main(["--watch"])
    # No builds for a temporary partial output or an unchanged result.
    assert len(builds) == len(compiles) == 3
    assert builds[0] is None
    assert "diversity_lpips_alex_mean" not in builds[1]
    assert "diversity_lpips_alex_mean" in builds[2]
    state = json.loads((tmp_path/"report/build/active_report_status.json").read_text())
    assert state["status"] == "watching"


def test_watcher_recovers_from_temporary_input_and_compile_errors(
        tmp_path, report_module, monkeypatch):
    watcher_inputs(tmp_path)
    plan = tmp_path/"configs/active_study.json"
    original = plan.read_text()
    plan.write_text("{")
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    compiles, states = [], []
    monkeypatch.setattr(report_module, "build", lambda: {"updated_at": "test", "completed": 0})

    def compile_report(*_):
        compiles.append(1)
        if len(compiles) == 1:
            raise RuntimeError("temporary compiler failure")

    def sleep(_):
        states.append(json.loads((tmp_path/"report/build/active_report_status.json").read_text()))
        if len(states) == 1:
            plan.write_text(original)
        elif len(states) == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(report_module, "compile_report", compile_report)
    monkeypatch.setattr(report_module.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        report_module.main(["--watch"])
    assert [s["status"] for s in states] == ["error", "error", "watching"]
    assert len(compiles) == 2


def test_background_watcher_is_detached_cpu_only_and_reuses_existing_process(
        tmp_path, report_module, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import launch_scale50k
    monkeypatch.setattr(report_module, "ROOT", tmp_path)
    monkeypatch.setattr(launch_scale50k, "process_record", lambda pid: {"pid": pid})
    monkeypatch.setattr(launch_scale50k, "process_matches", lambda record: record == {"pid": 1234})
    launched = []

    def popen(command, **kwargs):
        launched.append((command, kwargs))
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(report_module.subprocess, "Popen", popen)
    args = SimpleNamespace(interval=60, max_pages=4, no_compile=False)
    assert report_module.start_background(args)["status"] == "started"
    assert report_module.start_background(args)["status"] == "already_running"
    assert len(launched) == 1
    command, options = launched[0]
    assert "--watch" in command and "--background" not in command
    assert options["start_new_session"] is True
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert options["stdin"] == report_module.subprocess.DEVNULL
