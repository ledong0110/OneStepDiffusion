import json
import copy
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from few_step_diffusion.config import load_config
from few_step_diffusion.runtime import sha256


def test_paired_diversity_uncertainty_preserves_caption_pairing(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import build_assignment2_report as report
    reference = {"diversity_protocol": "fixture", "diversity_prompt_hash": "same captions",
        "diversity_prompt_count": 4, "diversity_seed_list": [1, 2, 3, 4],
        "diversity_samples_per_prompt": 4, "diversity_lpips_identity": {"weights": "same weights"},
        "diversity_pairwise_scores_per_prompt": [.1, .3, .7, .9], "diversity_lpips_alex_mean": .5}
    candidate = {**reference, "diversity_pairwise_scores_per_prompt": [.2, .4, .8, 1.],
                 "diversity_lpips_alex_mean": .6}
    # A constant per-caption change has no caption-resampling uncertainty,
    # even though the raw per-caption levels vary substantially.
    interval = report.paired_diversity_interval(reference, candidate)
    assert interval["mean_difference"] == pytest.approx(.1)
    assert interval["percentile_interval_95"] == pytest.approx([.1, .1])
    for key, value in (("diversity_prompt_hash", "different captions"),
                       ("diversity_seed_list", [4, 3, 2, 1]),
                       ("diversity_lpips_identity", {"weights": "different weights"}),
                       ("diversity_lpips_alex_mean", .7),
                       ("diversity_pairwise_scores_per_prompt", [.2, .4, .8])):
        with pytest.raises(ValueError):
            report.paired_diversity_interval(reference, {**candidate, key: value})


def test_frontmatter_distinguishes_pilot_from_final_angular_model(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import build_assignment2_report as report
    assert report.response_frontmatter(tmp_path, None, None) is None
    reference = {"fid": 30., "cumulative_training_updates": 36000}
    pilot = {"fixed_reference": reference, "results": [{"method": "dmd_response_angular_30",
        "fid": 29., "cumulative_training_updates": 36500}]}
    final = {"fixed_reference": reference, "results": [{"method": "dmd_response_magnitude_30",
        "fid": 27., "cumulative_training_updates": 38000}]}
    summary = report.response_frontmatter(tmp_path, pilot, final)
    assert summary["source"] == "pilot" and summary["fid"] == 29.
    assert "final model has not yet been evaluated" in (tmp_path/"response_summary.tex").read_text()
    final["results"].append({"method": "dmd_response_angular_30", "fid": 28.,
                             "cumulative_training_updates": 38000})
    summary = report.response_frontmatter(tmp_path, pilot, final)
    assert summary["source"] == "final" and summary["fid"] == 28.
    assert summary["additional_updates"] == 2000
    assert "2,000 fewer updates" in (tmp_path/"response_summary.tex").read_text()
    assert "best" not in (tmp_path/"response_summary.tex").read_text()


def test_concise_comparison_rejects_additional_update_pilots(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import build_assignment2_report as report
    baseline = {"fid": 29.94, "inception_score_mean": 34.12, "clip_score_mean": 31.10,
                "cumulative_training_updates": 36000, "nfe": 1}
    records = {method: {**baseline, "method": method} for method in
               ("baseline", "consistency", "dmd_perceptual", "swiftbrush_lora", "instaflow")}
    candidate = {**baseline, "method": "dmd_response_angular_30", "fid": 29.07,
        "cumulative_training_updates": 36500, "resolved_config": {"train": {
            "dmd_response_weight": .17909640596767878, "dmd_response_gradient_fraction": .3}}}
    teacher = {**baseline, "fid": 24.1}
    for updates in (18000, 36500, 37000, None):
        with pytest.raises(ValueError, match="exactly 36,000 cumulative"):
            report.concise_quality_table(records, teacher, {**candidate, "cumulative_training_updates": updates})
    candidate.update(method="response_scratch", cumulative_training_updates=36000, fid=28.58)
    table = report.concise_quality_table(records, teacher, candidate)
    assert "Ours (direction alignment) & 36,000 & 1" in table
    assert r"\textbf{28.58}" in table and r"\textbf{24.10}" not in table
    report.write_selected_response(tmp_path, {"study": "independent_response_36k"}, candidate, baseline)
    for filename in ("selected_response_summary.tex", "selected_response_settings.tex", "selected_response_decision.tex"):
        text = (tmp_path/filename).read_text()
        assert not any(old in text for old in ("36,500", "37,000", "pilot", "component tables"))


def test_main_figure_and_table_include_the_same_selected_proposal(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import build_assignment2_report as report
    common = {"fid": 30., "inception_score_mean": 34., "clip_score_mean": 31.,
              "cumulative_training_updates": 36000, "nfe": 1}
    records = {m: {**common, "method": m, "model_identity": m} for m in report.STANDARD_METHODS}
    candidate = {**common, "method": "response_scratch", "model_identity": "selected-36000",
                 "fid": 28.58}
    teacher = {**common, "method": "teacher", "model_identity": "teacher", "nfe": 128}
    columns = [teacher, candidate, *reversed(list(records.values()))]
    rendered = {}
    def capture(samples, indices, target, **kwargs):
        rendered.update(samples=samples, **kwargs)
        return {"methods": [r["method"] for r in samples]}
    monkeypatch.setattr(report, "draw_samples", capture)
    report.write_main_comparison(tmp_path, columns, {"rows": [0, 1, 2]}, records, teacher, candidate)
    assert rendered["samples"][-2] == candidate
    assert rendered["display_labels"][candidate["method"]] == "Ours"
    table = (tmp_path/"concise_quality.tex").read_text()
    assert "Ours (direction alignment) & 36,000 & 1" in table
    assert r"\textbf{28.58}" in table
    with pytest.raises(ValueError, match="one selected proposal"):
        report.write_main_comparison(tmp_path, [teacher, *records.values()],
                                    {"rows": [0]}, records, teacher, candidate)
    columns[1] = {**candidate, "model_identity": "different-37000"}
    with pytest.raises(ValueError, match="same evaluated models"):
        report.write_main_comparison(tmp_path, columns, {"rows": [0]}, records, teacher, candidate)


def test_assignment_publication_limits_main_report_but_not_appendix(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import build_assignment2_report as report
    build = tmp_path/"report/build"
    build.mkdir(parents=True)
    (build/"assignment2.pdf").write_bytes(b"synthetic PDF publication fixture")
    (build/"assignment2.log").write_text("ASSIGNMENT2-MAIN-PAGES:9\n")
    target = tmp_path/"CE6190_group_11_report_project2.pdf"
    target.write_bytes(b"previous valid report")
    monkeypatch.setattr(report.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(report.subprocess, "check_output", lambda *args, **kwargs: "Pages: 25\n")
    with pytest.raises(ValueError, match="excluding the appendix"):
        report.compile_report(tmp_path, "11")
    assert target.read_bytes() == b"previous valid report"
    # The same 25-page PDF is permitted when only eight pages belong to the main report.
    (build/"assignment2.log").write_text("ASSIGNMENT2-MAIN-PAGES:8\n")
    report.compile_report(tmp_path, "11")
    assert target.read_bytes() == (build/"assignment2.pdf").read_bytes()
    # A missing, duplicate or impossible boundary must not publish an unchecked PDF.
    (build/"assignment2.pdf").write_bytes(b"unchecked replacement")
    for log in ("", "ASSIGNMENT2-MAIN-PAGES:0\n", "ASSIGNMENT2-MAIN-PAGES:26\n",
                "ASSIGNMENT2-MAIN-PAGES:4\nASSIGNMENT2-MAIN-PAGES:8\n"):
        (build/"assignment2.log").write_text(log)
        with pytest.raises(ValueError, match="main-report page boundary"):
            report.compile_report(tmp_path, "11")
        assert target.read_bytes() == b"synthetic PDF publication fixture"


def test_compact_final_appendix_waits_for_model_matched_evidence(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import build_assignment2_report as report
    out = tmp_path/"report/generated/assignment2"
    out.mkdir(parents=True)
    (tmp_path/"configs").mkdir()
    methods = [f"dmd_response_{mode}_50" for mode in ("angular", "magnitude", "vector")]
    plan = {"jobs": [{"method": m} for m in methods],
            "additional_updates": 1200, "cumulative_training_updates": 37200}
    (tmp_path/"configs/dmd_response_final.json").write_text(json.dumps(plan))
    common = {"fid": 30., "inception_score_mean": 34., "clip_score_mean": 31.,
              "diversity_lpips_alex_mean": .71, "cumulative_training_updates": 37200}
    reference = {**common, "method": "dmd_perceptual", "model_identity": "fixed",
                 "cumulative_training_updates": 36000}
    records = [{**common, "method": m, "model_identity": f"evaluated-{m}"} for m in methods]
    identities = [{k: r[k] for k in ("method", "model_identity")} for r in records]
    study = {"fixed_reference": reference, "results": records,
             "completed_response_diagnostics": identities,
             "qualitative": {"methods": copy.deepcopy([reference, *identities])}}
    assert report.write_completed_response_final(tmp_path, out, study)
    section = out/"response_completed_final_section.tex"
    assert "1,200 updates (37,200 total)" in section.read_text()
    assert all(label in section.read_text() for label in ("Angular", "Magnitude", "Vector"))
    # A refresh with partial evidence must remove any formerly completed section.
    for key, value in (("results", records[:2]), ("completed_response_diagnostics", identities[:2]),
                       ("qualitative", None), ("qualitative", {"methods": [reference, *identities[:2]]}),
                       ("results", [{**records[0], "diversity_lpips_alex_mean": None}, *records[1:]])):
        assert not report.write_completed_response_final(tmp_path, out, {**study, key: value})
        assert section.read_text().startswith("% Awaiting")
    for field, key, value in (("results", "method", records[1]["method"]),
                              ("results", "cumulative_training_updates", 36500),
                              ("completed_response_diagnostics", "model_identity", "stale model")):
        changed = copy.deepcopy(study)
        changed[field][0][key] = value
        with pytest.raises(ValueError):
            report.write_completed_response_final(tmp_path, out, changed)
    changed = copy.deepcopy(study)
    changed["qualitative"]["methods"][1]["model_identity"] = "pilot model"
    with pytest.raises(ValueError, match="samples differ"):
        report.write_completed_response_final(tmp_path, out, changed)
    assert report.write_completed_response_final(tmp_path, out, study)
    if shutil.which("pdflatex"):
        (out/"response_final_calibration.tex").write_text("Synthetic calibration fixture.\n")
        (out/"response_final_findings.tex").write_text("Synthetic findings fixture.\n")
        (out/"response_final_diversity.tex").write_text("Synthetic diversity fixture.\n")
        (out/"response_final_alignment_section.tex").write_text("")
        (out/"response_final_samples.tex").write_text("")
        source = tmp_path/"report/fixture.tex"
        source.write_text(r"\documentclass{article}\usepackage{amsmath,booktabs,graphicx}"+"\n"+
            r"\begin{document}\input{generated/assignment2/response_completed_final_section.tex}\end{document}")
        result = subprocess.run(["pdflatex", "-halt-on-error", "-interaction=batchmode", source.name],
            cwd=source.parent, capture_output=True, text=True)
        assert result.returncode == 0, (source.parent/"fixture.log").read_text()[-5000:]


def test_final_report_uses_actual_budget_and_compiles(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import build_assignment2_report as report
    import launch_dmd_conflict as launcher
    out = tmp_path/"report/generated/assignment2"
    out.mkdir(parents=True)
    assert report.response_strength_study(tmp_path, out, "dmd_response_final", "response_final") is None
    assert (out/"response_final_section.tex").read_text().startswith("% No final")
    (tmp_path/"configs").mkdir()
    artifacts = tmp_path/"artifacts/dmd_teacher_response"
    artifacts.mkdir(parents=True)
    common = {"parent_checkpoint_sha256": "c"*64, "model_identity": "m"*64,
              "training_only": True, "weights_unchanged": True}
    calibration = {**common, "gradient_calibration": {"coefficients": {
        mode: {"weights": {str(f): f*scale for f in (.1,.3,.5)}}
        for mode, scale in (("angular",.6), ("magnitude",2.5), ("vector",.24))}}}
    calibration_file = artifacts/"gradient_calibration.json"
    calibration_file.write_text(json.dumps(calibration))
    (artifacts/"frozen_reference.json").write_text(json.dumps({**common, "prompt_count": 128,
        "summary": {"1": {"response_cosine": {"mean": .72},
                          "student_to_teacher_response_norm": {"mean": 1.07}}}}))
    jobs = []
    for mode in ("angular", "magnitude", "vector"):
        config = load_config("configs/dmd_perceptual.yaml")
        key = f"{mode}_30"
        config.update(name=f"dmd_response_{key}")
        config["train"].update(updates_per_stage=1200, dmd_response_variant=key, dmd_response_mode=mode,
            dmd_response_weight=calibration["gradient_calibration"]["coefficients"][mode]["weights"]["0.3"],
            dmd_response_gradient_fraction=.3)
        config["dmd_refinement"] = {"calibration_sha256": sha256(calibration_file)}
        (tmp_path/"configs"/f"{key}.yaml").write_text(yaml.safe_dump(config))
        jobs.append({"key": key, "method": config["name"], "run_dir": f"runs/{key}",
                     "config": f"configs/{key}.yaml"})
    plan = {"jobs": jobs, "additional_updates": 1200, "cumulative_training_updates": 37200,
            "parent_checkpoint_sha256": "c"*64, "reference_baseline": {"model_identity": "m"*64},
            "final_design": {"gradient_fraction": .3, "reason": "Synthetic 30% selection_test."}}
    (tmp_path/"configs/dmd_response_final.json").write_text(json.dumps(plan))
    def refresh(root, study):
        folder = root/"runs/sd15"/study
        folder.mkdir(parents=True, exist_ok=True)
        (folder/"comparison.json").write_text(json.dumps({"results": [],
            "fixed_reference": {"fid": 29.94, "diversity_lpips_alex_mean": .7, "model_identity": "m"*64}}))
        (root/"report/generated"/f"{study}_results.tex").write_text(
            r"\begin{tabular}{lr}Synthetic fixture & 37,200 \\ \end{tabular}")
    monkeypatch.setattr(launcher, "refresh_results", refresh)
    report.response_strength_study(tmp_path, out, "dmd_response_final", "response_final")
    section = (out/"response_final_section.tex").read_text()
    assert "1,200 further updates (37,200 total)" in section
    assert r"30\% selection\_test" in section
    assert "500 further" not in section
    assert "Magnitude" in (out/"response_final_diversity.tex").read_text()
    assert "vector: 0.0720" in (out/"response_final_calibration.tex").read_text()
    if shutil.which("pdflatex"):
        source = tmp_path/"report/fixture.tex"
        source.write_text(r"\documentclass{article}\usepackage{amsmath,booktabs,graphicx}"+"\n"+
            r"\begin{document}\input{generated/assignment2/response_final_section.tex}\end{document}")
        result = subprocess.run(["pdflatex", "-halt-on-error", "-interaction=batchmode", source.name],
            cwd=source.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert result.returncode == 0, (source.parent/"fixture.log").read_text()[-5000:]
    calibration_file.write_text(calibration_file.read_text()+"\n")
    with pytest.raises(ValueError, match="pinned training coefficients"):
        report.response_strength_study(tmp_path, out, "dmd_response_final", "response_final")


def test_response_report_binds_diagnostic_to_model_and_prompts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import build_assignment2_report as report
    import analyze_teacher_responses as diagnostic
    out = tmp_path/"generated"
    out.mkdir()
    run = tmp_path/"runs/angular_30"
    run.mkdir(parents=True)
    (run/"last.pt").write_bytes(b"completed test training state")
    job = {"key": "angular_30", "method": "dmd_response_angular_30", "run_dir": "runs/angular_30",
           "config_hash": "config"}
    plan = {"jobs": [job]}
    reference = {"policy": "frozen-teacher-finite-response-diagnostic-v1",
        "model_identity": "reference", "parent_checkpoint_sha256": "parent",
        "teacher_pair_weights_sha256": "bank", "teacher_pair_protocol_hash": "protocol",
        "training_only": True, "weights_unchanged": True, "optimizer_updates": 0,
        "prompt_hash": "prompts", "start_prompt": 128, "prompt_count": 2,
        "measurements": [{"prompt_index": 128+i, "image_id": i, "caption": f"caption {i}",
            "scales": {"1": {"response_cosine": .70+.02*i,
                "student_to_teacher_response_norm": 1.06+.02*i, "point_latent_mse": .45}}}
            for i in range(2)]}
    # The comparison summary intentionally omits config_hash; the scheduled job
    # and completed_source provide the validated configuration identity.
    comparison = {"results": [{"method": job["method"], "model_identity": "candidate"}]}
    assert report.response_alignment_evidence(tmp_path, out, plan, comparison,
        "dmd_response_strength", "response", reference) == []
    assert (out/"response_alignment_section.tex").read_text().startswith("% Awaiting")
    candidate = {**copy.deepcopy(reference), "model_identity": "candidate", "config_hash": "config",
        "reference_model_identity": "reference", "source_checkpoint_sha256": sha256(run/"last.pt"),
        "model_role": "evaluated_response_variant"}
    for row in candidate["measurements"]:
        row["scales"]["1"]["response_cosine"] += .05
    source = run/"response_diagnostic.json"
    source.write_text(json.dumps(candidate))
    # completed_source independently validates run completion, full evaluation,
    # and adapter/config identities in test_response_diagnostic.py.
    monkeypatch.setattr(diagnostic, "completed_source", lambda *args: (run, {}, plan))
    def render():
        return report.response_alignment_evidence(tmp_path, out, plan, comparison,
            "dmd_response_strength", "response", reference)
    evidence = render()
    assert evidence[0]["cosine_change"] == pytest.approx(.05)
    section = (out/"response_alignment_section.tex").read_text()
    assert "0.7100" in section and "0.7600" in section
    assert "training-bank measurements" in section
    assert (out/"response_diagnostic_angular_30.json").read_bytes() == source.read_bytes()
    if shutil.which("pdflatex"):
        (out/"fixture.tex").write_text(r"\documentclass{article}\usepackage{booktabs,graphicx}"+
            r"\begin{document}\input{response_alignment_section.tex}\end{document}")
        completed = subprocess.run(["pdflatex", "-halt-on-error", "-interaction=batchmode", "fixture.tex"],
            cwd=out, capture_output=True, text=True)
        assert completed.returncode == 0, (out/"fixture.log").read_text()[-5000:]
    for key, value in (("model_identity", "different model"), ("config_hash", "different config"),
                       ("source_checkpoint_sha256", "old checkpoint"),
                       ("prompt_hash", "other prompts"), ("optimizer_updates", 1)):
        changed = {**candidate, key: value}
        source.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            render()
    changed = copy.deepcopy(candidate)
    changed["measurements"][0]["caption"] = "wrong caption despite matching hash"
    source.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="matched prompts"):
        render()
