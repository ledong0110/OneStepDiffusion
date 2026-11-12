import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def report_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location("ce7454_report", "scripts/build_ce7454_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_references_are_excluded_from_the_main_text_limit(report_module):
    assert report_module.check_page_budget("CE7454-MAIN-PAGES:4\n", 5) == (4, 1)
    assert report_module.check_page_budget("CE7454-MAIN-PAGES:4\n", 7) == (4, 3)
    with pytest.raises(ValueError, match="main text has 5 pages"):
        report_module.check_page_budget("CE7454-MAIN-PAGES:5\n", 6)


@pytest.mark.parametrize("log,total", [
    ("", 5), ("CE7454-MAIN-PAGES:0\n", 5), ("CE7454-MAIN-PAGES:5\n", 5),
    ("CE7454-MAIN-PAGES:6\n", 5),
    ("CE7454-MAIN-PAGES:4\nCE7454-MAIN-PAGES:4\n", 5),
])
def test_invalid_or_missing_boundary_cannot_hide_content(report_module, log, total):
    with pytest.raises(ValueError, match="page boundary"):
        report_module.check_page_budget(log, total)


def test_watcher_uses_the_course_main_text_budget(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    import build_active_report
    import build_ce7454_report
    metadata = tmp_path/"report/generated/ce7454/provenance.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("{}")
    calls = []
    monkeypatch.setattr(build_ce7454_report, "build", lambda *args, **kwargs: calls.append((args, kwargs)))
    build_active_report.compile_report(tmp_path, max_pages=4)
    assert calls == [((tmp_path,), {"refresh": False, "max_pages": 4})]


def test_analysis_tables_preserve_metric_signs_and_recorded_exposure(report_module, tmp_path):
    reference = {"method": "dmd_perceptual", "cumulative_training_updates": 36000,
                 "fid": 29.94, "inception_score_mean": 34.12, "clip_score_mean": 31.10}
    ours = {**reference, "method": "response_scratch", "fid": 28.58,
            "inception_score_mean": 33.02, "clip_score_mean": 30.91}
    proposal = {**ours, "effective_batch_size": 16,
                "training_data_phases": [{"count": n, "updates": 18000} for n in (50000, 68287)],
                "teacher_pair_phases": [{"teacher_pair_count": 5000} for _ in range(2)],
                "cumulative_teacher_pair_count": 10000}
    records = {"response_scratch": ours, "dmd_perceptual": reference}
    summary = report_module.write_analysis_tables(tmp_path, records, proposal)
    assert summary["metric_changes"][0]["difference"] == pytest.approx(-1.36)
    assert summary["metric_changes"][0]["percent"] == pytest.approx(-100*1.36/29.94)
    assert "-1.36 & -4.54" in (tmp_path/"metric_changes.tex").read_text()
    assert "-1.10" in (tmp_path/"metric_changes.tex").read_text()
    assert "-0.19" in (tmp_path/"metric_changes.tex").read_text()
    assert "Total & 118,287 & 576,000 & 10,000" in (tmp_path/"training_exposure.tex").read_text()
    proposal["training_data_phases"][1]["updates"] = 19000
    with pytest.raises(ValueError, match="recorded two-phase 36k protocol"):
        report_module.write_analysis_tables(tmp_path, records, proposal)


def test_only_requested_result_tables_are_published(report_module):
    labels = ("tab:quality", "tab:metric_changes", "tab:exposure")
    aux = "\n".join(r"\contentsline {table}{"+"\n"+r"\newlabel{"+label+"}{fixture}" for label in labels)
    aux += "\n"+"\n".join(r"\newlabel{"+label+"@cref}{fixture}" for label in labels)
    report_module.check_result_tables(aux)
    with pytest.raises(ValueError, match="regression-control tables are excluded"):
        report_module.check_result_tables(aux.replace("tab:metric_changes", "tab:ablation"))
    with pytest.raises(ValueError, match="regression-control tables are excluded"):
        report_module.check_result_tables(aux+"\n"+r"\contentsline {table}{")
