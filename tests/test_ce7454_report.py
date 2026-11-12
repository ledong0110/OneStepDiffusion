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
