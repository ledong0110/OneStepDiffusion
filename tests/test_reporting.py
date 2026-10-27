import copy
import json
from pathlib import Path

import pytest

from few_step_diffusion.reporting import collect_results, tex_escape, write_report_inputs

SCHEMA = Path("schemas/result.schema.json")


@pytest.fixture
def record():
    r = json.loads(Path("examples/result.template.json").read_text())
    # Synthetic fixture is confined to pytest temp directories, never the paper folder.
    r.update(status="complete", fid=20.0, inception_score_mean=7.0, inception_score_std=0.1)
    return r


def save(folder, name, record):
    (folder / name).write_text(json.dumps(record))


def test_team_json_names_and_seed_aggregation(tmp_path, record):
    save(tmp_path, "friend_alice.json", record)
    second = copy.deepcopy(record)
    second.update(seed=43, fid=22.0)
    save(tmp_path, "friend_bob.json", second)
    records, skipped = collect_results(tmp_path, SCHEMA)
    out = tmp_path / "generated"
    summary = write_report_inputs(records, skipped, out)
    assert summary[0]["fid_mean"] == 21
    assert summary[0]["seeds"] == 2
    assert r"\pm" in (out / "quality_table.tex").read_text()
    assert (out / "quality_vs_steps.pdf").exists()


def test_smoke_results_cannot_enter_paper(tmp_path, record):
    record.update(status="smoke", fid=None)
    save(tmp_path, "smoke.json", record)
    records, skipped = collect_results(tmp_path, SCHEMA)
    assert records == [] and len(skipped) == 1


@pytest.mark.parametrize("field,value", [("num_samples", 1000), ("training_updates", 10),
                                        ("timesteps", [999, 700, 400, 100]),
                                        ("effective_batch_size", 16),
                                        ("teacher_identity", "different-teacher")])
def test_incompatible_protocols_are_rejected(tmp_path, record, field, value):
    save(tmp_path, "a.json", record)
    second = copy.deepcopy(record)
    second.update(seed=43)
    second[field] = value
    save(tmp_path, "b.json", second)
    with pytest.raises(ValueError):
        collect_results(tmp_path, SCHEMA)


def test_conflicting_duplicates_and_nan_are_rejected(tmp_path, record):
    save(tmp_path, "a.json", record)
    second = copy.deepcopy(record)
    second["fid"] = 5
    save(tmp_path, "b.json", second)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        collect_results(tmp_path, SCHEMA)
    (tmp_path / "b.json").write_text('{"fid": NaN}')
    with pytest.raises(ValueError, match="Nonstandard numeric"):
        collect_results(tmp_path, SCHEMA)


def test_method_labels_are_tex_escaped():
    assert tex_escape(r"method_1%\input{secret}") == r"method\_1\%\textbackslash{}input\{secret\}"
