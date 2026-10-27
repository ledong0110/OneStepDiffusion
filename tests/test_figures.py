import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from few_step_diffusion.figures import make_sample_figure, read_samples


def test_json_only_samples_generate_a_cvpr_comparison(tmp_path):
    record = json.loads(Path("examples/result.template.json").read_text())
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(buffer, format="PNG")
    record.update(status="complete", sample_images_png_base64=[
        base64.b64encode(buffer.getvalue()).decode("ascii")]*8)
    rows = make_sample_figure([record], tmp_path, steps=4, seed=42)
    assert rows[0]["sample_indices"] == list(range(8))
    assert (tmp_path / "samples_comparison.pdf").exists()
    assert "figure*" in (tmp_path / "qualitative_figure.tex").read_text()
    assert len(read_samples(record)) == 8


def test_smoke_samples_do_not_enter_paper(tmp_path):
    record = json.loads(Path("examples/result.template.json").read_text())
    record.update(status="smoke")
    assert make_sample_figure([record], tmp_path, steps=4) == []
    assert not (tmp_path / "samples_comparison.pdf").exists()


def test_portable_sample_dimensions_are_checked():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, format="PNG")
    with pytest.raises(ValueError, match="32x32"):
        read_samples({"sample_images_png_base64": [base64.b64encode(buffer.getvalue()).decode()]})
