import base64
import copy
import io
import json

from PIL import Image
import pytest

from few_step_diffusion.figures import make_qualitative_panels
from few_step_diffusion.qualitative import (CORE_METHODS, EDGE_METHODS, load_qualitative,
                                            panel_specs, protocol_hash)
from few_step_diffusion.reporting import collect_results


def qualitative(method="proposed", seed=42):
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), "blue").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return {"schema_version": 1, "status": "qualitative", "method": method,
            "seed": seed, "sampling_steps": 1, "nfe": 1, "image_size": 512,
            "eval_seed": 12345, "protocol_hash": protocol_hash(12345),
            "teacher_identity": "TEST TEACHER", "precision": "bf16", "lora_rank": 32,
            "model_identity": "a"*64, "training_updates": 600,
            "panels": {kind: [{**item, "png_base64": encoded} for item in items]
                       for kind, items in panel_specs().items()}}


def test_portable_figures_validate_prompts_noise_and_checkpoint_and_exclude_metrics(tmp_path):
    r = qualitative()
    (tmp_path/"friend_renamed_file.json").write_text(json.dumps(r))
    records = load_qualitative(tmp_path)
    assert len(records) == 1
    metrics, skipped = collect_results(tmp_path, "schemas/result.schema.json")
    assert not metrics and skipped[0]["reason"] == "qualitative"
    changed = copy.deepcopy(r)
    changed["panels"]["comparison"][0]["noise_seed"] += 1
    (tmp_path/"friend_renamed_file.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="initial noise"):
        load_qualitative(tmp_path)
    (tmp_path/"friend_renamed_file.json").write_text(json.dumps(r))
    with pytest.raises(ValueError, match="checkpoint differs"):
        load_qualitative(tmp_path, [{**r, "model_identity": "b"*64}])


def test_fixed_prompt_panels_render_all_three_requested_figures(tmp_path):
    for method in dict.fromkeys(CORE_METHODS+EDGE_METHODS):
        (tmp_path/f"{method}.json").write_text(json.dumps(qualitative(method)))
    output = tmp_path/"generated"
    output.mkdir()
    imported = load_qualitative(tmp_path)
    manifest = make_qualitative_panels(imported, output)
    assert len(manifest) == 9
    for name in ("samples_comparison.pdf", "samples_comparison_full.pdf", "ablation_visual.pdf", "prompt_control.pdf"):
        assert (output/name).is_file()
    text = (output/"extra_qualitative.tex").read_text()
    assert "Visual edge-loss ablation" in text and "Compositional prompt control" in text
    assert "not continuous embedding interpolation" in text
    assert all(i["noise_seed"] == 12545 for i in panel_specs()["control"])
