import base64
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image
import pytest

from few_step_diffusion.dmd_diversity import VARIANTS
from test_sd_reporting import sd_result


def test_controlled_dmd_report_rejects_parent_changes_and_compiles(tmp_path):
    spec = importlib.util.spec_from_file_location("dmd_report", Path("scripts/build_dmd_report.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = tmp_path/"report"
    shutil.copytree("report", report, ignore=shutil.ignore_patterns("build", "generated", "assets"))
    main = report/"dmd_diversity.tex"
    main.write_text(main.read_text().replace("Teacher-Guided Diversity Matching for One-Step Diffusion",
                                           "SYNTHETIC TEST FIXTURES: Layout Verification"))
    results = tmp_path/"results"
    results.mkdir()
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), "blue").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    records = []
    for method, (regression, weight) in VARIANTS.items():
        r = sd_result()
        r.update(method=method, dmd_study_policy="dmd-perceptual-diversity-v1",
            dmd_study_parent_config_hash="a"*64, dmd_regression_type=regression,
            dmd_diversity_weight=weight, dmd_regression_weight=.25,
            dmd_pair_seeds=2, teacher_pair_count=4, teacher_pair_unique_prompts=2,
            teacher_pair_nfe=4*r["initial_steps"]*2, teacher_pair_generation_seconds=1,
            paired_regression_batch_size=4, teacher_pair_protocol_hash="c"*64,
            teacher_pair_weights_sha256="d"*64,
            parent_training_updates=18000, parent_training_prompt_hash="e"*64,
            cumulative_training_updates=21000, continuation_only_steps=1,
            continuation_policy="fresh-coco-one-step-v1",
            continuation={"policy": "fresh-coco-one-step-v1", "parent_training_updates": 18000,
                "parent_training_prompt_hash": "e"*64, "parent_training_prompt_count": 50000,
                "fresh_training_prompt_count": 68287, "combined_unique_training_images": 118287,
                "fresh_training_prompt_hash": r["distillation_prompt_hash"], "training_image_overlap": 0,
                "additional_updates": 3000, "parent_checkpoint_sha256": "f"*64},
            sample_images_png_base64=[encoded]*8, sample_prompts=["SYNTHETIC TEST FIXTURE"]*8,
            sample_image_ids=list(range(8)), diversity_protocol="fixture-four-seeds",
            diversity_prompt_hash="a"*64, diversity_seed_list=[12345, 12346, 12347, 12348],
            diversity_lpips_alex_mean=.2, diversity_lpips_identity={"net": "alex"}, training_seconds=100)
        (results/f"{method}.result.json").write_text(json.dumps(r))
        records.append(r)
    generated = report/"generated/dmd_diversity"
    module.build(results, generated)
    assert "Full proposal" in (generated/"quality.tex").read_text()
    assert (generated/"samples.pdf").exists()
    assert (results/"comparison.csv").exists()
    if shutil.which("latexmk") and shutil.which("pdfinfo"):
        compiled = subprocess.run(["latexmk", "-pdf", "-silent", "-halt-on-error",
            "-interaction=nonstopmode", "-outdir=build", "dmd_diversity.tex"], cwd=report,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert compiled.returncode == 0, compiled.stdout
        info = subprocess.check_output(["pdfinfo", str(report/"build/dmd_diversity.pdf")], text=True)
        assert int(next(line.split(":")[1] for line in info.splitlines() if line.startswith("Pages:"))) <= 4
    records[0]["continuation"]["parent_checkpoint_sha256"] = "0"*64
    (results/f"{records[0]['method']}.result.json").write_text(json.dumps(records[0]))
    with pytest.raises(ValueError, match="same parent checkpoint"):
        module.build(results, generated)
