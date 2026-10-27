"""Exercise the complete CVPR table/figure layout with synthetic temp-only fixtures."""

import base64
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from few_step_diffusion.diffusion import make_grid, nested_grid
from few_step_diffusion.reporting import write_report_inputs


@pytest.mark.skipif(not shutil.which("latexmk") or not shutil.which("pdfinfo"),
                    reason="Optional TeX/PDF tools are not installed")
@pytest.mark.parametrize("profile", ["cifar10", "sd15", "scale50k", "continued"])
def test_full_matrix_and_sample_figure_fit_four_pages(tmp_path, profile):
    sd = profile != "cifar10"
    destination = tmp_path / "synthetic-layout-test"
    shutil.copytree("report", destination, ignore=shutil.ignore_patterns("build", "generated", "assets"))
    main = destination / "main.tex"
    main.write_text(main.read_text().replace(
        "Noise-Aware Edge Regularization for Few-Step Diffusion Distillation",
        "SYNTHETIC TEST FIXTURES: Layout Verification"))
    template = json.loads(Path("examples/result.template.json").read_text())
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512) if sd else (32, 32), "blue").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    root = make_grid(1000, 64)
    records = []
    methods = ["teacher", "baseline", "edge_constant", "proposed", "swiftbrush_vsd"]
    if sd:
        methods += ["consistency", "dmd", "instaflow", "edge_weight_low", "edge_weight_high"]
    for method in methods:
        seeds = [0] if method == "teacher" else [42, 43, 44]
        steps_list = [1] if method in ("swiftbrush_vsd", "consistency", "dmd", "instaflow") else [32, 16, 8, 4, 2, 1]
        if method == "teacher":
            steps_list = [64]+steps_list
        for steps in steps_list:
            for seed in seeds:
                records.append({**template, "status": "complete", "method": method, "seed": seed,
                                "sampling_steps": steps, "nfe": steps*(2 if sd and method == "teacher" else 1), "fid": 20+seed/10,
                                "inception_score_mean": 7, "inception_score_std": 0.1,
                                "training_updates": 30000, "training_seconds": 3600,
                                "training_hardware": "TEST GPU",
                                "auxiliary_updates": 30000 if method in ("swiftbrush_vsd", "dmd") else 0,
                                "timesteps": nested_grid(root, steps).tolist(),
                                "sample_images_png_base64": [encoded]*8,
                                "environment": {"hardware": "TEST GPU"},
                                **({"dataset": "COCO-2017", "image_size": 512, "teacher_guidance_scale": 7.5,
                                    "num_samples": 1000, "num_real": 1000,
                                    "clip_score_mean": 25, "sample_prompts": ["TEST FIXTURE"]*8,
                                    **({"edge_weight": {"baseline": 0, "edge_constant": .1, "proposed": .1,
                                         "edge_weight_low": .03, "edge_weight_high": .3}[method],
                                        "edge_schedule": "constant" if method == "edge_constant" else "noise_aware"}
                                       if method in ("baseline", "edge_constant", "proposed", "edge_weight_low", "edge_weight_high")
                                       else {})} if sd else {})})
    from few_step_diffusion.config import load_config
    qualitative = []
    if sd:
        from few_step_diffusion.qualitative import panel_specs
        for method, steps in [(m, 1) for m in methods if m != "teacher"]+[("teacher", 64), ("teacher", 1)]:
            qualitative.append({"method": method, "sampling_steps": steps,
                "nfe": steps*(2 if method == "teacher" else 1), "seed": 0 if method == "teacher" else 42,
                "source_file": "SYNTHETIC TEST FIXTURE", "panels": {
                    kind: [{**item, "png_base64": encoded} for item in items]
                    for kind, items in panel_specs().items()}})
    study = load_config("configs/sd15_50k.yaml" if profile in ("scale50k", "continued") else "configs/sd15.yaml") if sd else None
    if profile == "continued":
        records = [r for r in records if r["method"] not in ("teacher", "swiftbrush_vsd") and r["sampling_steps"] == 1]
        for r in records:
            r.update(parent_training_updates=18000, cumulative_training_updates=36000,
                training_updates=18000, continuation={"fresh_training_prompt_count": 68287,
                    "parent_training_prompt_count": 50000, "training_image_overlap": 0})
        study["data"].update(train_offset=50000, train_limit=68287)
        study["train"]["updates_per_stage"] = 18000
    write_report_inputs(records, [], destination / "generated",
                        study_config=study,
                        qualitative_records=qualitative)
    result = subprocess.run(["latexmk", "-pdf", "-silent", "-halt-on-error",
                             "-interaction=nonstopmode", "-outdir=build", "main.tex"],
                            cwd=destination, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, result.stdout
    info = subprocess.check_output(["pdfinfo", str(destination / "build/main.pdf")], text=True)
    pages = int(next(line.split(":")[1].strip() for line in info.splitlines() if line.startswith("Pages:")))
    assert pages <= 4, f"Full matrix layout exceeds four pages: {pages}"
