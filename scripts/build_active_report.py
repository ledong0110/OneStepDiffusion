#!/usr/bin/env python3
"""Build the active 36k-step paper from verified full-validation result JSONs."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from few_step_diffusion.config import config_fingerprint
from few_step_diffusion.dmd_diversity import VARIANTS
from few_step_diffusion.figures import read_samples
from few_step_diffusion.reporting import collect_results, tex_escape
from few_step_diffusion.runtime import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"baseline": "Progressive distillation", "consistency": "Consistency distillation",
          "dmd": "DMD", "instaflow": "InstaFlow",
          "swiftbrush_lora": "SwiftBrush", "dmd_grouped": "Grouped DMD control",
          "dmd_perceptual": "Perceptual only", "dmd_diversity": "Diversity only",
          "dmd_perceptual_diversity": "Ours"}
SHORT = {"baseline": "PD", "consistency": "Consistency", "dmd": "DMD",
         "instaflow": "InstaFlow", "swiftbrush_lora": "SwiftBrush",
         "dmd_perceptual_diversity": "Ours", "teacher": "SD teacher"}
PRIMARY_METHODS = ("baseline", "consistency", "dmd", "instaflow", "swiftbrush_lora",
                   "dmd_perceptual_diversity")
COMMON = ("dataset", "image_size", "num_samples", "num_real", "reference_split",
          "eval_seed", "metric_implementation", "prompt_protocol", "evaluation_prompt_hash",
          "reference_manifest_hash", "teacher_identity", "clip_model_identity",
          "clip_score_definition")


def load_result(root, path):
    records, _ = collect_results(root, root/"schemas/result.schema.json", result_files=[path])
    if len(records) != 1:
        raise ValueError(f"Expected one completed research result: {path}")
    r = records[0]
    if r["num_samples"] != 5000 or r["num_real"] != 5000 or r["reference_split"] != "validation":
        raise ValueError(f"Full-validation report requires all 5,000 generated/reference images: {path}")
    if config_fingerprint(r["resolved_config"]) != r["config_hash"]:
        raise ValueError(f"Result/config identity mismatch: {path}")
    if r["method"] != "teacher":
        if (r["sampling_steps"] != 1 or r["nfe"] != 1 or r["lora_rank"] != 32 or
                r.get("effective_batch_size") != 16):
            raise ValueError("All active students must use one step, one NFE and LoRA rank 32")
        if r.get("cumulative_training_updates") != 36000 or r["seed"] != 42:
            raise ValueError("Only completed 36,000-step seed-42 students enter the main table")
        if r.get("edge_weight", 0) != 0:
            raise ValueError("Edge experiments are excluded from the active report")
        cfg, precision = (4.5, "fp32") if r["method"] == "swiftbrush_lora" else (7.5, "bf16")
        if (r["teacher_guidance_scale"], r["precision"]) != (cfg, precision):
            raise ValueError("Result does not match its declared CFG/precision cohort")
        adapter = root/r["resolved_config"]["output_dir"]/"stage_1/adapter.safetensors"
        if adapter.exists() and sha256(adapter) != r["model_identity"]:
            raise ValueError(f"Result/checkpoint identity mismatch: {path}")
    return r


def verify_common(records):
    if not records:
        raise ValueError("No full-validation results available")
    for key in COMMON:
        if any(r.get(key) != records[0].get(key) for r in records):
            raise ValueError(f"Incompatible full-validation protocols: {key}")
    students = [r for r in records if r["method"] != "teacher"]
    for key in ("distillation_prompt_hash", "parent_training_prompt_hash", "parent_training_updates"):
        if any(r.get(key) != students[0].get(key) for r in students):
            raise ValueError(f"Student training-data/budget mismatch: {key}")
    study = [r for r in records if r["method"] in VARIANTS]
    for key in ("teacher_pair_protocol_hash", "teacher_pair_weights_sha256",
                "dmd_study_parent_config_hash", "distillation_prompt_hash"):
        if any(r.get(key) != study[0].get(key) for r in study):
            raise ValueError(f"Factorial study mismatch: {key}")
    if len({r["continuation"]["parent_checkpoint_sha256"] for r in study}) > 1:
        raise ValueError("Factorial variants must share the same initialization checkpoint")


def verify_validation(root, record):
    manifest = root/"data/coco2017/val2017_5000.manifest.json"
    annotations = root/"data/coco2017/annotations/captions_val2017.json"
    if not manifest.exists() or not annotations.exists():
        return {"coverage": "5000/5000 according to native result metadata", "local_manifest_checked": False}
    rows = json.loads(manifest.read_text())["records"]
    annotation_ids = {i["id"] for i in json.loads(annotations.read_text())["images"]}
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if (len(rows) != 5000 or len(annotation_ids) != 5000 or
            {r["image_id"] for r in rows} != annotation_ids or
            digest != record["evaluation_prompt_hash"] or digest != record["reference_manifest_hash"]):
        raise ValueError("Saved evaluation does not cover the complete official validation split")
    return {"coverage": "5000/5000 official validation image IDs", "local_manifest_checked": True,
            "manifest_sha256": digest}


def draw_samples(records, indices, target, *, selected=True, extra_demo=None):
    import textwrap
    first = records[0]
    for r in records:
        for key in ("sample_prompts", "sample_image_ids", "sample_labels", "eval_seed", "evaluation_prompt_hash"):
            if r.get(key) != first.get(key):
                raise ValueError(f"Qualitative rows must share recorded prompts and noise: {key}")
    images = [read_samples(r) for r in records]
    prompts = list(first["sample_prompts"])
    image_ids = list(first["sample_image_ids"])
    if extra_demo:
        import base64
        import io
        from PIL import Image
        for column, r in enumerate(records):
            item = extra_demo["methods"][r["method"]]
            if item["model_identity"] != r["model_identity"]:
                raise ValueError("Custom fox sample differs from the evaluated checkpoint")
            im = Image.open(io.BytesIO(base64.b64decode(item["png_base64"], validate=True)))
            if im.size != (512, 512) or im.format != "PNG":
                raise ValueError("Custom demo requires an authentic 512x512 PNG")
            images[column].append(np.asarray(im.convert("RGB")))
        prompts.append(extra_demo["prompt"])
        image_ids.append("Custom prompt")
    if any(max(indices) >= len(items) for items in images):
        raise ValueError("Requested demo index is absent from the authentic portable samples")
    n = len(records)
    side, label, gap, header = 1.03, 1.36, .055, .44
    width, height = label+n*(side+gap), header+len(indices)*(side+gap)
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(width, height))
        grid = fig.add_gridspec(len(indices)+1, n+1, width_ratios=[label]+[side]*n,
            height_ratios=[header]+[side]*len(indices), left=.002, right=.998,
            top=.998, bottom=.002, wspace=.055, hspace=.055)
        for column, r in enumerate(records):
            ax = fig.add_subplot(grid[0, column+1])
            ax.axis("off")
            name = SHORT.get(r["method"], LABELS.get(r["method"], r["method"]))
            ax.text(.5, .76, name, ha="center", va="center", fontsize=9,
                    weight="bold" if r["method"] == "dmd_perceptual_diversity" else "normal")
            detail = f"{r['sampling_steps']} step{'s' if r['sampling_steps'] > 1 else ''} / {r['nfe']} NFE"
            ax.text(.5, .39, detail, ha="center", va="center", fontsize=6.9)
            if r["method"] == "swiftbrush_lora":
                ax.text(.5, .06, "CFG 4.5 / FP32", ha="center", va="center", fontsize=6.2, color="#555555")
            for row, index in enumerate(indices):
                if column == 0:
                    prompt = prompts[index]
                    ax = fig.add_subplot(grid[row+1, 0])
                    ax.axis("off")
                    ax.text(0, .55, textwrap.fill(prompt, 28 if index == 8 else 22), fontsize=5.8 if index == 8 else 7.4,
                            ha="left", va="center")
                    label = (first["sample_labels"][index] if first.get("sample_labels") else
                             f"COCO {image_ids[index]}" if index < 8 else "Custom prompt")
                    ax.text(0, .08, label, fontsize=6.4, color="#666666")
                ax = fig.add_subplot(grid[row+1, column+1])
                ax.axis("off")
                ax.imshow(np.asarray(images[column][index]), interpolation="nearest")
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.015)
        fig.savefig(target.with_suffix(".png"), dpi=170, bbox_inches="tight", pad_inches=.015)
        plt.close(fig)
    return {"selection": "manual qualitative selection" if selected else "all eight saved evaluation outputs",
            "sample_indices": indices, "image_ids": [image_ids[i] for i in indices],
            "prompts": [prompts[i] for i in indices], "eval_seed": first["eval_seed"],
            "custom_prompt_noise_seed": extra_demo["noise_seed"] if extra_demo else None,
            "methods": [{"method": r["method"], "source_file": r["source_file"],
                         "model_identity": r["model_identity"]} for r in records]}


def load_main_comparison(root, records, path):
    """Bind each user-selected row to its evaluated model and shared prompt/noise."""
    payload = json.loads((root/path).read_text())
    rows = payload["rows"]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if (payload.get("protocol") != "matched-main-figure-v1" or payload.get("image_size") != 512 or
            payload["protocol_hash"] != digest or not rows or
            len({row["key"] for row in rows}) != len(rows)):
        raise ValueError("Invalid selected main-comparison protocol")
    comparisons, pending = [], []
    for record in records:
        method = record["method"]
        item = payload["methods"].get(method)
        if item is None:
            pending.append(method)
            continue
        for key in ("model_identity", "sampling_steps", "nfe", "precision"):
            if item.get(key) != record[key]:
                raise ValueError(f"Main comparison differs from its evaluated checkpoint: {key}")
        if method != "teacher" and item.get("cumulative_training_updates") != 36000:
            raise ValueError("Main comparison requires completed 36,000-step students")
        if payload["eval_seed"] != record["eval_seed"]:
            raise ValueError("Main comparison uses a different noise stream")
        images = item["rows"]
        if [{k:v for k,v in image.items() if k != "png_base64"} for image in images] != rows:
            raise ValueError("Main comparison row prompts or noise differ across methods")
        for row, image in zip(rows, images):
            if row["source"] == "validation":
                index = row["noise_stream_index"]
                if (record["sample_prompts"][index] != row["prompt"] or
                        record["sample_images_png_base64"][index] != image["png_base64"]):
                    raise ValueError("Saved validation sample changed in the main comparison")
        comparison = {**record, "sample_prompts": [r["prompt"] for r in rows],
            "sample_image_ids": [r["key"] for r in rows], "sample_labels": [r["label"] for r in rows],
            "sample_images_png_base64": [image["png_base64"] for image in images],
            "evaluation_prompt_hash": digest}
        if len(read_samples(comparison)) != len(rows):
            raise ValueError("Main comparison requires native 512x512 PNGs for every row")
        comparisons.append(comparison)
    if not {"teacher", "dmd_perceptual_diversity"}.issubset({r["method"] for r in comparisons}):
        raise ValueError("Main comparison is waiting for the proposal and teacher images")
    return comparisons, {"source_file": str(path), "rows": rows,
                         "protocol_hash": digest, "pending_methods": pending}


def draw_prompt_control(root, record, out):
    from few_step_diffusion.qualitative import ACTIONS, ANIMALS, load_qualitative
    import base64
    import io
    from PIL import Image

    selection = json.loads((root/"configs/report_selection.json").read_text())
    eval_seed = selection.get("prompt_control_eval_seed", 12345)
    folder = root/"results/sd15/active_one_step/qualitative"/f"seed{eval_seed}"
    target = out/"prompt_control.tex"
    if not folder.exists():
        target.write_text("% New-proposal prompt-control samples are pending.\n")
        return None
    imported = load_qualitative(folder, [record])
    match = next((r for r in imported if r["method"] == record["method"]), None)
    if not match or match.get("cumulative_training_updates") != 36000:
        raise ValueError("Prompt-control figure requires the evaluated 36k proposal")
    items = match["panels"]["control"]
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(4.8, 3.6))
        grid = fig.add_gridspec(4, 5, width_ratios=[.55, 1, 1, 1, 1],
            height_ratios=[.22, 1, 1, 1], left=.002, right=.998, top=.998,
            bottom=.002, wspace=.025, hspace=.025)
        for column, animal in enumerate(ANIMALS):
            ax = fig.add_subplot(grid[0, column+1])
            ax.axis("off")
            ax.text(.5, .5, animal.capitalize(), fontsize=9, ha="center", va="center")
            for row, action in enumerate(ACTIONS):
                if column == 0:
                    ax = fig.add_subplot(grid[row+1, 0])
                    ax.axis("off")
                    ax.text(.9, .5, ("Reading\na book", "Drinking\na latte", "Eating\npizza")[row],
                            fontsize=8, ha="right", va="center")
                item = items[row*4+column]
                pixels = Image.open(io.BytesIO(base64.b64decode(item["png_base64"], validate=True))).convert("RGB")
                ax = fig.add_subplot(grid[row+1, column+1])
                ax.axis("off")
                ax.imshow(pixels, interpolation="nearest")
        for ext in ("pdf", "png"):
            fig.savefig(out/f"prompt_control.{ext}", dpi=180, bbox_inches="tight", pad_inches=.01)
        plt.close(fig)
    target.write_text("\\begin{figure}[t]\n\\centering\n"
        "\\includegraphics[width=\\linewidth]{generated/active_study/prompt_control.pdf}\n"
        "\\caption{Selected compositional prompt-control grid from the 36,000-step proposal "
        "at one step / one NFE. Prompts follow `A DSLR photograph of a \\{animal\\} "
        "\\{action\\}, sitting at an outdoor cafe table.' All cells share initial noise "
        f"seed {items[0]['noise_seed']}. The displayed grid is manually selected for visual quality "
        "and animal/action adherence from a recorded candidate seed pool. This is discrete "
        "prompt composition.}\\label{fig:control}\n\\end{figure}\n")
    return {"source_file": match["source_file"], "model_identity": match["model_identity"],
            "eval_seed": eval_seed, "noise_seed": items[0]["noise_seed"],
            "selection_policy": selection.get("prompt_control_selection_policy"),
            "prompts": [{k:v for k,v in i.items() if k != "png_base64"} for i in items]}


def draw_interpolation(root, record, out):
    import base64
    import io
    from PIL import Image
    path = root/"results/sd15/active_one_step/prompt_interpolation.json"
    target = out/"interpolation_figure.tex"
    if not path.exists():
        target.write_text("% Text-embedding interpolation samples are pending.\n")
        return None
    payload = json.loads(path.read_text())
    weights = [0, .25, .5, .75, 1]
    if (payload["model_identity"] != record["model_identity"] or payload["weights"] != weights or
            payload["protocol"] != "linear-clip-hidden-state-v1" or
            payload.get("cumulative_training_updates") != 36000 or payload["nfe"] != 1):
        raise ValueError("Interpolation checkpoint/protocol differs from the evaluated proposal")
    selection = json.loads((root/"configs/report_selection.json").read_text())
    seeds = selection.get("interpolation_noise_seeds", [12546, 12548])
    rows = {row["noise_seed"]: row for row in payload["rows"]}
    if any(seed not in rows for seed in seeds):
        target.write_text("% Selected interpolation rows are still generating.\n")
        return None
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(5.2, .5+len(seeds)))
        grid = fig.add_gridspec(len(seeds)+1, 5, height_ratios=[.42]+[1]*len(seeds),
            left=.003, right=.997, top=.998, bottom=.002, wspace=.02, hspace=.02)
        ax = fig.add_subplot(grid[0, :])
        ax.axis("off")
        ax.text(.02, .72, "Dog", fontsize=12, ha="left", va="center")
        ax.text(.98, .72, "Cat", fontsize=12, ha="right", va="center")
        ax.annotate("", xy=(.83,.72), xytext=(.17,.72), arrowprops={"arrowstyle": "->", "lw": 1.5})
        for col, weight in enumerate(weights):
            ax.text((col+.5)/5, .1, f"{weight:g}", fontsize=8, ha="center", va="center")
        for row, seed in enumerate(seeds):
            items = rows[seed]["images"]
            if [i["weight"] for i in items] != weights:
                raise ValueError("Interpolation image order differs from its embedding weights")
            for col, item in enumerate(items):
                image = Image.open(io.BytesIO(base64.b64decode(item["png_base64"], validate=True)))
                if image.size != (512,512) or image.format != "PNG":
                    raise ValueError("Interpolation requires native 512x512 PNGs")
                ax = fig.add_subplot(grid[row+1, col])
                ax.axis("off")
                ax.imshow(image, interpolation="nearest")
        for ext in ("pdf", "png"):
            fig.savefig(out/f"prompt_interpolation.{ext}", dpi=180, bbox_inches="tight", pad_inches=.01)
        plt.close(fig)
    target.write_text("\\begin{figure}[t]\n\\centering\n"
        "\\includegraphics[width=\\linewidth]{generated/active_study/prompt_interpolation.pdf}\n"
        "\\caption{Text-embedding interpolation from dog to cat with the 36,000-step proposal "
        "at one step / one NFE. Endpoint prompts use `A DSLR photograph of a \\{animal\\} "
        "reading an open book, sitting at an outdoor cafe table.' Columns linearly interpolate "
        "the CLIP hidden states at the labelled weights; each row keeps the same initial noise. "
        "Two rows are selected from three candidate seeds.}\\label{fig:interpolation}\n\\end{figure}\n")
    return {"source_file": str(path), "noise_seeds": seeds, "weights": weights,
            "prompts": payload["prompts"], "model_identity": payload["model_identity"]}


def quality_table(records, teacher):
    displayed = [records[m] for m in PRIMARY_METHODS if m in records]
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Method (36k steps) & NFE & FID $\downarrow$ & IS $\uparrow$ & CLIP $\uparrow$ \\", r"\midrule"]
    for method in PRIMARY_METHODS:
        record = records.get(method)
        if record:
            values = []
            for key in ("fid", "inception_score_mean", "clip_score_mean"):
                best = (min if key == "fid" else max)(r[key] for r in displayed)
                value = f"{record[key]:.2f}"
                values.append(r"\textbf{"+value+"}" if record[key] == best else value)
            rows.append(f"{LABELS[method]} & 1 & "+" & ".join(values)+r" \\")
        else:
            rows.append(f"{LABELS[method]} & 1 & "+r"\multicolumn{3}{c}{\emph{Pending}} \\")
    rows += [r"\midrule", r"\multicolumn{5}{l}{\emph{Teacher reference: 64 steps, CFG 7.5}} \\",
        f"SD1.5 teacher & 128 & {teacher['fid']:.2f} & {teacher['inception_score_mean']:.2f} & {teacher['clip_score_mean']:.2f}"+r" \\",
        r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)+"\n"


def measured_findings(records):
    full = records["dmd_perceptual_diversity"]
    measured = (f"The full proposal obtains FID {full['fid']:.2f}, IS {full['inception_score_mean']:.2f} "
                f"and CLIP {full['clip_score_mean']:.2f} at one NFE after 36,000 total training steps. ")
    dmd = records.get("dmd")
    if dmd:
        delta = dmd["fid"]-full["fid"]
        measured += (f"Its FID is {delta:.2f} lower ({100*delta/dmd['fid']:.1f} percent) than the DMD baseline "
            "at the same total step budget. This comparison changes both the regression objective and anchor grouping. ")
    control = records.get("dmd_grouped")
    if control:
        measured += f"Against the matched grouped control, the FID difference is {full['fid']-control['fid']:+.2f}. "
    else:
        measured += "The grouped control and loss ablations are unfinished, so the individual loss contributions remain unresolved. "
    perceptual = records.get("dmd_perceptual")
    if perceptual:
        measured += f"Perceptual regression alone obtains FID {perceptual['fid']:.2f}. "
        if "diversity_lpips_alex_mean" in perceptual:
            measured += (f"Adding diversity matching changes FID by {full['fid']-perceptual['fid']:+.2f}, "
                f"while mean pairwise LPIPS-Alex distance changes from {perceptual['diversity_lpips_alex_mean']:.4f} "
                f"to {full['diversity_lpips_alex_mean']:.4f}. ")
            if full["fid"] > perceptual["fid"]:
                measured += "The added term does not improve FID in this paired comparison. "
    diversity = records.get("dmd_diversity")
    if control and diversity and all("diversity_lpips_alex_mean" in r for r in (control, diversity)):
        measured += (f"With latent regression, diversity matching changes FID by {diversity['fid']-control['fid']:+.2f} "
            f"and pairwise LPIPS from {control['diversity_lpips_alex_mean']:.4f} "
            f"to {diversity['diversity_lpips_alex_mean']:.4f}. ")
    measured += "A larger perceptual distance alone does not establish higher visual quality or semantic diversity."
    return measured


def build(root=ROOT, *, study=None, selection=None):
    root = Path(root)
    plan = json.loads((study or root/"configs/active_study.json").read_text())
    selected = json.loads((selection or root/"configs/report_selection.json").read_text())
    out = root/"report/generated/active_study"
    out.mkdir(parents=True, exist_ok=True)
    jobs = {j["method"]: j for j in plan["jobs"]}
    if set(jobs) != set(LABELS):
        raise ValueError("Active report must contain exactly the proposal, three ablations and five baselines")
    records, pending = {}, []
    for method, job in jobs.items():
        path = root/job["result_dir"]/f"{method}_seed42_1steps.result.json"
        if path.exists():
            r = load_result(root, path)
            if r["method"] != method:
                raise ValueError("Result method does not match its selected experiment")
            records[method] = r
        else:
            pending.append(method)
    teacher = load_result(root, root/"results/sd15/scale50k/teacher_seed0_64steps.result.json")
    verify_common([*records.values(), teacher])
    coverage = verify_validation(root, teacher)
    full = records.get("dmd_perceptual_diversity")
    if not full:
        raise ValueError("The primary paper requires a completed full proposal")
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%d %B %Y, %H:%M SGT")
    complete = len(records)
    status = (f"Results updated {timestamp}. {complete} of 9 planned student experiments have completed "
              "the full 36,000-step budget and full-validation quality evaluation. "
              "Pending cells are withheld until the final checkpoint has a verified 5,000-image evaluation.")
    (out/"status.tex").write_text(tex_escape(status)+"\n")
    summary = []
    (out/"quality.tex").write_text(quality_table(records, teacher))
    rows = [r"\begin{tabular}{llrrr}", r"\toprule",
            r"Variant & Regression & $\lambda_d$ & FID $\downarrow$ & Diversity \\", r"\midrule"]
    for method, (regression, weight) in VARIANTS.items():
        r = records.get(method)
        prefix = f"{LABELS[method]} & {'LPIPS' if regression == 'lpips_vgg' else 'Latent MSE'} & {weight:g} & "
        fid = f"{r['fid']:.2f}" if r else ""
        if r and r["fid"] == min(records[m]["fid"] for m in VARIANTS if m in records):
            fid = r"\textbf{"+fid+"}"
        values = (f"{fid} & {r['diversity_lpips_alex_mean']:.4f}"
                  if r and "diversity_lpips_alex_mean" in r else r"\multicolumn{2}{c}{\emph{Pending}}")
        rows.append(prefix+values+r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (out/"ablation.tex").write_text("\n".join(rows)+"\n")
    dmd = records.get("dmd")
    (out/"measured.tex").write_text(tex_escape(measured_findings(records))+"\n")
    gallery_records = [{**records[m]} if m != "teacher" else teacher
                       for m in selected["method_order"] if m in records or m == "teacher"]
    # Add newly finished PD/InstaFlow automatically to both grids at the same prompts/noise.
    for method in ("baseline", "instaflow"):
        if method in records and method not in selected["method_order"]:
            gallery_records.insert(0, records[method])
    indices = selected["sample_indices"]
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= 8 for i in indices):
        raise ValueError("Demo indices must be unique entries from the eight recorded samples")
    fox = None
    custom_path = selected.get("custom_demo_result", selected.get("fox_demo_result"))
    if selected.get("main_comparison_result"):
        comparisons, protocol = load_main_comparison(root, gallery_records, selected["main_comparison_result"])
        figure = draw_samples(comparisons, list(range(len(protocol["rows"]))), out/"selected_samples")
        figure.update(protocol)
        (out/"custom_figure.tex").write_text("% Main comparison contains the selected custom prompts.\n")
    elif custom_path:
        fox = json.loads((root/custom_path).read_text())
        demo_records = [r for r in gallery_records if r["method"] in fox["methods"]]
        figure = draw_samples(demo_records, indices+[8], out/"selected_samples", extra_demo=fox)
        draw_samples(demo_records, [8], out/"custom_comparison", extra_demo=fox)
        (out/"custom_figure.tex").write_text("\\begin{figure*}[t]\n\\centering\n"
            "\\includegraphics[width=\\textwidth]{generated/active_study/custom_comparison.pdf}\n"
            "\\caption{Selected custom prompt. Every method uses the exact same recorded "
            "prompt and initial noise. Selection is based on the proposal output; all other "
            "methods are generated without individual seed selection.}\\end{figure*}\n")
    else:
        figure = draw_samples(gallery_records, indices, out/"selected_samples")
        (out/"custom_figure.tex").write_text("% No custom comparison selected.\n")
    gallery = draw_samples(gallery_records, list(range(8)), out/"all_samples", selected=False)
    draw_samples(gallery_records, list(range(4)), out/"all_samples_first4", selected=False)
    draw_samples(gallery_records, list(range(4, 8)), out/"all_samples_last4", selected=False)
    failures = draw_samples(gallery_records, selected["failure_indices"], out/"failure_samples")
    prompt_control = draw_prompt_control(root, full, out)
    interpolation = draw_interpolation(root, full, out)
    figure["selection_policy"] = selected["selection_policy"]
    for method, job in jobs.items():
        r = records.get(method)
        summary.append({"method": method, "total_training_steps": 36000, "nfe": 1,
            "status": "complete" if r else "pending", "num_samples": r["num_samples"] if r else "",
            "fid": r["fid"] if r else "", "is": r["inception_score_mean"] if r else "",
            "clip": r["clip_score_mean"] if r else "", "teacher_cfg": 4.5 if method == "swiftbrush_lora" else 7.5,
            "precision": "fp32" if method == "swiftbrush_lora" else "bf16",
            "source_file": r["source_file"] if r else str(Path(job["result_dir"])/f"{method}_seed42_1steps.result.json")})
    with (out/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    manifest = {"updated_at": timestamp, "validation": coverage, "training_steps": 36000,
        "completed": complete, "planned": 9, "pending_methods": pending,
        "results": [{k:v for k,v in r.items() if k != "sample_images_png_base64"}
                    for r in [*records.values(), teacher]],
        "selected_figure": figure, "all_samples_figure": gallery, "failure_figure": failures,
        "prompt_control_figure": prompt_control, "interpolation_figure": interpolation}
    atomic_json(out/"manifest.json", manifest)
    commands = {"ActiveReportDate": timestamp, "ActiveFID": f"{full['fid']:.2f}",
                "ActiveCLIP": f"{full['clip_score_mean']:.2f}",
                "BaselineDMDFID": f"{dmd['fid']:.2f}" if dmd else "pending",
                "BaselineDMDCLIP": f"{dmd['clip_score_mean']:.2f}" if dmd else "pending"}
    ablations_done = all(m in records and "diversity_lpips_alex_mean" in records[m] for m in VARIANTS)
    commands["ActiveAblationNote"] = (
        "The factorial controls are complete; the single training seed limits the strength of attribution."
        if ablations_done else
        "The factorial controls are unfinished, so individual loss contributions remain unresolved.")
    if ablations_done and records["dmd_perceptual"]["fid"] < full["fid"]:
        commands["ActiveAblationNote"] = (
            "The perceptual-only ablation has lower FID; adding diversity matching trades FID "
            "for a small increase in pairwise perceptual distance.")
    (out/"settings.tex").write_text("\n".join(
        f"\\newcommand{{\\{key}}}{{{tex_escape(value)}}}" for key, value in commands.items())+"\n")
    return manifest


def compile_report(root, max_pages=4):
    for name in ("main", "qualitative"):
        subprocess.run(["latexmk", "-pdf", "-silent", "-halt-on-error", "-interaction=nonstopmode",
                        "-outdir=build", name+".tex"], cwd=root/"report", check=True)
    info = subprocess.check_output(["pdfinfo", str(root/"report/build/main.pdf")], text=True)
    pages = int(next(line.split(":")[1] for line in info.splitlines() if line.startswith("Pages:")))
    if pages > max_pages:
        raise ValueError(f"Report has {pages} pages, exceeding the {max_pages}-page project limit")
    published = root/"main_report.pdf"
    temporary = root/".main_report.pdf.tmp"
    try:
        shutil.copyfile(root/"report/build/main.pdf", temporary)
        os.replace(temporary, published)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Report: {published} ({pages}/{max_pages} pages)", flush=True)


def input_signature(root):
    plan = json.loads((root/"configs/active_study.json").read_text())
    paths = [root/"configs/active_study.json", root/"configs/report_selection.json",
             root/"report/active_study.tex", root/"report/qualitative.tex", root/"report/references.bib",
             root/"report/preamble.tex", root/"report/cvpr.sty", root/"report/ieeenat_fullname.bst",
             root/"results/sd15/scale50k/teacher_seed0_64steps.result.json"]
    paths += [root/j["result_dir"]/f"{j['method']}_seed42_1steps.result.json" for j in plan["jobs"]]
    paths += list((root/"results/sd15/active_one_step/qualitative").rglob("*.qualitative.json"))
    selection = json.loads((root/"configs/report_selection.json").read_text())
    if selection.get("custom_demo_result"):
        paths.append(root/selection["custom_demo_result"])
    if selection.get("main_comparison_result"):
        paths.append(root/selection["main_comparison_result"])
    paths.append(root/"results/sd15/active_one_step/prompt_interpolation.json")
    signature = []
    for path in sorted(set(paths)):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    return signature


def start_background(args):
    """Start one detached CPU watcher, reusing an existing verified process."""
    import fcntl
    from launch_scale50k import process_matches, process_record

    folder = ROOT/"report/build"
    folder.mkdir(parents=True, exist_ok=True)
    process_path = folder/"active_report_process.json"
    # Serialize launch requests; the worker also holds its own lifetime lock.
    with (folder/"active_report_start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            record = json.loads(process_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            record = None
        if record and process_matches(record):
            return {"status": "already_running", "pid": record["pid"]}
        with (folder/"active_report_watch.lock").open("a") as watcher_lock:
            fcntl.flock(watcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        command = [sys.executable, str(Path(__file__).resolve()), "--watch",
                   "--interval", str(args.interval), "--max-pages", str(args.max_pages)]
        if args.no_compile:
            command.append("--no-compile")
        log_path = folder/"active_report_watch.log"
        with log_path.open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
                         OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        atomic_json(process_path, process_record(child.pid))
        return {"status": "started", "pid": child.pid, "log": str(log_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--watch", action="store_true", help="Refresh automatically when final result JSONs change")
    parser.add_argument("--background", action="store_true",
                        help="Start or reuse a detached CPU watcher (implies --watch)")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args(argv)
    if args.background:
        print(json.dumps(start_background(args)), flush=True)
        return
    state_path = ROOT/"report/build/active_report_status.json"
    last = None
    # A separate watcher lock permits ordinary manual builds between refreshes.
    from contextlib import ExitStack
    import fcntl
    ROOT.joinpath("report/build").mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if args.watch:
            watcher_lock = stack.enter_context((ROOT/"report/build/active_report_watch.lock").open("a"))
            fcntl.flock(watcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from launch_scale50k import process_record
            atomic_json(ROOT/"report/build/active_report_process.json", process_record(os.getpid()))
        while True:
            try:
                signature = input_signature(ROOT)
                if signature != last:
                    with (ROOT/"report/build/active_report.lock").open("a") as lock:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        manifest = build()
                        if not args.no_compile:
                            compile_report(ROOT, args.max_pages)
                    atomic_json(state_path, {"status": "watching" if args.watch else "complete",
                        "pid": os.getpid(), "updated_at": manifest["updated_at"],
                        "completed_experiments": manifest["completed"], "planned_experiments": 9,
                        "pdf": "main_report.pdf", "build_pdf": "report/build/main.pdf"})
                    last = signature
            except Exception as exc:
                atomic_json(state_path, {"status": "error", "pid": os.getpid(), "error": str(exc)})
                if not args.watch:
                    raise
                print(f"Report update failed; will retry: {exc}", flush=True)
            if not args.watch:
                return
            time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
