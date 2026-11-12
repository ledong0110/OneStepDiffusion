#!/usr/bin/env python3
"""Build the separate CE6190 Assignment 2 report from completed native results.

The CE7454 paper refreshes separately after the shared assets are verified.
--compile-only uses saved tables/figures without data or checkpoints.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from build_active_report import (
    LABELS, ROOT, draw_interpolation, draw_prompt_control,
    draw_samples, load_main_comparison, load_result, quality_table, verify_common,
    verify_validation,
)
import matplotlib.pyplot as plt
from few_step_diffusion.dmd_response import MODES as RESPONSE_MODES, SETTINGS as RESPONSE_SETTINGS

VARIANTS = ("dmd_grouped", "dmd_perceptual", "dmd_diversity", "dmd_perceptual_diversity")
GENERATED = Path("report/generated/assignment2")
ASSIGNMENT_LABELS = {**LABELS, "dmd": "DMD (latent-MSE control)",
    "dmd_perceptual": "DMD + LPIPS", "dmd_grouped": "Grouped latent-MSE control",
    "dmd_diversity": "Latent MSE + distance", "dmd_perceptual_diversity": "DMD + distance"}
ASSIGNMENT_METHODS = ("baseline", "consistency", "dmd_perceptual", "instaflow",
                      "swiftbrush_lora", "dmd_perceptual_diversity")
FIGURE_LABELS = {"dmd_perceptual": "DMD + LPIPS", "dmd": "DMD-MSE",
                 "dmd_perceptual_diversity": "DMD + distance",
                 "dmd_noise_agreement": "Spatial agreement",
                 "dmd_noise_mean": "Paired mean", "dmd_noise_shifted": "Shifted map",
                 "dmd_pair_agreement": "Pair agreement", "dmd_pair_mean": "Four-probe mean",
                 "dmd_pair_shifted": "Shifted pairs"}
FIGURE_LABELS.update({method: f"{RESPONSE_SETTINGS[key][0].capitalize()} {RESPONSE_SETTINGS[key][1]:.0%}"
                     for key, method in RESPONSE_MODES.items()})
STANDARD_METHODS = ("baseline", "consistency", "dmd_perceptual", "swiftbrush_lora", "instaflow")


def verify_report_budget(records):
    """Reject pilot checkpoints rather than presenting them as 36k experiments."""
    for record in records:
        if record["method"] != "teacher" and record.get("cumulative_training_updates") != 36000:
            raise ValueError(f"Report evidence requires exactly 36,000 cumulative student updates: {record['method']}")


def write_independent_comparison(root, out):
    """Render only terminal 36k results; never reuse the legacy pilot narrative."""
    from few_step_diffusion.independent_response import validate_result
    records = []
    for method in ("dmd_lpips_scratch", "response_scratch"):
        path = root/f"results/sd15/independent36k/{method}_seed42_1steps.result.json"
        if path.exists():
            record = load_result(root, path)
            validate_result(record)
            records.append(record)
    verify_report_budget(records)
    rows = [r"\section{Independent 36,000-update comparison}",
        r"Both recipes start from pretrained SD1.5 and fresh rank-32 student and scorer adapters. "
        r"Each uses 18,000 updates per caption pool, identical phase-specific teacher banks, "
        r"and the same LPIPS regression. The response coefficient is fixed at 0.298494 "
        r"from the first update; the zero-response control sets it to zero. "
        r"Only completed 36,000-update models with full validation are included.",
        r"\begin{table}[t]\centering\small\begin{tabular}{lrrr}\toprule",
        r"Independent run & FID $\downarrow$ & IS $\uparrow$ & CLIP $\uparrow$ \\", r"\midrule"]
    for record in records:
        label = "Direction alignment" if record["method"] == "response_scratch" else "Without response term"
        rows.append(label+" & "+" & ".join(f"{record[k]:.2f}" for k in
            ("fid", "inception_score_mean", "clip_score_mean"))+r" \\")
    rows += [r"\bottomrule\end{tabular}",
             r"\caption{Independent training at one NFE; 5,000 validation samples and references.}\end{table}"]
    if len(records) < 2:
        rows.append("The zero-response control's final evaluation is pending; no quality score is assigned to it.")
    else:
        control, proposal = records
        rows.append(f"With all 36,000 updates and the initialization held fixed, FID changes from "
            f"{control['fid']:.2f} without response alignment to {proposal['fid']:.2f} with it. "
            "This is a single-training-seed comparison; repeated seeds are needed to assess reliability.")
    (out/"independent_results.tex").write_text("\n".join(rows)+"\n")
    return records


def is_direction_candidate(method):
    return method == "response_scratch" or method.startswith("dmd_response_angular_")


def ablation_table(records):
    rows = [r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"Variant & Regression & $\lambda_d$ & FID $\downarrow$ & IS $\uparrow$ & CLIP $\uparrow$ & Pairwise LPIPS \\",
            r"\midrule"]
    for method, loss, weight in zip(VARIANTS, ("Latent MSE", "LPIPS", "Latent MSE", "LPIPS"),
                                    (0, 0, .1, .1), strict=True):
        record = records[method]
        values = []
        for key in ("fid", "inception_score_mean", "clip_score_mean"):
            best = (min if key == "fid" else max)(records[m][key] for m in VARIANTS)
            value = f"{record[key]:.2f}"
            values.append(r"\textbf{"+value+"}" if record[key] == best else value)
        rows.append(f"{ASSIGNMENT_LABELS[method]} & {loss} & {weight:g} & "+" & ".join(values)+
                    f" & {record['diversity_lpips_alex_mean']:.4f}"+r" \\")
    return "\n".join(rows+[r"\bottomrule", r"\end{tabular}"])+"\n"


def sensitivity_figure(records, target):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.55), layout="constrained")
        for names, label, color, marker in (
            (("dmd_grouped", "dmd_diversity"), "Latent MSE", "#58758c", "o"),
            (("dmd_perceptual", "dmd_perceptual_diversity"), "LPIPS regression", "#b15a37", "s"),
        ):
            for ax, metric in zip(axes, ("fid", "diversity_lpips_alex_mean"), strict=True):
                values = [records[m][metric] for m in names]
                ax.plot([0, .1], values, color=color, marker=marker, label=label, lw=1.5)
                for x, value in zip((0, .1), values, strict=True):
                    ax.annotate(f"{value:.2f}" if metric == "fid" else f"{value:.4f}",
                                (x, value), xytext=(0, 7), textcoords="offset points",
                                ha="center", color=color, fontsize=8)
        for ax, title, limits in zip(axes, ("FID (lower is better)", "Within-prompt LPIPS-Alex"),
                                    ((28.7, 35.7), (.703, .735)), strict=True):
            ax.set(title=title, xlabel=r"Diversity weight $\lambda_d$", xlim=(-.025, .125), ylim=limits)
            ax.set_xticks([0, .1], ["0", "0.1"])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
        axes[0].legend(loc="center right", frameon=False, fontsize=8)
        for ext in ("pdf", "png"):
            fig.savefig(target.with_suffix("."+ext), dpi=180)
        plt.close(fig)


def write_settings(out, group_id, records=None):
    (out/"group.tex").write_text(f"\\newcommand{{\\AssignmentGroup}}{{{group_id}}}\n")
    if records is None:
        return
    full, dmd, control, perceptual, diversity = [records[m] for m in
        ("dmd_perceptual_diversity", "dmd_perceptual", "dmd_grouped", "dmd_perceptual", "dmd_diversity")]
    values = {"AFullFID": full["fid"], "AFullIS": full["inception_score_mean"],
              "AFullCLIP": full["clip_score_mean"], "ADMDFID": dmd["fid"],
              "AControlFID": control["fid"], "APerceptualFID": perceptual["fid"],
              "ALatentDMDFID": records["dmd"]["fid"],
              "AControlGain": control["fid"]-full["fid"],
              "ARegressionGain": control["fid"]-perceptual["fid"],
              "ADiversityCost": full["fid"]-perceptual["fid"],
              "ALatentDiversityGain": control["fid"]-diversity["fid"]}
    commands = {key: f"{value:.2f}" for key, value in values.items()}
    for key, record in (("AFullDistance", full), ("APerceptualDistance", perceptual),
                        ("AControlDistance", control), ("ADiversityDistance", diversity)):
        commands[key] = f"{record['diversity_lpips_alex_mean']:.4f}"
    (out/"settings.tex").write_text("\n".join(
        f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in commands.items())+"\n")


def coefficient_sweep(root, records, out):
    measured = {0.: records["dmd_perceptual"], .1: records["dmd_perceptual_diversity"]}
    plan_path = root/"configs/dmd_weight_sweep.json"
    jobs = json.loads(plan_path.read_text())["jobs"] if plan_path.exists() else []
    pending = []
    for job in jobs:
        path = root/job["result_dir"]/"dmd_perceptual_diversity_seed42_1steps.result.json"
        if not path.exists():
            pending.append(job["weight"])
            continue
        record = load_result(root, path)
        if (record["config_hash"] != job["config_hash"] or
                record.get("dmd_diversity_weight") != job["weight"] or
                record.get("dmd_regression_type") != "lpips_vgg" or
                record.get("dmd_regression_weight") != .25):
            raise ValueError("Coefficient sweep result differs from its scheduled setting")
        verify_common([records["dmd_perceptual"], record])
        if "diversity_lpips_alex_mean" not in record:
            pending.append(job["weight"])
            continue
        measured[job["weight"]] = record
    weights = sorted(set(measured) | set(pending))
    best = min(measured, key=lambda weight: measured[weight]["fid"])
    rows = [r"\begin{tabular}{rrrrr}", r"\toprule",
            r"$\lambda_d$ & FID $\downarrow$ & IS $\uparrow$ & CLIP $\uparrow$ & Pairwise LPIPS \\",
            r"\midrule"]
    for weight in weights:
        if weight not in measured:
            rows.append(f"{weight:g} & "+r"\multicolumn{4}{c}{\emph{Pending evaluation}} \\")
        else:
            r = measured[weight]
            fid = f"{r['fid']:.2f}"
            if weight == best:
                fid = r"\textbf{"+fid+"}"
            rows.append(f"{weight:g} & {fid} & {r['inception_score_mean']:.2f} & "
                        f"{r['clip_score_mean']:.2f} & {r['diversity_lpips_alex_mean']:.4f}"+r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (out/"weight_sweep.tex").write_text("\n".join(rows)+"\n")
    findings = ("The expanded comparison holds LPIPS regression at weight 0.25 and varies only "
            "the diversity coefficient. All settings use the same initialization, teacher-pair "
            "bank, seed and 36,000-update total budget. ")
    if pending:
        findings += ("Additional coefficients "+", ".join(f"{w:g}" for w in sorted(pending))+
                 " are unfinished and the sweep is paused; no quality values or coefficient-selection claims are assigned "
                 "to these unfinished experiments. ")
    findings += (f"Among completed settings, the lowest observed FID is {measured[best]['fid']:.2f} "
             f"at $\\lambda_d={best:g}$. Coefficient selection on this validation split is "
             "exploratory; an independent test split and repeated training seeds are needed "
             "to estimate selection bias and uncertainty.")
    (out/"weight_sweep_findings.tex").write_text(findings+"\n")
    return {"completed_weights": sorted(measured), "pending_weights": sorted(pending),
            "lowest_measured_fid_weight": best,
            "sources": {str(w): r["source_file"] for w, r in measured.items()}}


def optimization_study(root, out):
    """Import the new experiment without mixing 36k and 36.5k evidence."""
    from launch_dmd_conflict import refresh_results
    if not (root/"configs/dmd_conflict.json").exists():
        raise ValueError("Optimization study needs its pinned plan and fixed reference")
    refresh_results(root)
    shutil.copyfile(root/"report/generated/dmd_conflict_results.tex", out/"optimization_quality.tex")
    comparison = json.loads((root/"runs/sd15/dmd_conflict/comparison.json").read_text())
    methods = ["dmd_conflict", "dmd_conflict_norm_control"]
    if (root/"configs/dmd_block_conflict.json").exists():
        refresh_results(root, "dmd_block_conflict")
        module = json.loads((root/"runs/sd15/dmd_block_conflict/comparison.json").read_text())
        if module["fixed_reference"] != comparison["fixed_reference"]:
            raise ValueError("Module and global experiments must share the fixed DMD reference")
        methods += ["dmd_block_conflict", "dmd_block_norm_control"]
        comparison["results"] += module["results"]
        comparison["completed"] += module["completed"]
        comparison["planned"] += module["planned"]
    reference = comparison["fixed_reference"]
    comparison["best_fid_method_so_far"] = min([reference, *comparison["results"]],
                                                key=lambda record: record["fid"])["method"]
    labels = {"dmd_conflict": "Global projection", "dmd_conflict_norm_control": "Global norm ablation",
              "dmd_block_conflict": "Module projection", "dmd_block_norm_control": "Module norm ablation"}
    by_method = {r["method"]: r for r in comparison["results"]}
    short = {"dmd_conflict": "Global proj.", "dmd_conflict_norm_control": "Global norm",
             "dmd_block_conflict": "Module proj.", "dmd_block_norm_control": "Module norm"}
    rows = [r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Method & Updates & FID $\downarrow$ & IS $\uparrow$ & CLIP $\uparrow$ & LPIPS \\", r"\midrule",
        f"DMD + LPIPS (fixed) & 36,000 & {reference['fid']:.2f} & "
        f"{reference['inception_score_mean']:.2f} & {reference['clip_score_mean']:.2f} & "
        f"{reference['diversity_lpips_alex_mean']:.4f}"+r" \\"]
    for method in methods:
        record = by_method.get(method)
        values = (f"{record['fid']:.2f} & {record['inception_score_mean']:.2f} & {record['clip_score_mean']:.2f}"
                  if record else "-- & -- & --")
        diversity = (f"{record['diversity_lpips_alex_mean']:.4f}"
                     if record and record.get("diversity_lpips_alex_mean") is not None else "--")
        rows.append(short[method]+" & 36,500 & "+values+" & "+diversity+r" \\")
    (out/"optimization_quality.tex").write_text("\n".join(rows+[r"\bottomrule", r"\end{tabular}"])+"\n")
    findings = (f"The fixed DMD+LPIPS reference has FID {reference['fid']:.2f}. ")
    if not comparison["results"]:
        findings += ("Full quality evaluation of the optimization variants is pending. "
                     "Gradient diagnostics alone do not establish a quality improvement.")
    else:
        for result in comparison["results"]:
            difference = result["fid"]-reference["fid"]
            findings += (f"{labels[result['method']]} obtains FID {result['fid']:.2f}, "
                f"{abs(difference):.2f} {'lower' if difference < 0 else 'higher'} than the fixed "
                f"reference, and CLIP {result['clip_score_mean']:.2f}. ")
        for projection, ablation in (("dmd_conflict", "dmd_conflict_norm_control"),
                                    ("dmd_block_conflict", "dmd_block_norm_control")):
            if projection in by_method and ablation in by_method:
                delta = by_method[projection]["fid"]-by_method[ablation]["fid"]
                if abs(delta) < .005:
                    findings += (f"{labels[projection]} and its norm ablation both round to FID "
                                 f"{by_method[projection]['fid']:.2f} at the reported precision. ")
                else:
                    findings += (f"{labels[projection]} has FID {abs(delta):.2f} "
                                 f"{'lower' if delta < 0 else 'higher'} than its norm ablation. ")
        findings += ("These are single-seed measurements; small differences are not evidence "
                     "of statistical significance.")
    (out/"optimization_findings.tex").write_text(findings+"\n")
    diagnostic = root/"artifacts/dmd_gradient_analysis/project_step100.json"
    if diagnostic.exists():
        record = json.loads(diagnostic.read_text())
        if (not record.get("weights_unchanged") or record.get("optimizer_updates") != 0 or
                not record.get("training_pool_only") or len(record["measurements"]) != record["batches"]):
            raise ValueError("Incomplete or non-read-only gradient diagnostic")
        averages = {key: sum(r[key] for r in record["measurements"])/record["batches"]
                    for key in record["mean"]}
        diagnostic_text = (
            f"A read-only diagnostic at the 100-update global-projection checkpoint uses {record['batches']} "
            f"fixed training batches of {record['effective_batch_size']} examples. "
            f"On average, {100*averages['conflicting_block_fraction']:.1f}\\% of LoRA modules have "
            "opposing objective gradients. The correction norm relative to the unmodified combined "
            f"gradient is {100*averages['global_relative_correction']:.2f}\\% for global projection "
            f"and {100*averages['block_relative_correction']:.2f}\\% for module projection. "
            "No optimizer update is performed. This motivates the finer partition but is not "
            "evidence of better generated images.")
        shutil.copyfile(diagnostic, out/"optimization_diagnostic.json")
        gradient_diagnostic_figure(record, out/"gradient_diagnostic")
        (out/"gradient_diagnostic.tex").write_text(
            "\\begin{figure*}[t]\n\\centering\n"
            "\\includegraphics[width=.96\\textwidth]{generated/assignment2/gradient_diagnostic.pdf}\n"
            "\\caption{Measured gradient alignment on eight fixed training batches at the "
            "100-update global-projection checkpoint. Each heatmap row is one effective batch of "
            "16 examples; all 128 LoRA modules are shown, with each module's A/B matrices grouped "
            "together. Red indicates opposing DMD and LPIPS gradients. The right panel shows the "
            "norm of the proposed correction relative to the unmodified combined gradient. "
            "Weights are unchanged during this diagnostic; the figure describes local gradient "
            "geometry, not generated-image quality.}\n"
            "\\label{fig:gradient-diagnostic}\n\\end{figure*}\n")
    else:
        diagnostic_text = "The module-level diagnostic is pending; no measurement is assumed."
        (out/"gradient_diagnostic.tex").write_text("% Gradient measurements are pending.\n")
    (out/"optimization_diagnostic.tex").write_text(diagnostic_text+"\n")
    comparison["qualitative"] = optimization_samples(root, out, comparison)
    return comparison


def gradient_diagnostic_figure(record, target):
    import numpy as np
    measurements = record["measurements"]
    names = [item["module"] for item in measurements[0]["blocks"]]
    if len(names) != 128 or any([item["module"] for item in row["blocks"]] != names
                                for row in measurements):
        raise ValueError("Gradient figure requires all 128 modules in every diagnostic batch")
    families = [(prefix, label) for prefix, label in (("down_blocks.", "Down"),
                 ("mid_block.", "Mid"), ("up_blocks.", "Up"))]
    groups = [[i for i, name in enumerate(names) if name.startswith(prefix)] for prefix, _ in families]
    order = [i for group in groups for i in group]
    if sorted(order) != list(range(len(names))):
        raise ValueError("Unknown LoRA module family in gradient diagnostic")
    local = np.array([[row["blocks"][i]["cosine"] for i in order] for row in measurements])
    global_cosine = np.array([[row["global_cosine"]] for row in measurements])
    correction = 100*np.array([[row["global_relative_correction"], row["block_relative_correction"]]
                              for row in measurements])
    if (not all(np.isfinite(values).all() for values in (local, global_cosine, correction)) or
            np.abs(local).max() > 1+1e-6 or np.abs(global_cosine).max() > 1+1e-6 or
            (correction < 0).any()):
        raise ValueError("Invalid measured gradient values")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(7.35, 2.65))
        grid = fig.add_gridspec(2, 3, width_ratios=[.42, 3.75, 3.0], height_ratios=[1, .075],
                               left=.07, right=.99, bottom=.16, top=.83, wspace=.25, hspace=.40)
        global_ax, local_ax, bar_ax = [fig.add_subplot(grid[0, i]) for i in range(3)]
        for axis, values in ((global_ax, global_cosine), (local_ax, local)):
            heatmap = axis.imshow(values, aspect="auto", cmap="RdBu", vmin=-1, vmax=1,
                                  interpolation="nearest")
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_color("#d1d6dc")
        global_ax.set(title="Global", ylabel="Training batch", xticks=[0], xticklabels=["All"],
                      yticks=range(len(measurements)), yticklabels=range(1, len(measurements)+1))
        local_ax.set(title="Module alignment", yticks=[])
        offset, centers = 0, []
        for group in groups:
            centers.append(offset+(len(group)-1)/2)
            offset += len(group)
            if offset < len(names):
                local_ax.axvline(offset-.5, color="white", linewidth=1)
        local_ax.set_xticks(centers, [label for _, label in families])
        colorbar = fig.colorbar(heatmap, cax=fig.add_subplot(grid[1, :2]), orientation="horizontal",
                               ticks=[-1, 0, 1])
        colorbar.set_label("DMD / LPIPS gradient cosine", labelpad=1)
        colorbar.ax.tick_params(length=2, labelsize=7, pad=1)
        colorbar.outline.set_visible(False)
        positions = np.arange(len(measurements))
        for column, label, color in ((0, "Global", "#345675"), (1, "Module", "#c47c3c")):
            bar_ax.bar(positions+(column-.5)*.36, correction[:, column], width=.34,
                       label=label, color=color, zorder=3)
        bar_ax.set(title="Correction magnitude", xlabel="Training batch", ylabel="Correction norm (%)",
                   xticks=positions, xticklabels=positions+1, ylim=(0, correction.max()*1.25))
        bar_ax.spines[["top", "right"]].set_visible(False)
        bar_ax.grid(axis="y", color="#e5e8eb", zorder=0)
        bar_ax.legend(ncol=2, loc="upper left", frameon=False, fontsize=7,
                      borderaxespad=.2, handlelength=1, columnspacing=.8)
        for axis in (global_ax, local_ax, bar_ax):
            axis.title.set_fontsize(9)
        for extension in ("pdf", "svg", "png"):
            fig.savefig(target.with_suffix("."+extension), dpi=220, bbox_inches="tight", pad_inches=.025)
        plt.close(fig)


def noise_agreement_study(root, out, study="dmd_noise_agreement", prefix="noise_agreement"):
    """Keep the distinct score-update proposal separate from earlier projection pilots."""
    from launch_dmd_conflict import refresh_results
    from draw_noise_agreement import draw
    if study == "dmd_noise_agreement":
        draw(out/"noise_agreement_method")
    elif study == "dmd_pair_agreement":
        from draw_pair_agreement import draw as draw_pairs
        draw_pairs(out/"pair_agreement_method")
    if not (root/"configs"/f"{study}.json").exists():
        (out/f"{prefix}_quality.tex").write_text("Experiments are being prepared.\n")
        (out/f"{prefix}_findings.tex").write_text("No quality measurement is available yet.\n")
        (out/f"{prefix}_diversity.tex").write_text("Diversity evaluation is pending.\n")
        (out/f"{prefix}_samples.tex").write_text("% Awaiting evaluated checkpoints.\n")
        return None
    refresh_results(root, study)
    comparison = json.loads((root/"runs/sd15"/study/"comparison.json").read_text())
    shutil.copyfile(root/"report/generated"/f"{study}_results.tex", out/f"{prefix}_quality.tex")
    reference = comparison["fixed_reference"]
    records = {r["method"]: r for r in comparison["results"]}
    names = {"dmd_noise_agreement": "Spatial agreement", "dmd_noise_mean": "Paired mean",
             "dmd_noise_shifted": "Shifted-map ablation"}
    if study == "dmd_pair_agreement":
        names = {"dmd_pair_agreement": "Pair agreement", "dmd_pair_mean": "Four-probe mean",
                 "dmd_pair_shifted": "Shifted-pair ablation"}
    findings = ""
    for method, name in names.items():
        if method in records:
            result = records[method]
            delta = result["fid"]-reference["fid"]
            findings += (f"{name} obtains FID {result['fid']:.2f}, {abs(delta):.2f} "
                         f"{'lower' if delta < 0 else 'higher'} than the fixed reference. ")
    candidate_method, *ablations = names
    candidate = records.get(candidate_method)
    if candidate:
        for ablation in ablations:
            if ablation in records:
                delta = candidate["fid"]-records[ablation]["fid"]
                if abs(delta) < .005:
                    findings += (f"Its FID and that of {names[ablation].lower()} are equal "
                                 "at the reported two-decimal precision. ")
                else:
                    findings += (f"Its FID is {abs(delta):.2f} {'lower' if delta < 0 else 'higher'} "
                                 f"than {names[ablation].lower()}. ")
    findings += ("Full-validation evaluation is pending for one or more variants. "
                 if len(records) < 3 else "")
    findings += ("These exploratory single-seed comparisons do not establish statistical significance. "
                 "A gain over the fixed reference alone cannot separate the update rule from the additional training.")
    (out/f"{prefix}_findings.tex").write_text(findings+"\n")
    rows = [r"\begin{tabular}{lr}", r"\toprule", r"Method & Pairwise LPIPS \\", r"\midrule",
            f"DMD + LPIPS (fixed) & {reference['diversity_lpips_alex_mean']:.4f}"+r" \\"]
    for method, name in names.items():
        value = records.get(method, {}).get("diversity_lpips_alex_mean")
        rows.append(name+" & "+(f"{value:.4f}" if value is not None else "--")+r" \\")
    (out/f"{prefix}_diversity.tex").write_text("\n".join(rows+[r"\bottomrule", r"\end{tabular}"])+"\n")
    comparison["qualitative"] = optimization_samples(root, out, comparison,
        studies=(study,), prefix=f"{prefix}_samples")
    if study == "dmd_noise_agreement":
        from draw_probe_diagnostic import render
        comparison["independent_probe_diagnostic"] = render(root, out)
    else:
        comparison["development_diagnostic"] = pair_development_evidence(root, out)
    return comparison


def response_alignment_evidence(root, out, manifest, comparison, study, prefix, reference):
    """Display only diagnostics bound to fully evaluated, completed exports."""
    from analyze_teacher_responses import completed_source
    from few_step_diffusion.runtime import sha256
    section_path = out/f"{prefix}_alignment_section.tex"
    section_path.write_text("% Awaiting completed response diagnostics.\n")
    available = [(job, root/job["run_dir"]/"response_diagnostic.json")
                 for job in manifest["jobs"]
                 if (root/job["run_dir"]/"response_diagnostic.json").exists()]
    if not available:
        return []
    fields = ("response_cosine", "student_to_teacher_response_norm", "point_latent_mse")
    protocol = ("policy", "parent_checkpoint_sha256", "teacher_pair_weights_sha256",
                "teacher_pair_protocol_hash", "prompt_hash", "start_prompt", "prompt_count")

    def measurements(data):
        rows = data["measurements"]
        if (data.get("policy") != "frozen-teacher-finite-response-diagnostic-v1" or
                data.get("training_only") is not True or data.get("weights_unchanged") is not True or
                data.get("optimizer_updates") != 0 or data.get("gradient_calibration") or
                not rows or len(rows) != data["prompt_count"] or
                [r["prompt_index"] for r in rows] !=
                list(range(data["start_prompt"], data["start_prompt"]+data["prompt_count"]))):
            raise ValueError("Response report requires frozen, matched training-only diagnostics")
        values = {key: [r["scales"]["1"][key] for r in rows] for key in fields}
        if not all(math.isfinite(value) for column in values.values() for value in column):
            raise ValueError("Response diagnostic contains nonfinite measurements")
        return {key: math.fsum(column)/len(column) for key, column in values.items()}

    def table_row(label, stats):
        return label+" & "+" & ".join(f"{stats[key]:.4f}" for key in fields)+r' \\'

    baseline = measurements(reference)
    reference_prompts = [(r["prompt_index"], r["image_id"], r["caption"])
                         for r in reference["measurements"]]
    records = {r["method"]: r for r in comparison["results"]}
    rows = [r"\begin{tabular}{lrrr}", r"\toprule",
            r"Setting & Cosine $\uparrow$ & Norm ratio & Latent MSE $\downarrow$ \\",
            r"\midrule", table_row("DMD + LPIPS (fixed)", baseline)]
    evidence = []
    for job, source in available:
        run, _, _ = completed_source(root, study, job["key"])
        data = json.loads(source.read_text())
        result = records.get(job["method"], {})
        if (any(data.get(key) != reference[key] for key in protocol) or
                data.get("model_role") != "evaluated_response_variant" or
                data.get("reference_model_identity") != reference["model_identity"] or
                data.get("model_identity") != result.get("model_identity") or
                data.get("config_hash") != job["config_hash"] or
                data.get("source_checkpoint_sha256") != sha256(run/"last.pt") or
                [(r["prompt_index"], r["image_id"], r["caption"])
                 for r in data["measurements"]] != reference_prompts):
            raise ValueError("Response report diagnostic differs from the evaluated model or matched prompts")
        stats = measurements(data)
        mode, fraction = RESPONSE_SETTINGS[job["key"]]
        label = f"{mode.capitalize()} {fraction:.0%}".replace("%", r"\%")
        rows.append(table_row(label, stats))
        shutil.copyfile(source, out/f"{prefix}_diagnostic_{job['key']}.json")
        evidence.append({"method": job["method"], "source": str(source.relative_to(root)),
                         "source_sha256": sha256(source), "model_identity": data["model_identity"],
                         "summary": stats, "cosine_change": stats["response_cosine"]-baseline["response_cosine"]})
    section_path.write_text(
        "\\begin{table}[t]\n\\centering\\scriptsize\n\\resizebox{\\linewidth}{!}{%\n"+
        "\n".join(rows+[r"\bottomrule", r"\end{tabular}"])+"}\n"+
        r"\caption{Finite-response diagnostic on "+str(reference["prompt_count"])+
        " matched training captions, using two cached noises per caption. "
        "Cosine measures directional agreement; a norm ratio of one matches the teacher's "
        "response magnitude. Latent MSE measures pointwise reconstruction. "
        "These training-bank measurements diagnose the optimized mechanism, not held-out image quality. "
        "Only completed, fully evaluated checkpoints are shown.}\n\\end{table}\n")
    return evidence


def paired_diversity_interval(reference, candidate):
    """Resample matched captions, keeping the trained models and image seeds fixed."""
    import numpy as np
    fields = ("diversity_protocol", "diversity_prompt_hash", "diversity_prompt_count",
              "diversity_seed_list", "diversity_samples_per_prompt", "diversity_lpips_identity")
    if any(reference.get(key) is None or reference[key] != candidate.get(key) for key in fields):
        raise ValueError("Paired diversity uncertainty requires identical caption, seed and LPIPS protocols")
    count = reference["diversity_prompt_count"]
    seeds = reference["diversity_seed_list"]
    if count < 2 or len(set(seeds)) != len(seeds) or len(seeds) != reference["diversity_samples_per_prompt"]:
        raise ValueError("Invalid paired diversity caption or seed counts")
    values = []
    for record in (reference, candidate):
        scores = np.asarray(record["diversity_pairwise_scores_per_prompt"], dtype=np.float64)
        if (scores.shape != (count,) or not np.isfinite(scores).all() or
                not math.isclose(float(scores.mean()), record["diversity_lpips_alex_mean"], abs_tol=1e-8)):
            raise ValueError("Per-caption diversity scores disagree with the reported mean or count")
        values.append(scores)
    difference = values[1]-values[0]
    indices = np.random.default_rng(90421).integers(0, count, size=(10000, count))
    interval = np.quantile(difference[indices].mean(axis=1), [.025, .975])
    return {"method": "paired-caption-percentile-bootstrap", "bootstrap_seed": 90421,
        "bootstrap_replicates": 10000, "prompt_count": count,
        "mean_difference": float(difference.mean()), "percentile_interval_95": interval.tolist(),
        "paired_prompt_differences": difference.tolist(),
        "scope": "Caption variation only; fixed models and image seeds. Unadjusted exploratory intervals."}


def response_diversity_uncertainty(root, out, manifest, records, prefix):
    """Bind paired intervals to the same native evaluations used in the tables."""
    from few_step_diffusion.runtime import sha256
    completed = [job for job in manifest["jobs"]
                 if records.get(job["method"], {}).get("diversity_lpips_alex_mean") is not None]
    evidence = {}
    if completed:
        reference_path = root/manifest["reference_baseline"]["result_file"]
        if sha256(reference_path) != manifest["reference_baseline"]["result_sha256"]:
            raise ValueError("Diversity uncertainty reference differs from the pinned result")
        reference = json.loads(reference_path.read_text())
        if reference["model_identity"] != manifest["reference_baseline"]["model_identity"]:
            raise ValueError("Diversity uncertainty reference model differs")
        for job in completed:
            path = root/job["result_dir"]/f"{job['method']}_seed{manifest['seed']}_1steps.result.json"
            record = json.loads(path.read_text())
            summary = records[job["method"]]
            if (record["method"] != job["method"] or record["config_hash"] != job["config_hash"] or
                    any(record[key] != summary[key] for key in ("model_identity", "diversity_lpips_alex_mean"))):
                raise ValueError("Diversity uncertainty source differs from the evaluated model")
            evidence[job["method"]] = {**paired_diversity_interval(reference, record),
                "model_identity": record["model_identity"], "result_file": str(path.relative_to(root)),
                "result_sha256": sha256(path), "reference_result_sha256": sha256(reference_path),
                "reference_model_identity": reference["model_identity"],
                "prompt_hash": reference["diversity_prompt_hash"], "generation_seeds": reference["diversity_seed_list"]}
    (out/f"{prefix}_diversity_uncertainty.json").write_text(json.dumps(evidence, indent=2)+"\n")
    return evidence


def response_strength_study(root, out, study="dmd_response_strength", prefix="response"):
    """Report the new response hypothesis without selecting a final model automatically."""
    from launch_dmd_conflict import refresh_results
    from few_step_diffusion.dmd_response import SETTINGS
    from few_step_diffusion.config import load_config
    from few_step_diffusion.runtime import sha256
    from few_step_diffusion.reporting import tex_escape
    final = study == "dmd_response_final"
    if not final:
        from draw_response_alignment import draw
        draw(out/"response_alignment_method")
    if final:
        (out/f"{prefix}_section.tex").write_text("% No final response study has been selected.\n")
    if not (root/"configs"/f"{study}.json").exists():
        for part in ("quality", "findings", "calibration", "diversity"):
            (out/f"{prefix}_{part}.tex").write_text("The response study is being prepared.\n")
        (out/f"{prefix}_samples.tex").write_text("% Awaiting evaluated checkpoints.\n")
        (out/f"{prefix}_alignment_section.tex").write_text("% Awaiting completed response diagnostics.\n")
        return None
    refresh_results(root, study)
    comparison = json.loads((root/"runs/sd15"/study/"comparison.json").read_text())
    manifest = json.loads((root/"configs"/f"{study}.json").read_text())
    calibration_path = root/"artifacts/dmd_teacher_response/gradient_calibration.json"
    calibration = json.loads(calibration_path.read_text())
    diagnostic = json.loads((root/"artifacts/dmd_teacher_response/frozen_reference.json").read_text())
    if any(data["parent_checkpoint_sha256"] != manifest["parent_checkpoint_sha256"] or
           data["model_identity"] != manifest["reference_baseline"]["model_identity"] or
           data["training_only"] is not True or data["weights_unchanged"] is not True
           for data in (calibration, diagnostic)):
        raise ValueError("Response diagnostics differ from the fixed training-only reference")
    if any(load_config(root/job["config"])["dmd_refinement"]["calibration_sha256"] != sha256(calibration_path)
           for job in manifest["jobs"]):
        raise ValueError("Reported response calibration differs from the pinned training coefficients")
    shutil.copyfile(root/"report/generated"/f"{study}_results.tex", out/f"{prefix}_quality.tex")
    weights = calibration["gradient_calibration"]["coefficients"]["angular"]["weights"]
    stats = diagnostic["summary"]["1"]
    (out/f"{prefix}_calibration.tex").write_text(
        f"On {diagnostic['prompt_count']} training captions, the frozen student's mean response "
        f"cosine is {stats['response_cosine']['mean']:.3f} and its mean response-norm ratio to the teacher "
        f"is {stats['student_to_teacher_response_norm']['mean']:.3f}. This motivates testing direction alignment "
        "without forcing larger changes. A separate set of 32 training captions sets the three coefficients "
        "to $\\lambda\\in\\{"+", ".join(f"{weights[str(f)]:.4f}" for f in (.1, .3, .5))+r"\}$. "
        "These yield median initial latent-gradient ratios of 10\\%, 30\\% and 50\\% relative to DMD. "
        "The ratios are measured at the frozen EMA checkpoint and are not enforced during training.\n")
    reference = comparison["fixed_reference"]
    records = {r["method"]: r for r in comparison["results"]}
    uncertainty = response_diversity_uncertainty(root, out, manifest, records, prefix)
    comparison["diversity_uncertainty"] = uncertainty
    findings = []
    rows = [r"\begin{tabular}{lrrl}", r"\toprule",
            r"Setting & LPIPS & $\Delta$ & 95\% interval \\", r"\midrule",
            f"DMD + LPIPS (fixed) & {reference['diversity_lpips_alex_mean']:.4f} & -- & --"+r" \\"]
    for job in manifest["jobs"]:
        mode, fraction = SETTINGS[job["key"]]
        label = f"{mode.capitalize()} {fraction:.0%}".replace("%", r"\%")
        record = records.get(job["method"], {})
        if record:
            findings.append(f"{label} obtains FID {record['fid']:.2f} "
                            f"({record['fid']-reference['fid']:+.2f} relative to fixed DMD).")
        value = record.get("diversity_lpips_alex_mean")
        if value is None:
            rows.append(label+r" & -- & -- & -- \\")
        else:
            estimate = uncertainty[job["method"]]
            low, high = estimate["percentile_interval_95"]
            rows.append(f"{label} & {value:.4f} & {estimate['mean_difference']:+.5f} & "
                        f"[{low:+.5f}, {high:+.5f}]"+r" \\")
    if len(records) < len(manifest["jobs"]):
        findings.append("Full-validation results remain pending for one or more variants.")
    findings.append("These are exploratory single-seed measurements; extra training and repeated "
                    "validation inspection limit causal and generalization claims.")
    (out/f"{prefix}_findings.tex").write_text(" ".join(findings)+"\n")
    (out/f"{prefix}_diversity.tex").write_text("\n".join(rows+[r"\bottomrule", r"\end{tabular}"])+"\n")
    comparison["qualitative"] = optimization_samples(root, out, comparison,
        studies=(study,), prefix=f"{prefix}_samples")
    comparison["diagnostic"] = {"prompt_count": diagnostic["prompt_count"], "summary": stats,
                               "calibration": calibration["gradient_calibration"]}
    comparison["completed_response_diagnostics"] = response_alignment_evidence(
        root, out, manifest, comparison, study, prefix, diagnostic)
    if final:
        selected = manifest["final_design"]
        fraction = selected["gradient_fraction"]
        coefficients = calibration["gradient_calibration"]["coefficients"]
        descriptions = [f"{mode}: {coefficients[mode]['weights'][str(fraction)]:.4f}"
                        for mode in ("angular", "magnitude", "vector")]
        (out/f"{prefix}_calibration.tex").write_text(
            "All three objectives use the same training-calibrated median initial gradient ratio "
            f"of {fraction*100:g}\\%. Their fixed loss coefficients are "+", ".join(descriptions)+".\n")
        section = (
            r"\subsection{Selected response method and component ablations}"+"\n"
            "Following the completed strength pilot, we compare angular alignment with "
            "magnitude-only and normalized full-vector response losses. For the paired "
            "displacements defined above, the two controls are "
            r"$(\|d_S\|_2/\|d_T\|_2-1)^2$ and $\|d_S-d_T\|_2^2/\|d_T\|_2^2$. "
            "The fixed DMD reference is not retrained. Every arm starts from its full 36,000-update "
            f"state and receives {manifest['additional_updates']:,} further updates "
            f"({manifest['cumulative_training_updates']:,} total). No arm starts from a tuned pilot export. "
            "Selection was exploratory: "+tex_escape(selected["reason"])+"\n"
            f"\\input{{generated/assignment2/{prefix}_calibration.tex}}\n"
            "\\begin{table}[t]\n\\centering\\scriptsize\n"
            f"\\resizebox{{\\linewidth}}{{!}}{{\\input{{generated/assignment2/{prefix}_quality.tex}}}}\n"
            r"\caption{Selected response method and equal-budget component ablations. "
            "All available quality results use 5,000 validation images; dashes indicate pending results.}\n"
            "\\end{table}\n"
            f"\\input{{generated/assignment2/{prefix}_findings.tex}}\n"
            "\\begin{table}[t]\n\\centering\\scriptsize\n"
            f"\\resizebox{{\\linewidth}}{{!}}{{\\input{{generated/assignment2/{prefix}_diversity.tex}}}}\n"
            r"\caption{Within-caption LPIPS-Alex for the selected response study, using "
            "128 captions and four seeds. Changes relative to fixed DMD have paired-caption "
            "95\\% percentile intervals from 10,000 bootstrap resamples. Models and image seeds "
            "are fixed; intervals are not adjusted for exploratory comparisons and do not measure "
            "training-seed uncertainty.}\n\\end{table}\n"
            f"\\input{{generated/assignment2/{prefix}_samples.tex}}\n"
            f"\\input{{generated/assignment2/{prefix}_alignment_section.tex}}\n")
        (out/f"{prefix}_section.tex").write_text(section)
    return comparison


def write_completed_response_final(root, out, response_final):
    """Publish final component evidence only after all three arms are verified."""
    target = out/"response_completed_final_section.tex"
    target.write_text("% Awaiting completed component study.\n")
    if not response_final:
        return False
    plan = json.loads((root/"configs/dmd_response_final.json").read_text())
    expected = {job["method"] for job in plan["jobs"]}
    records = response_final["results"]
    diagnostics = response_final["completed_response_diagnostics"]
    qualitative = response_final.get("qualitative")
    if (len(records) < len(expected) or len(diagnostics) < len(expected) or
            any(r.get("diversity_lpips_alex_mean") is None for r in records) or
            not qualitative or len(qualitative["methods"]) < len(expected)+1):
        return False
    if (len(expected) != 3 or len(records) != 3 or len(diagnostics) != 3 or
            {r["method"] for r in records} != expected or
            {r["method"] for r in diagnostics} != expected or
            any(r["cumulative_training_updates"] != plan["cumulative_training_updates"]
                for r in records)):
        raise ValueError("Final component report requires all three distinct, equal-budget arms")
    identities = {r["method"]: r["model_identity"] for r in records}
    if any(r["model_identity"] != identities[r["method"]] for r in diagnostics):
        raise ValueError("Final component diagnostics differ from the evaluated models")
    reference = response_final["fixed_reference"]
    identities[reference["method"]] = reference["model_identity"]
    if (len(qualitative["methods"]) != 4 or
            {r["method"]: r["model_identity"] for r in qualitative["methods"]} != identities):
        raise ValueError("Final component samples differ from the evaluated models")
    rows = [r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Objective & FID & IS & CLIP & Pairwise LPIPS \\", r"\midrule"]
    for label, record in [("DMD (fixed)", reference)]+[
            (r["method"].split("_")[-2].capitalize(), r) for r in response_final["results"]]:
        if label == "Angular":
            label = "Angular (ours)"
        rows.append(label+" & "+" & ".join(f"{record[m]:.2f}" for m in
            ("fid", "inception_score_mean", "clip_score_mean"))+
            f" & {record['diversity_lpips_alex_mean']:.4f}"+r" \\")
    final_section = (r"\subsection{Extended training and component controls}"+"\n"
        "Angular, magnitude-only and normalized full-vector objectives start from the same fixed DMD "
        f"checkpoint and receive {plan['additional_updates']:,} updates ({plan['cumulative_training_updates']:,} total). "
        r"The controls are $(\|d_S\|/\|d_T\|-1)^2$ and $\|d_S-d_T\|^2/\|d_T\|^2$. "
        "\\input{generated/assignment2/response_final_calibration.tex}\n"
        "\\begin{table}[t]\n\\centering\\scriptsize\n\\resizebox{\\linewidth}{!}{%\n"+
        "\n".join(rows+[r"\bottomrule", r"\end{tabular}"])+"}\n"
        r"\caption{Completed equal-budget response-component study at 37,000 updates per arm. FID, IS and CLIP use all 5,000 "
        "validation images; pairwise LPIPS uses 128 captions and four seeds. The fixed DMD reference "
        "has 36,000 updates.}\n\\end{table}\n"
        "\\input{generated/assignment2/response_final_findings.tex}\n"
        "\\begin{table}[t]\n\\centering\\scriptsize\n"
        "\\resizebox{\\linewidth}{!}{\\input{generated/assignment2/response_final_diversity.tex}}\n"
        r"\caption{Within-caption perceptual variation for the completed component study. "
        "Changes from fixed DMD use 10,000 paired-caption bootstrap resamples. The 95\\% "
        "intervals keep models and image seeds fixed and are unadjusted for exploratory comparisons.}\n"
        "\\end{table}\n"
        "\\input{generated/assignment2/response_final_alignment_section.tex}\n"
        "\\input{generated/assignment2/response_final_samples.tex}\n")
    modes = {r["method"].split("_")[-2]: r for r in records}
    angular, magnitude, vector = [modes[mode] for mode in ("angular", "magnitude", "vector")]
    interpretation = []
    if (angular["fid"] < min(magnitude["fid"], vector["fid"]) and
            angular["diversity_lpips_alex_mean"] >
            max(magnitude["diversity_lpips_alex_mean"], vector["diversity_lpips_alex_mean"])):
        interpretation.append("At the matched budget, angular alignment has the lowest FID and "
                              "largest measured within-caption LPIPS among the three response objectives.")
    if all(magnitude[m] > max(angular[m], vector[m]) for m in ("inception_score_mean", "clip_score_mean")):
        interpretation.append("Magnitude-only alignment has the highest IS and CLIP; "
                              "the metric rankings therefore differ.")
    diagnostic_stats = {r["method"].split("_")[-2]: r.get("summary", {}) for r in diagnostics}
    if (all("point_latent_mse" in diagnostic_stats[mode] for mode in modes) and
            diagnostic_stats["vector"]["point_latent_mse"] <
            min(diagnostic_stats[mode]["point_latent_mse"] for mode in ("angular", "magnitude")) and
            vector["fid"] >= reference["fid"]):
        interpretation.append("Vector alignment has the smallest training-bank latent MSE but does not "
                              "improve FID over fixed DMD. Better reconstruction alone therefore does "
                              "not establish better generated-image quality.")
    final_section += " ".join(interpretation)+"\n"
    target.write_text(final_section)
    return True


def response_frontmatter(out, pilot, final):
    """Summarize measured angular results without promoting a pilot to final."""
    final_angles = [record for record in (final or {}).get("results", [])
                    if record["method"].startswith("dmd_response_angular_")]
    pilot_angles = [record for record in (pilot or {}).get("results", [])
                    if record["method"].startswith("dmd_response_angular_")]
    if final_angles:
        if len(final_angles) != 1:
            raise ValueError("The final study must identify exactly one angular candidate")
        candidate, study = final_angles[0], final
        label = "The follow-up angular-response model"
        source = "final"
    elif pilot_angles:
        candidate, study = min(pilot_angles, key=lambda r: r["fid"]), pilot
        label = "The best completed direction-alignment pilot"
        source = "pilot"
    else:
        for part in ("summary", "conclusion"):
            (out/f"response_{part}.tex").write_text("Direction-alignment evaluation remains pending.\n")
        return None
    reference = study["fixed_reference"]
    total = candidate["cumulative_training_updates"]
    extra = total-reference["cumulative_training_updates"]
    if extra <= 0:
        raise ValueError("Response summary requires the actual additional training budget")
    summary = (f"{label} obtains FID {candidate['fid']:.2f} after {total:,} total updates, "
               f"compared with {reference['fid']:.2f} for the fixed DMD reference "
               f"({extra:,} fewer updates). ")
    if source == "pilot":
        summary += "This is an exploratory pilot result; a final model has not yet been evaluated. "
    (out/"response_summary.tex").write_text(summary+"\n")
    (out/"response_conclusion.tex").write_text(summary+
        "The unequal budgets prevent attributing the entire difference to the added response term.\n")
    return {"source": source, "method": candidate["method"], "fid": candidate["fid"],
            "reference_fid": reference["fid"], "cumulative_training_updates": total,
            "additional_updates": extra}


def pair_development_evidence(root, out):
    """Preserve the actual training-only evidence used to choose the follow-up."""
    import numpy as np
    from few_step_diffusion.runtime import sha256
    source = root/"artifacts/dmd_noise_agreement/probe_diagnostic/strength_and_independent_pairs.json"
    data = json.loads(source.read_text())
    plan = json.loads((root/"configs/dmd_pair_agreement.json").read_text())
    rows = data["measurements"]
    captions = sorted({r["caption_index"] for r in rows})
    if (not data.get("weights_unchanged") or data.get("optimizer_updates") != 0 or
            not data.get("training_pool_only") or not data.get("independent_pair_check") or
            data["checkpoint_sha256"] != plan["parent_checkpoint_sha256"] or
            data["sensitivity_strengths"] != [0, .25, .5, 1, 2, 4] or
            captions != list(range(4, 36)) or len(rows) != 96 or
            len({(r["caption_index"], r["timestep"]) for r in rows}) != 96):
        raise ValueError("Follow-up development evidence differs from the declared protocol")
    means = {}
    for family in ("two_probes", "four_probes"):
        means[family] = {str(beta): float(np.mean([
            r["sensitivity"][family]["strengths"]["agreement"][f"{beta:g}"]["cosine_gain_over_mean"]
            for r in rows])) for beta in data["sensitivity_strengths"]}
        if not all(np.isfinite(list(means[family].values()))) or float(max(means[family], key=means[family].get)) != 1:
            raise ValueError("Declared filter-strength selection does not match development measurements")
    values = np.array([np.mean([r["sensitivity"]["four_probes"]["strengths"]["agreement"]["1"]
                               ["cosine_gain_over_mean"] for r in rows if r["caption_index"] == caption])
                       for caption in captions])
    indices = np.random.default_rng(4317).integers(0, 32, size=(10000, 32))
    low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
    macros = {"PairDirectionGain": float(values.mean()), "PairDirectionLow": float(low), "PairDirectionHigh": float(high)}
    (out/"pair_development_statistics.tex").write_text("\n".join(
        f"\\newcommand{{\\{key}}}{{{value:.5f}}}" for key, value in macros.items())+"\n")
    shutil.copyfile(source, out/"pair_development_measurements.json")
    return {"source_file": str(source.relative_to(root)), "source_sha256": sha256(source),
            "selected_strength": 1, "training_only": True, "optimizer_updates": 0,
            "mean_alignment_gains": means, "four_probe_caption_interval_95": [float(low), float(high)]}


def completed_preview(root, job, result, reference, *, step=500):
    """Load native images only when the archived model matches a completed evaluation."""
    from few_step_diffusion.runtime import sha256
    folder = root/job["run_dir"]/"checkpoint_previews"/f"step_{step:06d}"
    payload = folder/"samples.json"
    if not result or not payload.exists():
        return None
    saved = json.loads(payload.read_text())
    if (saved["checkpoint_sha256"] != sha256(folder/"checkpoint.pt") or
            saved["cumulative_training_updates"] != result["cumulative_training_updates"]):
        raise ValueError("Optimization samples do not match the final archived checkpoint")
    baseline, sample = saved["records"]
    if (baseline["method"] != "dmd_perceptual" or
            baseline["model_identity"] != reference["model_identity"] or
            sample["model_identity"] != result["model_identity"] or
            sample["method"] != job["method"] or sample["global_step"] != step or
            sample.get("cumulative_training_updates") != result["cumulative_training_updates"] or
            sample["sampling_steps"] != 1 or sample["nfe"] != 1):
        raise ValueError("Optimization samples differ from the evaluated models")
    return saved


def select_main_candidate(root, columns, studies):
    """An explicit pinned selection may replace the earlier distance-extension column.

    Study imports must already have validated full metrics. Pending arms, absent
    diversity scores, and checkpoint/image mismatches cannot be promoted.
    """
    from few_step_diffusion.runtime import sha256
    path = root/"configs/assignment2_selected_method.json"
    if not path.exists():
        return columns, None
    selection = json.loads(path.read_text())
    study = selection.get("study")
    if study == "independent_response_36k":
        return select_independent_main(root, columns, selection)
    if (selection.get("policy") != "assignment2-evaluated-main-candidate-v1" or
            study not in studies or not studies[study] or
            not isinstance(selection.get("reason"), str) or not selection["reason"].strip()):
        raise ValueError("Invalid final-candidate selection policy, study or rationale")
    comparison = studies[study]
    plan = json.loads((root/"configs"/f"{study}.json").read_text())
    results = {r["method"]: r for r in comparison["results"]}
    if any(j["method"] not in results or
           results[j["method"]].get("diversity_lpips_alex_mean") is None
           for j in plan["jobs"] if not j.get("cancelled", False)):
        raise ValueError("Final-candidate selection requires completed quality and diversity for all study arms")
    jobs = [j for j in plan["jobs"] if j["method"] == selection.get("method")]
    if len(jobs) != 1:
        raise ValueError("Selected candidate is absent or ambiguous in the validated study")
    job = jobs[0]
    result = results[job["method"]]
    result_path = root/job["result_dir"]/f"{job['method']}_seed{plan['seed']}_1steps.result.json"
    if (selection.get("result_sha256") != sha256(result_path) or
            selection.get("model_identity") != result["model_identity"]):
        raise ValueError("Selected result or model differs from its pinned identity")
    saved = completed_preview(root, job, result, comparison["fixed_reference"],
                              step=plan["additional_updates"])
    if saved is None:
        raise ValueError("Selected candidate needs final native checkpoint samples")
    if selection.get("checkpoint_sha256") != saved["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint differs from its pinned identity")
    baseline, sample = saved["records"]
    reference = next(r for r in columns if r["method"] == "dmd_perceptual")
    for key in ("model_identity", "sample_prompts", "sample_image_ids", "sample_labels",
                "eval_seed", "evaluation_prompt_hash", "sample_images_png_base64"):
        if baseline.get(key) != reference.get(key):
            raise ValueError(f"Selected preview uses a different main-figure reference: {key}")
    for key in ("sample_prompts", "sample_image_ids", "sample_labels", "eval_seed", "evaluation_prompt_hash"):
        if sample.get(key) != reference.get(key):
            raise ValueError(f"Selected preview uses different prompts or noise: {key}")
    replacement = {**sample, "source_file": str(result_path.relative_to(root))}
    updated = [replacement if r["method"] == "dmd_perceptual_diversity" else r for r in columns]
    if sum(r["method"] == job["method"] for r in updated) != 1:
        raise ValueError("Main figure must contain exactly one selected candidate column")
    return updated, {**selection, "source_file": str(path.relative_to(root)),
                     "result_file": str(result_path.relative_to(root)),
                     "cumulative_training_updates": result["cumulative_training_updates"]}


def select_independent_main(root, columns, selection):
    """Bind the main paper to the completed independent run and its native figures."""
    from few_step_diffusion.independent_response import validate_result
    from few_step_diffusion.runtime import sha256
    if (selection.get("policy") != "assignment2-evaluated-main-candidate-v1" or
            selection.get("method") != "response_scratch" or not selection.get("reason")):
        raise ValueError("Invalid independent main-candidate selection")
    plan = json.loads((root/"configs/independent_response_36k.json").read_text())
    job = next(j for j in plan["jobs"] if j["method"] == selection["method"])
    run = root/job["run_dir"]
    result_path = root/"results/sd15/independent36k/response_scratch_seed42_1steps.result.json"
    result = load_result(root, result_path)
    validate_result(result)
    if (result["num_samples"] != 5000 or result["num_real"] != 5000 or
            result["config_hash"] != job["config_hash"] or
            result["model_identity"] != selection.get("model_identity") or
            sha256(result_path) != selection.get("result_sha256") or
            sha256(run/"stage_1/adapter.safetensors") != result["model_identity"] or
            sha256(run/"last.pt") != selection.get("checkpoint_sha256")):
        raise ValueError("Independent main model differs from its pinned full evaluation/checkpoint")
    complete = json.loads((run/"run.json").read_text())
    if complete.get("status") != "complete" or complete.get("global_step") != 36000:
        raise ValueError("Independent main model must finish all 36,000 updates")
    main_path = run/"qualitative/main_comparison.json"
    if sha256(main_path) != selection.get("qualitative_main_sha256"):
        raise ValueError("Independent main images changed after selection")
    payload = json.loads(main_path.read_text())
    reference = next(r for r in columns if r["method"] == "dmd_perceptual")
    rows = payload["rows"]
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    sample = payload["methods"][result["method"]]
    if (payload.get("protocol") != "matched-main-figure-v1" or payload.get("image_size") != 512 or
            payload["protocol_hash"] != digest or digest != reference["evaluation_prompt_hash"] or
            payload["eval_seed"] != reference["eval_seed"] or
            sample.get("model_identity") != result["model_identity"] or
            sample.get("source_result_sha256") != selection["result_sha256"] or
            sample.get("cumulative_training_updates") != 36000 or
            sample.get("sampling_steps") != 1 or sample.get("nfe") != 1 or
            [r["prompt"] for r in rows] != reference["sample_prompts"] or
            [{k:v for k,v in r.items() if k != "png_base64"} for r in sample["rows"]] != rows):
        raise ValueError("Independent main samples differ from the evaluated model or matched prompts/noise")
    for row, image in zip(rows, sample["rows"], strict=True):
        if row["source"] == "validation":
            index = row["noise_stream_index"]
            if (row["prompt"] != result["sample_prompts"][index] or
                    image["png_base64"] != result["sample_images_png_base64"][index]):
                raise ValueError("Independent main validation image differs from the full evaluation")
    replacement = {**result, "sample_prompts": [r["prompt"] for r in rows],
        "sample_image_ids": [r["key"] for r in rows], "sample_labels": [r["label"] for r in rows],
        "sample_images_png_base64": [r["png_base64"] for r in sample["rows"]],
        "evaluation_prompt_hash": digest}
    updated = [replacement if r["method"] == "dmd_perceptual_diversity" else r for r in columns]
    if sum(r["method"] == "response_scratch" for r in updated) != 1:
        raise ValueError("Main figure must have exactly one independent proposal column")
    control_path = root/"results/sd15/independent36k/dmd_lpips_scratch_seed42_1steps.result.json"
    if control_path.exists():
        control = load_result(root, control_path)
        validate_result(control)
        if control["num_samples"] != 5000 or control["num_real"] != 5000:
            raise ValueError("Independent control requires full validation")
    return updated, {**selection, "source_file": "configs/assignment2_selected_method.json",
        "result_file": str(result_path.relative_to(root)), "cumulative_training_updates": 36000,
        "independent_control_complete": control_path.exists()}


def write_main_settings(out, selection):
    from few_step_diffusion.reporting import tex_escape
    name = ("the distance extension" if selection is None else
            "the selected "+FIGURE_LABELS.get(selection["method"], selection["method"])+" candidate")
    if selection and is_direction_candidate(selection["method"]):
        name = "our direction-aligned DMD model (Ours)"
    budget = ("Both students have 36,000 training updates." if selection is None else
              f"The response model has {selection['cumulative_training_updates']:,} training updates; "
              "the fixed DMD reference has 36,000.")
    (out/"main_selection.tex").write_text(
        f"\\newcommand{{\\AMainMethodDescription}}{{{tex_escape(name)}}}\n"
        f"\\newcommand{{\\AMainBudgetNote}}{{{tex_escape(budget)}}}\n"
        "\\newcommand{\\AControlSelectionNote}{"+
        ("The complete grid was selected from four candidate seeds." if selection is None else
         "The grid uses one fixed seed and is an illustrative selection.")+"}\n"
        "\\newcommand{\\AInterpolationSelectionNote}{"+
        ("Two rows are selected from three candidate seeds." if selection is None else
         "The two rows are selected qualitative examples.")+"}\n")


def concise_quality_table(records, teacher, candidate=None):
    displayed = [records[method] for method in STANDARD_METHODS]
    if candidate is not None:
        displayed.append(candidate)
    verify_report_budget(displayed)
    rows = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"Method & Total updates & NFE & FID $\downarrow$ & IS $\uparrow$ & CLIP $\uparrow$ \\", r"\midrule"]
    for record in displayed:
        label = ("Ours (direction alignment)" if is_direction_candidate(record["method"]) else
                 ASSIGNMENT_LABELS.get(record["method"], FIGURE_LABELS.get(record["method"], record["method"])))
        values = []
        for metric in ("fid", "inception_score_mean", "clip_score_mean"):
            best = (min if metric == "fid" else max)(r[metric] for r in displayed)
            value = f"{record[metric]:.2f}"
            values.append(r"\textbf{"+value+"}" if record[metric] == best else value)
        rows.append(f"{label} & {record['cumulative_training_updates']:,} & {record['nfe']} & "+
                    " & ".join(values)+r" \\")
    rows.extend([r"\midrule", "SD1.5 teacher (64 steps) & -- & 128 & "+
                 " & ".join(f"{teacher[m]:.2f}" for m in ("fid", "inception_score_mean", "clip_score_mean"))+r" \\",
                 r"\bottomrule", r"\end{tabular}"])
    return "\n".join(rows)+"\n"


def write_main_comparison(out, comparisons, protocol, records, teacher, candidate):
    """Keep the evaluated proposal in both the main table and native sample grid."""
    methods = (*STANDARD_METHODS, candidate["method"], "teacher")
    if (len(set(methods)) != len(methods) or len(comparisons) != len(methods) or
            {r["method"] for r in comparisons} != set(methods)):
        raise ValueError("Main comparison requires five baselines, one selected proposal and the teacher")
    by_method = {r["method"]: r for r in comparisons}
    for record in [*(records[m] for m in STANDARD_METHODS), candidate, teacher]:
        if by_method[record["method"]]["model_identity"] != record["model_identity"]:
            raise ValueError("Main figure and quality table must use the same evaluated models")
    labels = {**FIGURE_LABELS, candidate["method"]: "Ours"}
    main = draw_samples([by_method[method] for method in methods],
                        list(range(len(protocol["rows"]))), out/"main_samples",
                        display_labels=labels)
    (out/"concise_quality.tex").write_text(concise_quality_table(records, teacher, candidate))
    return main


def write_selected_response(out, selection, record, reference, final_comparison=None):
    """Tie main results and appendix prose to the selected, evaluated checkpoint."""
    metrics = {"ASelectedFID": f"{record['fid']:.2f}",
               "ASelectedIS": f"{record['inception_score_mean']:.2f}",
               "ASelectedCLIP": f"{record['clip_score_mean']:.2f}",
               "ASelectedUpdates": f"{record['cumulative_training_updates']:,}",
               "ASelectedExtraUpdates": f"{record['cumulative_training_updates']-reference['cumulative_training_updates']:,}",
               "ASelectedFIDGain": f"{reference['fid']-record['fid']:.2f}",
               "ASelectedFIDGainPercent": f"{100*(reference['fid']-record['fid'])/reference['fid']:.2f}",
               "ASelectedISChange": f"{record['inception_score_mean']-reference['inception_score_mean']:.2f}",
               "ASelectedCLIPChange": f"{record['clip_score_mean']-reference['clip_score_mean']:.2f}",
               "AReferenceIS": f"{reference['inception_score_mean']:.2f}"}
    (out/"selected_response_metrics.tex").write_text("".join(
        f"\\newcommand{{\\{key}}}{{{value}}}\n" for key, value in metrics.items()))
    decision = out/"selected_response_decision.tex"
    decision.write_text("% Final checkpoint comparison awaits completed component evidence.\n")
    if selection and selection.get("study") == "independent_response_36k":
        control_status = ("The matched independent zero-response control is reported in the appendix."
            if selection.get("independent_control_complete") else
            "The matched independent zero-response control's final full-validation scores are pending.")
        with (out/"selected_response_metrics.tex").open("a") as handle:
            handle.write("\\newcommand{\\AIndependentControlStatus}{"+control_status+"}\n")
        (out/"selected_response_summary.tex").write_text(
            f"Our independently trained direction-alignment model obtains FID {record['fid']:.2f} "
            f"after 36,000 updates, compared with {reference['fid']:.2f} for the original DMD+LPIPS "
            f"reference. CLIP is {record['clip_score_mean']:.2f}, compared with "
            f"{reference['clip_score_mean']:.2f}.\n")
        weight = record["resolved_config"]["train"]["dmd_response_weight"]
        (out/"selected_response_settings.tex").write_text(
            "The main model starts from pretrained SD1.5 with fresh rank-32 student and scorer adapters. "
            f"Angular coefficient $\\lambda={weight:.6f}$ is fixed "
            "and applied from the first update. Training uses 18,000 updates on each disjoint caption pool, "
            "with two phase-specific teacher banks containing 10,000 outputs in total.\n")
        decision.write_text("The main table and qualitative figures use the independently trained 36,000-update "
            "model. The original DMD reference uses a different initialization history and teacher bank. "
            +control_status+"\n")
        return
    if not selection or not record["method"].startswith("dmd_response_"):
        shutil.copyfile(out/"response_summary.tex", out/"selected_response_summary.tex")
        (out/"selected_response_settings.tex").write_text("The evaluated strengths are listed below.\n")
        return
    total = record["cumulative_training_updates"]
    extra = total-reference["cumulative_training_updates"]
    if extra <= 0:
        raise ValueError("Selected response model must disclose its additional training budget")
    (out/"selected_response_summary.tex").write_text(
        f"Our selected direction-aligned DMD model (Ours) obtains FID {record['fid']:.2f} "
        f"after {total:,} total updates, compared with {reference['fid']:.2f} for the "
        f"36,000-update DMD reference. Its CLIP score is {record['clip_score_mean']:.2f}, "
        f"compared with {reference['clip_score_mean']:.2f}.\n")
    tr = record["resolved_config"]["train"]
    (out/"selected_response_settings.tex").write_text(
        f"Our selected model has {total:,} total updates and "
        f"angular coefficient $\\lambda={tr['dmd_response_weight']:.4f}$, calibrated to a "
        f"{100*tr['dmd_response_gradient_fraction']:g}\\% initial gradient ratio. "
        f"It receives {extra:,} more updates than the frozen DMD reference.\n")
    matches = [r for r in (final_comparison or {}).get("results", []) if r["method"] == record["method"]]
    if (len(matches) == 1 and total < matches[0]["cumulative_training_updates"] and
            record["fid"] < matches[0]["fid"]):
        final = matches[0]
        decision.write_text(
            f"The selected {total:,}-update checkpoint has lower FID ({record['fid']:.2f}) than its "
            f"{final['cumulative_training_updates']:,}-update counterpart ({final['fid']:.2f}). "
            "We retain it for the main comparison and conditioning examples below. The completed component comparison "
            "uses each arm's own final checkpoint. This checkpoint choice is exploratory "
            "validation-based selection, not an independent test-set result.\n")


def optimization_samples(root, out, comparison, studies=("dmd_conflict", "dmd_block_conflict"),
                         prefix="optimization_samples"):
    """Use only final checkpoint samples whose model identities match full evaluations."""
    expected = {record["method"]: record for record in comparison["results"]}
    columns = []
    for study in studies:
        plan = root/"configs"/f"{study}.json"
        if not plan.exists():
            continue
        manifest = json.loads(plan.read_text())
        for job in manifest["jobs"]:
            result = expected.get(job["method"])
            saved = completed_preview(root, job, result, comparison["fixed_reference"],
                                      step=manifest.get("additional_updates", 500))
            if saved is None:
                continue
            baseline, sample = saved["records"]
            if not columns:
                columns.append(baseline)
            elif baseline != columns[0]:
                raise ValueError("Optimization previews must use the same fixed DMD images")
            columns.append(sample)
    if len(columns) < 2:
        (out/f"{prefix}.tex").write_text("% Final optimization samples are pending.\n")
        for ext in ("pdf", "png"):
            (out/f"{prefix}.{ext}").unlink(missing_ok=True)
        return None
    labels = {"dmd_perceptual": "DMD + LPIPS", "dmd_conflict": "Global proj.",
              "dmd_conflict_norm_control": "Global norm", "dmd_block_conflict": "Module proj.",
              "dmd_block_norm_control": "Module norm", "dmd_noise_agreement": "Agreement",
              "dmd_noise_mean": "Paired mean", "dmd_noise_shifted": "Shifted map"}
    labels.update({method: FIGURE_LABELS[method] for method in ("dmd_pair_agreement", "dmd_pair_mean", "dmd_pair_shifted")})
    from few_step_diffusion.dmd_response import MODES as RESPONSE_MODES, SETTINGS
    labels.update({method: f"{SETTINGS[key][0].capitalize()} {SETTINGS[key][1]:.0%}"
                   for key, method in RESPONSE_MODES.items()})
    if prefix == "response_final_samples":
        labels["dmd_response_angular_50"] = "Ours (37k)"
    indices = list(range(len(columns[0]["sample_prompts"])))
    selection = draw_samples(columns, indices, out/prefix, display_labels=labels)
    selection["selection"] = "Three prompts fixed before the optimization experiments; identical noise across models"
    (out/f"{prefix}.tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        f"\\includegraphics[width=.94\\textwidth]{{generated/assignment2/{prefix}.pdf}}\n"
        "\\caption{One-step outputs from the fixed DMD+LPIPS reference and completed optimization variants. "
        "Prompts and initial noise are identical across columns and were fixed before these experiments. "
        "Training budgets are reported in the accompanying quality table. These three examples illustrate "
        "appearance and do not replace the full-validation comparison.}\n"
        f"\\label{{fig:{prefix.replace('_', '-')}}}\n\\end{{figure*}}\n")
    return selection


def build_assets(root, group_id):
    out = root/GENERATED
    out.mkdir(parents=True, exist_ok=True)
    from build_paired_comparison import write_paired_comparison
    from few_step_diffusion.recent_reporting import write_recent_results
    main_selection = json.loads((root/"configs/assignment2_selected_method.json").read_text())
    if main_selection.get("study") != "independent_response_36k":
        raise ValueError("Both reports require the completed independent 36,000-update proposal")
    if main_selection.get("study") == "independent_response_36k":
        (out/"paired_detail.tex").write_text("% Earlier pilot pairs are not relabelled as the independent model.\n")
    else:
        write_paired_comparison(root, out)
    write_recent_results(root, out)
    independent_records = write_independent_comparison(root, out)
    plan = json.loads((root/"configs/active_study.json").read_text())
    records = {}
    for job in plan["jobs"]:
        method = job["method"]
        record = load_result(root, root/job["result_dir"]/f"{method}_seed42_1steps.result.json")
        if record["method"] != method or method in records:
            raise ValueError("Duplicate or mismatched study method")
        records[method] = record
    verify_report_budget(records.values())
    if set(records) != set(LABELS):
        raise ValueError("Assignment report requires all nine completed student evaluations")
    teacher = load_result(root, root/"results/sd15/scale50k/teacher_seed0_64steps.result.json")
    verify_common([*records.values(), teacher])
    coverage = verify_validation(root, teacher)
    for method in VARIANTS:
        if "diversity_lpips_alex_mean" not in records[method]:
            raise ValueError(f"Missing completed diversity evaluation: {method}")
    write_settings(out, group_id, records)
    (out/"quality.tex").write_text(quality_table(records, teacher,
        methods=(*ASSIGNMENT_METHODS, "dmd"), labels=ASSIGNMENT_LABELS))
    (out/"ablation.tex").write_text(ablation_table(records))
    sensitivity_figure(records, out/"sensitivity")
    selected = json.loads((root/"configs/report_selection.json").read_text())
    columns = [records[m] for m in ASSIGNMENT_METHODS]+[teacher]
    comparisons, protocol = load_main_comparison(root, columns, selected["main_comparison_result"])
    if protocol["pending_methods"]:
        raise ValueError("Main comparison must include every completed baseline")
    comparisons, final_selection = select_main_candidate(root, comparisons, {})
    write_main_settings(out, final_selection)
    visual_record = (records["dmd_perceptual_diversity"] if final_selection is None else
                     {**json.loads((root/final_selection["result_file"]).read_text()),
                      "source_file": final_selection["result_file"]})
    main = write_main_comparison(out, comparisons, protocol, records, teacher, visual_record)
    write_selected_response(out, final_selection, visual_record, records["dmd_perceptual"],
                            None)
    appendix_labels = dict(FIGURE_LABELS)
    if final_selection and is_direction_candidate(visual_record["method"]):
        appendix_labels[visual_record["method"]] = "Ours"
    appendix_samples = draw_samples([r for r in comparisons if r["method"] in
        {"dmd_perceptual", visual_record["method"], "teacher"}],
        list(range(len(protocol["rows"]))), out/"response_main_samples", display_labels=appendix_labels)
    regression_rows = [r"\begin{tabular}{lrrr}", r"\toprule",
                       r"Regression & FID $\downarrow$ & IS $\uparrow$ & CLIP $\uparrow$ \\", r"\midrule"]
    for method, label in (("dmd_grouped", "Latent MSE"), ("dmd_perceptual", "LPIPS (original DMD)")):
        regression_rows.append(label+" & "+" & ".join(f"{records[method][m]:.2f}" for m in
            ("fid", "inception_score_mean", "clip_score_mean"))+r" \\")
    (out/"regression_control.tex").write_text("\n".join(regression_rows+[r"\bottomrule", r"\end{tabular}"])+"\n")
    additional = draw_samples([records["dmd_perceptual"], visual_record, teacher],
                              [2, 6], out/"additional_samples", display_labels=FIGURE_LABELS)
    for ext in ("pdf", "png"):
        (out/f"failure_samples.{ext}").unlink(missing_ok=True)
    control_options, interpolation_options = {}, {}
    if final_selection:
        folder = root/visual_record["resolved_config"]["output_dir"]/"qualitative"
        visual_selection = {**selected, "prompt_control_selection_policy":
                            "Reuse the earlier-report prompt-control seed without reselection for the final model."}
        control_options = {"folder": folder/f"seed{selected['prompt_control_eval_seed']}",
                           "selection": visual_selection}
        interpolation_options = {"path": folder/"prompt_interpolation.json", "selection": visual_selection}
    control = draw_prompt_control(root, visual_record, out, **control_options)
    interpolation = draw_interpolation(root, visual_record, out, **interpolation_options)
    if control is None or interpolation is None:
        raise ValueError("Assignment report requires completed prompt-control/interpolation panels")
    # Captions live in assignment2.tex; helper-generated captions target the older paper.
    (out/"prompt_control.tex").unlink(missing_ok=True)
    (out/"interpolation_figure.tex").unlink(missing_ok=True)
    fields = ("method", "cumulative_training_updates", "sampling_steps", "nfe", "num_samples",
              "num_real", "fid", "inception_score_mean", "clip_score_mean",
              "diversity_lpips_alex_mean", "teacher_guidance_scale", "precision", "model_identity")
    result_records = [*records.values(), teacher]
    if final_selection:
        result_records.append(visual_record)
    result_records.extend(r for r in independent_records if r["method"] != visual_record["method"])
    verify_report_budget(result_records)
    summary = [{key: r.get(key) for key in fields} for r in result_records]
    with (out/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    provenance = {"group": group_id, "coverage": coverage, "metrics": summary,
                  "evidence_policy": {"student_cumulative_updates": 36000,
                                      "full_validation_samples": 5000, "pilot_results_included": False},
                  "selected_main_candidate": final_selection,
                  "main_scope": "Five standard pipelines, the selected proposal (Ours), and the teacher; proposal design and ablations are in the appendix.",
                  "selected_appendix_candidate": final_selection, "appendix_samples": appendix_samples,
                  "main": main, "main_protocol": protocol, "additional_samples": additional,
                  "prompt_control": control, "interpolation": interpolation,
                  "source_results": [r["source_file"] for r in result_records]}
    (out/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")


def compile_report(root, group_id):
    subprocess.run(["latexmk", "-pdf", "-silent", "-halt-on-error", "-interaction=nonstopmode",
                    "-outdir=build", "assignment2.tex"], cwd=root/"report", check=True)
    source = root/"report/build/assignment2.pdf"
    info = subprocess.check_output(["pdfinfo", str(source)], text=True)
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))
    if pages < 4:
        raise ValueError(f"Assignment 2 requires at least four pages; found {pages}")
    log = (root/"report/build/assignment2.log").read_text()
    if re.search(r"(?:Citation|Reference) .+ undefined|There were undefined references", log):
        raise ValueError("Unresolved report references")
    markers = re.findall(r"^ASSIGNMENT2-MAIN-PAGES:\s*(\d+)\s*$", log, re.MULTILINE)
    if len(markers) != 1 or not 1 <= int(markers[0]) <= pages:
        raise ValueError("Missing or invalid main-report page boundary; refusing publication")
    main_pages = int(markers[0])
    if main_pages > 8:
        raise ValueError(f"The main report, including references but excluding the appendix, "
                         f"must not exceed 8 pages; found {main_pages}")
    target = root/f"CE6190_group_{group_id}_report_project2.pdf"
    temporary = target.with_suffix(".pdf.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    print(f"Assignment 2 report: {target} ({main_pages} main + {pages-main_pages} appendix pages; "
          f"{pages} total)", flush=True)
    return target


def package_code(root, group_id):
    """Local source-only submission; no data, model weights or runtime folders."""
    selected = set()
    for directory in ("src", "scripts", "configs", "schemas", "tests"):
        for path in (root/directory).rglob("*"):
            if (path.is_file() and not path.is_symlink() and
                    path.suffix in {".py", ".sh", ".yaml", ".yml", ".json"} and
                    "__pycache__" not in path.parts):
                selected.add(path)
    for name in ("README.md", "Makefile", "pyproject.toml", ".gitignore",
                 "docs/ASSIGNMENT2_REQUIREMENTS.md", "docs/ASSIGNMENT2_DEFENSE.md",
                 "docs/DMD_IMPROVEMENT_PLAN.md", "DMD_CONFLICT_RESULTS.md", "DMD_BLOCK_CONFLICT_RESULTS.md",
                 "docs/DMD_NOISE_AGREEMENT_PLAN.md", "DMD_NOISE_AGREEMENT_RESULTS.md",
                 "docs/DMD_PAIR_AGREEMENT_PLAN.md", "DMD_PAIR_AGREEMENT_RESULTS.md",
                 "docs/DMD_RESPONSE_PLAN.md", "DMD_RESPONSE_STRENGTH_RESULTS.md",
                 "artifacts/dmd_teacher_response/gradient_calibration.json",
                 "artifacts/dmd_teacher_response/frozen_reference.json",
                 "report/assignment2.tex",
                 "report/ce7454.tex", "report/main.tex",
                 "report/assets/assignment2_method_overview.png", "docs/ASSIGNMENT2_DIAGRAM_DESIGN.md",
                 "report/assignment2_references.bib", "report/cvpr.sty", "report/preamble.tex",
                 "report/ieeenat_fullname.bst", "report/THIRD_PARTY.md"):
        selected.add(root/name)
    selected.update(root.glob("requirements*.txt"))
    if (root/"DMD_RESPONSE_FINAL_RESULTS.md").exists():
        selected.add(root/"DMD_RESPONSE_FINAL_RESULTS.md")
    selected.update((root/"artifacts/dmd_response_strength/metadata_repair").glob("*.json"))
    for path in (root/"third_party").rglob("*"):
        if path.is_file() and not path.is_symlink() and any(
                word in path.name.lower() for word in ("license", "notice", "copying")):
            selected.add(path)
    selected.add(root/"docs/RECENT_BASELINES.md")
    selected.add(root/"docs/INDEPENDENT_RESPONSE.md")
    selected.update(p for p in (root/"third_party/fastgen_reference").rglob("*")
                    if p.is_file() and p.suffix in {".py", ".json"})
    selected.update(p for p in (root/GENERATED).iterdir() if p.suffix in {".tex", ".pdf", ".svg", ".csv", ".json"})
    selected.update(p for p in (root/"report/generated/ce7454").glob("*") if p.suffix in {".tex", ".json"})
    target = root/f"group_{group_id}_code_project2.zip"
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, path.relative_to(root))
        archive.writestr("ASSIGNMENT2_README.txt", (
            f"CE6190 Assignment 2 - Group {group_id}\n\n"
            "Report source: report/assignment2.tex\n"
            "Tables, measured metrics and native-model figures are included.\n"
            "Compile without Python, datasets or model checkpoints:\n"
            "  cd report\n"
            "  latexmk -pdf -outdir=build assignment2.tex\n\n"
            "The PDF submission is supplied separately. Original datasets and model\n"
            "weights are intentionally excluded. Training/evaluation code, configs,\n"
            "dependency locks, tests and third-party license notices are included.\n"
            "See README.md for environment setup and experiment commands; see\n"
            "docs/ASSIGNMENT2_REQUIREMENTS.md for the assignment coverage.\n"
            "Full asset regeneration requires the native completed evaluation JSONs;\n"
            "the included assets suffice to compile the submitted report offline.\n"))
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Submission ZIP failed its integrity check")
    os.replace(temporary, target)
    print(f"Assignment 2 code: {target} ({target.stat().st_size/1024**2:.1f} MiB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-id", default="11", type=lambda value: str(int(value)))
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--package", action="store_true", help="Also create the local code submission ZIP")
    args = parser.parse_args()
    from few_step_diffusion.runtime import experiment_lock
    with experiment_lock(ROOT/"report/build/assignment2_refresh"):
        build_submission(args)


def build_submission(args):
    original = ROOT/"CE7454_main_report.pdf"
    old_hash = hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None
    if args.compile_only:
        write_settings(ROOT/GENERATED, args.group_id)
    else:
        build_assets(ROOT, args.group_id)
    provenance = json.loads((ROOT/GENERATED/"provenance.json").read_text())
    if provenance.get("evidence_policy", {}).get("student_cumulative_updates") != 36000:
        raise ValueError("Rebuild report assets before compiling: only 36,000-update student results are allowed")
    verify_report_budget(provenance["metrics"])
    compile_report(ROOT, args.group_id)
    if old_hash and hashlib.sha256(original.read_bytes()).hexdigest() != old_hash:
        raise RuntimeError("The original paper changed during the independent assignment build")
    if (ROOT/"report/generated/ce7454/provenance.json").exists():
        from build_ce7454_report import build as build_ce7454
        build_ce7454(ROOT, refresh=False)
    if args.package:
        package_code(ROOT, args.group_id)


if __name__ == "__main__":
    main()
