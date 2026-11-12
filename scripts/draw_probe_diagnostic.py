#!/usr/bin/env python3
"""Render measured score-direction diagnostics with caption-level uncertainty."""

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODES = ("agreement", "shifted_agreement")


def render(root, out):
    source = root/"artifacts/dmd_noise_agreement/probe_diagnostic/additional_32_captions.json"
    target = out/"noise_probe_diagnostic"
    if not source.exists():
        target.with_suffix(".tex").write_text("% Independent-probe diagnostic is pending.\n")
        return None
    from few_step_diffusion.runtime import sha256
    plan = json.loads((root/"configs/dmd_noise_agreement.json").read_text())
    record = json.loads(source.read_text())
    reference = json.loads((root/plan["reference_baseline"]["result_file"]).read_text())
    if (record.get("status") != "diagnostic" or not record.get("weights_unchanged") or
            record.get("optimizer_updates") != 0 or not record.get("training_pool_only") or
            record.get("checkpoint_sha256") != plan["parent_checkpoint_sha256"] or
            record.get("training_prompt_hash") != reference["distillation_prompt_hash"] or
            record.get("generator_state") != "student" or record.get("scorer_state") != "fake" or
            record.get("candidate_score_probes") != 2 or record.get("independent_reference_probes") != 16 or
            record.get("patch_size") != 8 or record.get("captions") != 32 or
            record.get("start_caption") != 4 or
            record.get("timesteps") != [100, 500, 900]):
        raise ValueError("Independent-probe diagnostic differs from its declared frozen-reference protocol")
    rows = record["measurements"]
    captions = sorted({r["caption_index"] for r in rows})
    timesteps = record["timesteps"]
    by_case = {(r["caption_index"], r["timestep"]): r for r in rows}
    if (len(rows) != 96 or len(by_case) != 96 or captions != list(range(4, 36)) or
            set(by_case) != {(caption, timestep) for caption in captions for timestep in timesteps}):
        raise ValueError("Diagnostic must include every caption/timestep combination exactly once")
    values = np.empty((32, 3, 2), dtype=np.float64)
    for i, caption in enumerate(captions):
        if len({by_case[caption, t]["prompt"] for t in timesteps}) != 1:
            raise ValueError("A diagnostic caption changed between timesteps")
        for j, timestep in enumerate(timesteps):
            modes = by_case[caption, timestep]["modes"]
            cosines = [modes[m]["cosine_to_reference"] for m in ("paired_mean", *MODES)]
            if not np.isfinite(cosines).all() or np.max(np.abs(cosines)) > 1+1e-7:
                raise ValueError("Invalid recorded direction cosine")
            for k, mode in enumerate(MODES):
                actual = modes[mode]["cosine_to_reference"]-modes["paired_mean"]["cosine_to_reference"]
                if (not np.isfinite(actual) or abs(actual) > 2 or
                        not np.isclose(actual, modes[mode]["cosine_gain_over_mean"], rtol=0, atol=1e-7)):
                    raise ValueError("Diagnostic gains do not match recorded direction cosines")
                values[i, j, k] = actual
    # Resample captions, keeping each caption's three timesteps together.
    rng = np.random.default_rng(4317)
    indices = rng.integers(0, len(captions), size=(10000, len(captions)))
    bootstrap = values[indices].mean(axis=1)
    mean = values.mean(axis=0)
    intervals = np.quantile(bootstrap, [.025, .975], axis=0)
    overall = values.mean(axis=1)
    overall_intervals = np.quantile(overall[indices].mean(axis=1), [.025, .975], axis=0)
    summary = {"source_file": str(source.relative_to(root)), "source_sha256": sha256(source),
        "checkpoint_sha256": record["checkpoint_sha256"], "captions": 32, "cases": 96,
        "timesteps": timesteps, "bootstrap_replicates": 10000, "bootstrap_seed": 4317,
        "reference_noise_pairs": 8,
        "reference_sampling": "Eight independent antithetic pairs, independent of the candidate pair",
        "resampling_unit": "caption, preserving its three timestep cases",
        "modes": {mode: {"mean_cosine_gain": float(overall[:, k].mean()),
            "caption_interval_95": overall_intervals[:, k].tolist(),
            "by_timestep": [{"timestep": t, "mean": float(mean[j, k]),
                "caption_interval_95": intervals[:, j, k].tolist(),
                "positive_cases": int((values[:, j, k] > 0).sum())} for j, t in enumerate(timesteps)]}
            for k, mode in enumerate(MODES)},
        "limitation": "Intervals describe sampled caption/noise variation at one frozen checkpoint and "
            "three fixed timesteps, not training-seed uncertainty or image quality."}
    target.with_suffix(".json").write_text(json.dumps(summary, indent=2)+"\n")
    shutil.copyfile(source, out/"noise_probe_measurements.json")
    labels, colors = ("Spatial agreement", "Shifted map"), ("#1d6b86", "#b86a36")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
        fig.subplots_adjust(left=.095, right=.99, bottom=.25, top=.80, wspace=.18)
        for j, (axis, timestep) in enumerate(zip(axes, timesteps, strict=True)):
            axis.axhline(0, color="#79838b", linestyle="--", linewidth=.8, zorder=0)
            for k, (label, color) in enumerate(zip(labels, colors, strict=True)):
                jitter = np.linspace(-.12, .12, len(captions))
                axis.scatter(k+jitter, values[:, j, k], color=color, s=8, alpha=.32, linewidths=0)
                axis.errorbar(k, mean[j, k], yerr=[[mean[j, k]-intervals[0, j, k]],
                    [intervals[1, j, k]-mean[j, k]]], color=color, marker="D", markersize=4,
                    capsize=4, linewidth=1.8, label=label, zorder=3)
            axis.set(title=f"t = {timestep}", xticks=[0, 1], xticklabels=["Agreement", "Shifted"],
                     xlim=(-.45, 1.45))
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#e6e9ec", linewidth=.5, zorder=0)
        axes[0].set_ylabel("Cosine gain over paired mean")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
                   frameon=False, bbox_to_anchor=(.55, 1.02))
        for extension in ("pdf", "svg", "png"):
            fig.savefig(target.with_suffix("."+extension), dpi=220, bbox_inches="tight", pad_inches=.03)
        plt.close(fig)
    gain = summary["modes"]["agreement"]["mean_cosine_gain"]
    shifted = summary["modes"]["shifted_agreement"]["mean_cosine_gain"]
    interval_findings = []
    for j, timestep in enumerate(timesteps):
        relation = ("lies above zero" if intervals[0, j, 0] > 0 else
                    "lies below zero" if intervals[1, j, 0] < 0 else "includes zero")
        interval_findings.append(f"at $t={timestep}$ the interval {relation}")
    target.with_suffix(".tex").write_text(
        "\\begin{figure*}[t]\n\\centering\n"
        "\\includegraphics[width=.94\\textwidth]{generated/assignment2/noise_probe_diagnostic.pdf}\n"
        "\\caption{Read-only score-direction diagnostic at the frozen DMD+LPIPS state. "
        "Each point is one of 32 additional training captions; the same caption and generated latent "
        "are tested at three noise levels. The reference direction averages eight additional "
        "independent antithetic pairs (16 probes), separate from the candidate pair. "
        "Diamonds show mean cosine gain relative to the paired mean, with 95\\% caption-bootstrap "
        "intervals. No model weights are updated. The finite-probe reference remains noisy, "
        "and direction agreement is not an image-quality metric.}\n"
        "\\label{fig:noise-probe-diagnostic}\n\\end{figure*}\n"
        "A preliminary four-caption diagnostic motivated a repeat on 32 additional training "
        "captions with a separate noise seed and unchanged filter. "
        f"Across these 96 cases, spatial agreement changes cosine alignment by {gain:+.4f}; "
        f"the shifted map changes it by {shifted:+.4f}. "
        "For spatial agreement, "+"; ".join(interval_findings)+". "
        "Intervals use 10,000 caption-level bootstrap draws, preserving "
        "the three cases per caption. They describe caption/noise variation conditional on this "
        "frozen checkpoint, not variation across training seeds. This supports testing spatial "
        "association but does not establish better FID.\n")
    return summary


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    render(project, project/"report/generated/assignment2")
