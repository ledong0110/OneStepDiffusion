#!/usr/bin/env python3
"""Import the four DMD study JSONs into a dedicated CVPR experiment report."""

import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from few_step_diffusion.dmd_diversity import VARIANTS
from few_step_diffusion.figures import read_samples
from few_step_diffusion.reporting import collect_results, tex_escape

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"dmd_grouped": "DMD control", "dmd_perceptual": "+ perceptual",
          "dmd_diversity": "+ diversity", "dmd_perceptual_diversity": "Full proposal"}


def build(results, generated, teacher_diversity=None):
    records, skipped = collect_results(results, ROOT/"schemas/result.schema.json")
    by_method = {r["method"]: r for r in records}
    if set(by_method) != set(VARIANTS) or len(records) != 4:
        raise ValueError("The controlled report requires exactly one completed result for each variant")
    ordered = [by_method[method] for method in VARIANTS]
    parent_hashes = {r["continuation"]["parent_checkpoint_sha256"] for r in ordered}
    if len(parent_hashes) != 1:
        raise ValueError("DMD variants did not start from the same parent checkpoint")
    for key in ("sample_prompts", "sample_image_ids", "diversity_protocol", "diversity_prompt_hash",
                "diversity_seed_list", "diversity_lpips_identity"):
        if any(r.get(key) != ordered[0].get(key) for r in ordered):
            raise ValueError(f"DMD study qualitative/diversity protocols differ: {key}")
    if any("diversity_lpips_alex_mean" not in r for r in ordered):
        raise ValueError("Complete held-out diversity evaluation before building the DMD report")
    generated.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for r in ordered:
        regression, weight = VARIANTS[r["method"]]
        rows.append(f"{LABELS[r['method']]} & {'LPIPS' if regression == 'lpips_vgg' else 'Latent MSE'} & "
            f"{weight:g} & {r['fid']:.3f} & {r['clip_score_mean']:.3f} & "
            f"{r['diversity_lpips_alex_mean']:.4f} & {r['training_seconds']/3600:.2f}"+r" \\")
        summary.append({"method": r["method"], "fid": r["fid"], "clip": r["clip_score_mean"],
            "lpips_alex_diversity": r["diversity_lpips_alex_mean"],
            "phase_generator_updates": r["training_updates"],
            "cumulative_generator_updates": r["cumulative_training_updates"]})
    table = [r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Variant & Regression & $\lambda_d$ & FID $\downarrow$ & CLIP $\uparrow$ & Diversity & Phase h \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (generated/"quality.tex").write_text("\n".join(table)+"\n")
    with (Path(results)/"comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    reference = None
    if teacher_diversity:
        reference = json.loads(Path(teacher_diversity).read_text())
        for key in ("diversity_prompt_hash", "diversity_seed_list", "diversity_lpips_identity"):
            if reference[key] != ordered[0][key]:
                raise ValueError("Teacher and student diversity evaluation protocols differ")
    teacher_note = (f"The 64-step teacher reference has mean LPIPS-Alex distance "
                    f"{reference['diversity_lpips_alex_mean']:.4f}. " if reference else "")
    full, control = by_method["dmd_perceptual_diversity"], by_method["dmd_grouped"]
    difference = full["fid"]-control["fid"]
    measured = (f"Full-proposal minus grouped-control FID is {difference:+.4f}. "
        "This is one training seed and does not establish statistical significance. "
        +teacher_note+"Diversity is the mean pairwise LPIPS-Alex distance across four seeds for each "
        "of 128 held-out prompts; a larger value alone does not establish better generation.")
    (generated/"measured.tex").write_text(tex_escape(measured)+"\n")
    r = ordered[0]
    settings = {"DMDPhaseUpdates": f"{r['training_updates']:,}",
        "DMDCumulativeUpdates": f"{r['cumulative_training_updates']:,}",
        "DMDParentUpdates": f"{r['parent_training_updates']:,}",
        "DMDTrainingImages": f"{r['continuation']['fresh_training_prompt_count']:,}",
        "DMDEvalImages": f"{r['num_samples']:,}", "DMDGuidance": r["teacher_guidance_scale"],
        "DMDStudentRank": r["lora_rank"], "DMDTrainingSeed": r["seed"],
        "DMDPairImages": f"{r['teacher_pair_count']:,}",
        "DMDPairPrompts": f"{r['teacher_pair_unique_prompts']:,}",
        "DMDPairNFE": f"{r['teacher_pair_nfe']:,}"}
    (generated/"settings.tex").write_text("\n".join(
        f"\\newcommand{{\\{key}}}{{{tex_escape(value)}}}" for key, value in settings.items())+"\n")
    # Fixed first four evaluation outputs; columns are methods and rows share prompts/noise.
    samples = [read_samples(r)[:4] for r in ordered]
    if any(len(s) != 4 for s in samples):
        raise ValueError("Portable first-four image samples are missing")
    fig, axes = plt.subplots(4, 4, figsize=(8.1, 8.4), squeeze=False)
    for column, (r, images) in enumerate(zip(ordered, samples)):
        for row, pixels in enumerate(images):
            axes[row, column].imshow(np.asarray(pixels))
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(LABELS[r["method"]], fontsize=11)
    fig.subplots_adjust(left=.005, right=.995, bottom=.005, top=.96, wspace=.035, hspace=.035)
    fig.savefig(generated/"samples.pdf", bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    atomic = {"records": records, "skipped": skipped, "summary": summary, "teacher_reference": reference}
    (generated/"manifest.json").write_text(json.dumps(atomic, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/sd15/dmd_diversity_study")
    parser.add_argument("--teacher-diversity")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    build(Path(args.results_dir), ROOT/"report/generated/dmd_diversity", args.teacher_diversity)
    if not args.no_compile:
        subprocess.run(["latexmk", "-pdf", "-silent", "-halt-on-error", "-interaction=nonstopmode",
                        "-outdir=build", "dmd_diversity.tex"], cwd=ROOT/"report", check=True)
        info = subprocess.check_output(["pdfinfo", str(ROOT/"report/build/dmd_diversity.pdf")], text=True)
        pages = int(next(line.split(":")[1] for line in info.splitlines() if line.startswith("Pages:")))
        if pages > 4:
            raise ValueError("DMD experiment report exceeds the four-page project limit")
        print(f"Experiment report: report/build/dmd_diversity.pdf ({pages}/4 pages)")


if __name__ == "__main__":
    main()
