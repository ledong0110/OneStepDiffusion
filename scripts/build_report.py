#!/usr/bin/env python3
"""Import *.result.json from a shared folder, generate tables/plots, and compile CVPR PDF."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from few_step_diffusion.reporting import collect_results, write_checkpoint_evidence, write_report_inputs
from few_step_diffusion.config import dataset_name, load_config, validate
from few_step_diffusion.qualitative import load_qualitative

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir")
    p.add_argument("--profile", choices=("sd15", "celebahq", "cifar10"))
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--require-results", action="store_true", help="Fail if only smoke/example results exist")
    p.add_argument("--max-pages", type=int, default=4)
    p.add_argument("--sample-steps", type=int, default=1, help="Student NFE for the qualitative comparison")
    p.add_argument("--sample-seed", type=int, default=42, help="Training seed for all student sample rows")
    p.add_argument("--qualitative-dir", help="Portable *.qualitative.json folder; defaults to the results folder")
    p.add_argument("--study-config", help="Use this study's settings for report prose before progressive results arrive")
    p.add_argument("--checkpoint-results-dir", help="Separate matched intermediate baseline/proposed cohort")
    p.add_argument("--study", choices=("active", "legacy"), help="Default: active 36k study; explicit legacy result arguments retain the old paper")
    args = p.parse_args()
    if args.study == "active" or (args.study is None and not args.results_dir and not args.profile
                                  and not args.study_config):
        from build_active_report import main as active_report
        active_args = ["--max-pages", str(args.max_pages)]
        if args.no_compile:
            active_args.append("--no-compile")
        active_report(active_args)
        return
    default_folder = "results/full" if args.profile == "cifar10" else f"results/{args.profile or 'sd15'}/full"
    folder = Path(args.results_dir or default_folder).resolve()
    if not folder.is_dir():
        if args.require_results:
            p.error(f"Result folder does not exist: {folder}")
        print(f"No result folder yet: {folder}; building an explicitly unfinished draft")
    records, skipped = collect_results(folder, ROOT / "schemas/result.schema.json")
    if args.require_results and not records:
        p.error("No completed research results found")
    profile = args.profile or ({"CIFAR-10": "cifar10", "CelebA-HQ": "celebahq"}.get(
        records[0]["dataset"], "sd15") if records else "sd15")
    config = load_config(args.study_config or ROOT / (f"configs/{profile}.yaml" if profile != "cifar10" else "configs/base.yaml"))
    if records and any(r["dataset"] != dataset_name(config) for r in records):
        p.error("Result dataset differs from the chosen report profile")
    # Prefer an exported progressive config so pilot budgets are reflected in prose too.
    configured = next((r for r in records if r["method"] in ("baseline", "proposed", "edge_constant")
                       and r.get("resolved_config")), None)
    if configured and not args.study_config:
        config = configured["resolved_config"]
        validate(config)
        if dataset_name(config) != records[0]["dataset"]:
            p.error("Resolved experiment config does not match the result dataset")
    if args.study_config:
        for r in records:
            saved = r.get("resolved_config", {})
            if saved and saved["data"].get("train_limit") != config["data"].get("train_limit"):
                p.error("Result training-caption count differs from --study-config")
    preview_file = f"teacher_preview_{profile}.json" if profile != "cifar10" else "teacher_preview.json"
    preview_path = ROOT / "report/assets" / preview_file
    preview = json.loads(preview_path.read_text())["records"] if preview_path.exists() else []
    write_report_inputs(records, skipped, ROOT / "report/generated", args.sample_steps, args.sample_seed,
                        preview, study_config=config,
                        qualitative_records=load_qualitative(args.qualitative_dir or folder, records))
    checkpoints = Path(args.checkpoint_results_dir) if args.checkpoint_results_dir else folder.with_name(folder.name+"_8steps")
    imported_checkpoints = write_checkpoint_evidence(checkpoints, records, ROOT/"report/generated")
    if imported_checkpoints:
        print(f"Imported {len(imported_checkpoints)} separate intermediate checkpoint evaluations", flush=True)
    print(f"Imported {len(records)} evaluations; ignored {len(skipped)} nonresearch/duplicate files", flush=True)
    if args.no_compile:
        return
    if not shutil.which("latexmk"):
        p.error("latexmk missing. Install TeX Live (latex-extra, fonts-recommended, latexmk) or use --no-compile")
    subprocess.run(["latexmk", "-pdf", "-silent", "-interaction=nonstopmode", "-halt-on-error",
                    "-outdir=build", "legacy_edge.tex"], cwd=ROOT / "report", check=True)
    pdf = ROOT / "report/build/legacy_edge.pdf"
    if shutil.which("pdfinfo"):
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        pages = int(next(line.split(":", 1)[1].strip() for line in info.splitlines()
                         if line.startswith("Pages:")))
        if pages > args.max_pages:
            sys.exit(f"Compiled {pdf}, but {pages} pages exceeds the project limit of {args.max_pages}")
        print(f"Report: {pdf} ({pages}/{args.max_pages} pages)")
    else:
        print(f"Report: {pdf}; install pdfinfo to verify the page limit")
    if (ROOT/"report/generated/samples_comparison_full.pdf").is_file():
        subprocess.run(["latexmk", "-pdf", "-silent", "-interaction=nonstopmode", "-halt-on-error",
                        "-outdir=build", "legacy_qualitative.tex"], cwd=ROOT/"report", check=True)
        print(f"Legacy qualitative supplement: {ROOT/'report/build/legacy_qualitative.pdf'}")


if __name__ == "__main__":
    main()
