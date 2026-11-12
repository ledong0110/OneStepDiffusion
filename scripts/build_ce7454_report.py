#!/usr/bin/env python3
"""Build the four-page CE7454 project paper from verified completed results."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def write_inputs(root):
    shared = root/"report/generated/assignment2"
    provenance = json.loads((shared/"provenance.json").read_text())
    from build_assignment2_report import verify_report_budget
    if provenance.get("evidence_policy", {}).get("student_cumulative_updates") != 36000:
        raise ValueError("Refresh shared assets using the 36,000-update-only evidence policy")
    verify_report_budget(provenance["metrics"])
    selection = provenance["selected_main_candidate"]
    if selection["method"] != "response_scratch" or selection["cumulative_training_updates"] != 36000:
        raise ValueError("CE7454 paper requires the independently trained 36k proposal")
    identity = selection["model_identity"]
    for key in ("prompt_control", "interpolation"):
        if provenance[key]["model_identity"] != identity:
            raise ValueError("CE7454 qualitative panel belongs to a different proposal")
    if next(r for r in provenance["main"]["methods"] if r["method"] == "response_scratch")["model_identity"] != identity:
        raise ValueError("CE7454 main comparison uses a different proposal")
    records = {r["method"]: r for r in provenance["metrics"]}
    out = root/"report/generated/ce7454"
    out.mkdir(parents=True, exist_ok=True)
    methods = [("baseline", "Progressive distillation"), ("consistency", "Consistency distillation"),
               ("dmd_perceptual", "DMD + LPIPS"), ("swiftbrush_lora", "SwiftBrush"),
               ("instaflow", "InstaFlow"), ("response_scratch", "Ours (direction alignment)")]
    control_path = root/"results/sd15/independent36k/dmd_lpips_scratch_seed42_1steps.result.json"
    control = None
    if control_path.exists():
        from build_active_report import load_result
        from few_step_diffusion.independent_response import validate_result
        control = load_result(root, control_path)
        proposal = load_result(root, root/selection["result_file"])
        validate_result(control)
        for key in ("teacher_identity", "eval_seed", "evaluation_prompt_hash", "reference_manifest_hash", "teacher_pair_phases"):
            if control.get(key) != proposal.get(key):
                raise ValueError(f"CE7454 matched control differs from the proposal: {key}")
        for key in ("student_adapter_identity", "fake_adapter_identity"):
            if control["initialization"][key] != proposal["initialization"][key]:
                raise ValueError("CE7454 matched control has another fresh initialization")
        records["dmd_lpips_scratch"] = control
        methods.insert(-1, ("dmd_lpips_scratch", "Matched DMD + LPIPS"))
    rows = [r"\begin{tabular}{lrrr}\toprule",
            r"Method & FID $\downarrow$ & IS $\uparrow$ & CLIP $\uparrow$ \\", r"\midrule"]
    for method, label in methods:
        record = records[method]
        if record["num_samples"] != 5000 or record["num_real"] != 5000:
            raise ValueError("CE7454 quality table requires complete full validation")
        values = []
        for metric in ("fid", "inception_score_mean", "clip_score_mean"):
            best = (min if metric == "fid" else max)(records[m][metric] for m, _ in methods)
            value = f"{record[metric]:.2f}"
            values.append(r"\textbf{"+value+"}" if record[metric] == best else value)
        rows.append(label+" & "+" & ".join(values)+r" \\")
    rows += [r"\midrule", "SD1.5 teacher (64 steps) & "+" & ".join(
        f"{records['teacher'][m]:.2f}" for m in ("fid", "inception_score_mean", "clip_score_mean"))+r" \\",
        r"\bottomrule\end{tabular}"]
    (out/"quality.tex").write_text("\n".join(rows)+"\n")
    ours, reference = records["response_scratch"], records["dmd_perceptual"]
    best_method, best_label = min(methods, key=lambda pair: records[pair[0]]["fid"])
    opening = (f"Ours obtains the lowest student FID ({ours['fid']:.2f}). " if best_method == "response_scratch" else
               f"Ours obtains FID {ours['fid']:.2f}; {best_label} has the lowest student FID ({records[best_method]['fid']:.2f}). ")
    fid_gain = reference["fid"]-ours["fid"]
    findings = (opening+f"Relative to original DMD+LPIPS (FID {reference['fid']:.2f}), "
        f"FID is lower by {fid_gain:.2f} ({100*fid_gain/reference['fid']:.2f}\\%). "
        f"IS changes from {reference['inception_score_mean']:.2f} to {ours['inception_score_mean']:.2f} "
        f"and CLIP from {reference['clip_score_mean']:.2f} to {ours['clip_score_mean']:.2f}. "
        "Both models have 36,000 student updates and one-NFE inference. "
        "The reference differs in initialization history and teacher bank, so this is a comparison of complete recipes. ")
    if control:
        findings += (f"The matched independent control has FID {control['fid']:.2f}, "
            f"IS {control['inception_score_mean']:.2f} and CLIP {control['clip_score_mean']:.2f}. "
            "This pair shares initialization, data and teacher banks; it provides a single-seed estimate of the added loss's effect. ")
    (out/"findings.tex").write_text(findings+"\n")
    status = ("The matched zero-response control uses the same initialization, teacher banks and 36,000-update schedule."
              if control else "The matched independent zero-response control's final full-validation scores are pending.")
    conclusion = ("The matched independent control provides a single-seed comparison; repeated training seeds are needed to assess reliability."
                  if control else "Matched independent controls and repeated seeds are needed to assess the size and reliability of the added loss's contribution.")
    (out/"status.tex").write_text("\\newcommand{\\CEMatchedControlStatus}{"+status+"}\n"
                                "\\newcommand{\\CEControlConclusion}{"+conclusion+"}\n")
    (out/"ablation.tex").unlink(missing_ok=True)
    (out/"provenance.json").write_text(json.dumps({"course": "CE7454", "project": 2,
        "requirements_source": "docs/project_2_handout.docx", "max_pages_including_references": 4,
        "evidence_policy": provenance["evidence_policy"],
        "table_metrics": [records[m] for m, _ in methods]+[records["teacher"]],
        "selected_model": selection, "metrics": records["response_scratch"],
        "figures": {k: provenance[k] for k in ("main", "prompt_control", "interpolation")}}, indent=2)+"\n")


def build(root=ROOT, *, refresh=True, compile_pdf=True, max_pages=4):
    from few_step_diffusion.runtime import experiment_lock
    if refresh:
        from build_assignment2_report import build_assets
        with experiment_lock(root/"report/build/assignment2_refresh"):
            build_assets(root, "11")
    with experiment_lock(root/"report/build/ce7454_refresh"):
        write_inputs(root)
        if not compile_pdf:
            return
        subprocess.run(["latexmk", "-pdf", "-silent", "-interaction=nonstopmode", "-halt-on-error",
                        "-outdir=build", "main.tex"], cwd=root/"report", check=True)
        source = root/"report/build/main.pdf"
        info = subprocess.check_output(["pdfinfo", str(source)], text=True)
        pages = int(next(line.split(":", 1)[1] for line in info.splitlines() if line.startswith("Pages:")))
        if pages != 4 or pages > max_pages:
            raise ValueError(f"CE7454 requires a four-page report including references; compiled {pages} pages")
        log = (root/"report/build/main.log").read_text(errors="replace")
        if "There were undefined references" in log:
            raise ValueError("CE7454 paper has unresolved references")
        aux = (root/"report/build/main.aux").read_text(errors="replace")
        if aux.count(r"\contentsline {table}{") != 1:
            raise ValueError("CE7454 paper must contain only its main comparison table; Table 2 was removed")
        target = root/"CE7454_main_report.pdf"
        temporary = target.with_suffix(".pdf.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
        print(f"CE7454 report: {target} ({pages} pages, including references)", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-assets", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--max-pages", type=int, default=4)
    args = parser.parse_args(argv)
    build(refresh=not args.reuse_assets, compile_pdf=not args.no_compile, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
