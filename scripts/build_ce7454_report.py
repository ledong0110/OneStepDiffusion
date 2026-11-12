#!/usr/bin/env python3
"""Build the four-page CE7454 project paper from verified completed results."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def write_analysis_tables(out, records, proposal):
    """Summarize completed measurements and recorded training exposure."""
    from build_assignment2_report import verify_report_budget
    ours, reference = records["response_scratch"], records["dmd_perceptual"]
    verify_report_budget([ours, reference, proposal])
    changes = []
    rows = [r"\begin{tabular}{lrr}\toprule",
            r"Metric & Score change & Relative change (\%) \\", r"\midrule"]
    for metric, label in (("fid", r"FID $\downarrow$"),
                          ("inception_score_mean", r"IS $\uparrow$"),
                          ("clip_score_mean", r"CLIP $\uparrow$")):
        delta = ours[metric]-reference[metric]
        relative = 100*delta/reference[metric]
        changes.append({"metric": metric, "difference": delta, "percent": relative})
        rows.append(f"{label} & {delta:+.2f} & {relative:+.2f}"+r" \\")
    (out/"metric_changes.tex").write_text("\n".join(rows+[r"\bottomrule\end{tabular}"])+"\n")

    phases = proposal["training_data_phases"]
    banks = proposal["teacher_pair_phases"]
    batch = proposal["effective_batch_size"]
    if (len(phases) != 2 or len(banks) != 2 or batch != 16 or
            any(p["updates"] != 18000 for p in phases) or
            sum(p["updates"] for p in phases) != proposal["cumulative_training_updates"] or
            sum(b["teacher_pair_count"] for b in banks) != proposal["cumulative_teacher_pair_count"]):
        raise ValueError("Exposure table requires the recorded two-phase 36k protocol")
    exposure = []
    rows = [r"\begin{tabular}{lrrr}\toprule",
            r"Phase & Caption pool & DM draws & Teacher images \\", r"\midrule"]
    for index, (phase, bank) in enumerate(zip(phases, banks, strict=True), 1):
        item = {"phase": index, "captions": phase["count"], "updates": phase["updates"],
                "caption_draws": phase["updates"]*batch, "teacher_images": bank["teacher_pair_count"]}
        exposure.append(item)
        rows.append(f"{index} & {item['captions']:,} & {item['caption_draws']:,} & {item['teacher_images']:,}"+r" \\")
    rows += [r"\midrule", "Total & "+" & ".join(f"{sum(p[k] for p in exposure):,}" for k in
            ("captions", "caption_draws", "teacher_images"))+r" \\", r"\bottomrule\end{tabular}"]
    (out/"training_exposure.tex").write_text("\n".join(rows)+"\n")
    return {"metric_changes": changes, "training_exposure": exposure}


def check_result_tables(aux):
    expected = {"tab:quality", "tab:metric_changes", "tab:exposure"}
    labels = set(re.findall(r"\\newlabel\{(tab:[^}@]+)\}", aux))
    if labels != expected or aux.count(r"\contentsline {table}{") != len(expected):
        raise ValueError("CE7454 requires the quality, metric-change and exposure tables; regression-control tables are excluded")


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
    from build_active_report import load_result
    from few_step_diffusion.independent_response import validate_result
    proposal = load_result(root, root/selection["result_file"])
    validate_result(proposal)
    if proposal["model_identity"] != identity or any(proposal[k] != records["response_scratch"][k] for k in
            ("fid", "inception_score_mean", "clip_score_mean")):
        raise ValueError("Analysis tables differ from the selected full-validation model")
    out = root/"report/generated/ce7454"
    out.mkdir(parents=True, exist_ok=True)
    methods = [("baseline", "Progressive distillation"), ("consistency", "Consistency distillation"),
               ("dmd_perceptual", "DMD + LPIPS"), ("swiftbrush_lora", "SwiftBrush"),
               ("instaflow", "InstaFlow"), ("response_scratch", "Ours (direction alignment)")]
    control_path = root/"results/sd15/independent36k/dmd_lpips_scratch_seed42_1steps.result.json"
    control = None
    if control_path.exists():
        control = load_result(root, control_path)
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
    analysis_tables = write_analysis_tables(out, records, proposal)
    ours, reference = records["response_scratch"], records["dmd_perceptual"]
    best_method, best_label = min(methods, key=lambda pair: records[pair[0]]["fid"])
    opening = (f"Ours obtains the lowest student FID ({ours['fid']:.2f}). " if best_method == "response_scratch" else
               f"Ours obtains FID {ours['fid']:.2f}; {best_label} has the lowest student FID ({records[best_method]['fid']:.2f}). ")
    fid_gain = reference["fid"]-ours["fid"]
    findings = (opening+r"Table~\ref{tab:metric_changes} separates the FID gain from the IS and CLIP trade-offs. "
        "Both models have 36,000 student updates and one-NFE inference. "
        "The reference differs in initialization history and teacher bank, so this is a comparison of complete recipes. ")
    if control:
        findings += (f"The matched independent control has FID {control['fid']:.2f}, "
            f"IS {control['inception_score_mean']:.2f} and CLIP {control['clip_score_mean']:.2f}. "
            "This pair shares initialization, data and teacher banks; it provides a single-seed estimate of the added loss's effect. ")
    (out/"findings.tex").write_text(findings+"\n")
    teacher = records["teacher"]
    reference_gap = reference["fid"]-teacher["fid"]
    proposal_gap = ours["fid"]-teacher["fid"]
    if reference_gap <= 0:
        raise ValueError("Teacher-gap analysis requires a positive reference-to-teacher FID gap")
    (out/"teacher_gap.tex").write_text(
        f"Ours remains {proposal_gap:.2f} FID points above the teacher ({teacher['fid']:.2f}), "
        f"compared with a {reference_gap:.2f}-point gap for the original DMD+LPIPS reference. "
        f"The {fid_gain:.2f}-point difference between the two student recipes is "
        f"{100*fid_gain/reference_gap:.2f}\\% of that reference gap.\n")
    (out/"baseline_context.tex").write_text(
        f"DMD+LPIPS is the strongest of the five standard pipelines in this evaluation "
        f"(FID {reference['fid']:.2f}), ahead of SwiftBrush ({records['swiftbrush_lora']['fid']:.2f}) "
        f"and progressive distillation ({records['baseline']['fid']:.2f}).\n")
    status = ("The matched zero-response control uses the same initialization, teacher banks and 36,000-update schedule."
              if control else "The matched independent zero-response control's final full-validation scores are pending.")
    conclusion = ("The matched independent control provides a single-seed comparison; repeated training seeds are needed to assess reliability."
                  if control else "Matched independent controls and repeated seeds are needed to assess the size and reliability of the added loss's contribution.")
    (out/"status.tex").write_text("\\newcommand{\\CEMatchedControlStatus}{"+status+"}\n"
                                "\\newcommand{\\CEControlConclusion}{"+conclusion+"}\n")
    (out/"ablation.tex").unlink(missing_ok=True)
    (out/"provenance.json").write_text(json.dumps({"course": "CE7454", "project": 2,
        "requirements_source": "docs/project_2_handout.docx", "max_main_text_pages": 4,
        "references_count_toward_limit": False,
        "page_policy_source": "User clarification: references do not count toward the four-page main text",
        "evidence_policy": provenance["evidence_policy"],
        "table_metrics": [records[m] for m, _ in methods]+[records["teacher"]],
        "analysis_tables": analysis_tables,
        "selected_model": selection, "metrics": records["response_scratch"],
        "figures": {k: provenance[k] for k in ("main", "prompt_control", "interpolation")}}, indent=2)+"\n")


def check_page_budget(log, total_pages, max_pages=4):
    """Count all scientific content before the flushed references boundary."""
    markers = re.findall(r"^CE7454-MAIN-PAGES:\s*(\d+)\s*$", log, re.MULTILINE)
    if len(markers) != 1 or not 1 <= int(markers[0]) < total_pages:
        raise ValueError("Missing or invalid main-text/references page boundary; refusing publication")
    main_pages = int(markers[0])
    if main_pages > max_pages:
        raise ValueError(f"CE7454 main text has {main_pages} pages, exceeding {max_pages}; references are excluded")
    return main_pages, total_pages-main_pages


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
        log = (root/"report/build/main.log").read_text(errors="replace")
        main_pages, reference_pages = check_page_budget(log, pages, max_pages)
        if "There were undefined references" in log:
            raise ValueError("CE7454 paper has unresolved references")
        aux = (root/"report/build/main.aux").read_text(errors="replace")
        check_result_tables(aux)
        target = root/"CE7454_main_report.pdf"
        temporary = target.with_suffix(".pdf.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
        metadata_path = root/"report/generated/ce7454/provenance.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["pages"] = {"main_text": main_pages, "references": reference_pages, "total": pages}
        metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
        print(f"CE7454 report: {target} ({main_pages} main + {reference_pages} references pages)", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-assets", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--max-pages", type=int, default=4)
    args = parser.parse_args(argv)
    build(refresh=not args.reuse_assets, compile_pdf=not args.no_compile, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
