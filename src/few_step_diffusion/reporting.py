"""Validate team results, keep protocols separate, and produce LaTeX inputs."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import jsonschema
import numpy as np
from .figures import METHOD_LABELS, make_sample_figure, make_qualitative_panels
from .distillation_objectives import CONSISTENCY_SAMPLER, FLOW_SAMPLER, PRECONDITIONED_FLOW_SAMPLER

PROTOCOL_KEYS = ("dataset", "image_size", "num_samples", "num_real", "reference_split",
                 "eval_seed", "sampler", "precision", "metric_implementation",
                 "prompt_protocol", "evaluation_prompt_hash", "reference_manifest_hash",
                 "distillation_prompt_hash", "teacher_guidance_scale", "lora_rank",
                 "clip_model_identity", "clip_score_definition", "latency_scope", "comparison_steps",
                 "continuation_policy", "parent_training_prompt_hash", "parent_training_updates",
                 "dmd_study_policy", "dmd_study_parent_config_hash", "teacher_pair_protocol_hash",
                 "teacher_pair_weights_sha256")

ABLATION_METHODS = {"edge_weight_low", "edge_weight_high"}
EDGE_SETTINGS = {"baseline": (0.0, "off"), "edge_constant": (.1, "constant"),
                 "edge_weight_low": (.03, "noise_aware"), "proposed": (.1, "noise_aware"),
                 "edge_weight_high": (.3, "noise_aware")}


def protocol_value(record, key):
    # Consistency's endpoint mapping is an algorithm being compared, not a data
    # mismatch. Validate that method's sampler separately before normalizing it.
    if key == "sampler" and record["dataset"] == "COCO-2017":
        return "SD15-shared-teacher-evaluation-v1"
    return record.get(key)


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def reject_constant(value):
    raise ValueError(f"Nonstandard numeric constant in JSON: {value}")


def collect_results(folder, schema_path, *, result_files=None):
    schema = json.loads(Path(schema_path).read_text())
    validator = jsonschema.Draft202012Validator(schema)
    records, skipped, seen = [], [], {}
    paths = Path(folder).rglob("*.json") if result_files is None else map(Path, result_files)
    for path in sorted(paths):
        try:
            record = json.loads(path.read_text(), parse_constant=reject_constant)
            if not isinstance(record, dict):
                raise ValueError("A result must be a JSON object")
            if record.get("status") in ("smoke", "example", "qualitative"):
                skipped.append({"file": str(path), "reason": record["status"]})
                continue
            validator.validate(record)
            for key in ("fid", "inception_score_mean", "inception_score_std"):
                if record[key] is None or not math.isfinite(record[key]):
                    raise ValueError(f"Complete result needs a finite numeric {key}")
            sd = record["dataset"] == "COCO-2017"
            if record["sampler"] != ("SD15-SwiftBrush-source-x0-v1" if sd and record.get("student_prediction_type") == "swiftbrush-source-x0" else
                                      CONSISTENCY_SAMPLER if sd and record["method"] == "consistency" else
                                      PRECONDITIONED_FLOW_SAMPLER if sd and record["method"] == "instaflow" and record.get("student_prediction_type") == "epsilon-converted-velocity" else
                                      FLOW_SAMPLER if sd and record["method"] == "instaflow" else
                                      "SD15-DDIM-eta0-unclipped-nested-v1" if sd else
                                      "DDIM-eta0-unclipped-nested-v1"):
                raise ValueError("Sampler does not match the model/data protocol")
            branches = 2 if sd and record["method"] == "teacher" else 1
            if record["nfe"] != branches*record["sampling_steps"]:
                raise ValueError("NFE must count the actual conditional/unconditional denoiser branches")
            if record.get("comparison_steps") is not None and record["method"] != "teacher":
                if record["sampling_steps"] != record["comparison_steps"]:
                    raise ValueError("Student sampling steps differ from the matched-comparison protocol")
            if sd:
                if record["method"].startswith("dmd_"):
                    from .dmd_diversity import STUDY_POLICY, VARIANTS
                    if record["method"] not in VARIANTS:
                        raise ValueError("Unknown DMD study variant")
                    regression, diversity_weight = VARIANTS[record["method"]]
                    if (record.get("dmd_study_policy") != STUDY_POLICY or
                            record.get("dmd_regression_type") != regression or
                            record.get("dmd_diversity_weight") != diversity_weight or
                            record.get("dmd_pair_seeds") != 2 or
                            record.get("teacher_pair_unique_prompts", 0)*2 != record["teacher_pair_count"] or
                            record.get("paired_regression_batch_size") != record["effective_batch_size"] or
                            record.get("teacher_pair_nfe") != record["teacher_pair_count"]*record["initial_steps"]*2 or
                            not record.get("teacher_pair_protocol_hash") or
                            not record.get("teacher_pair_weights_sha256")):
                        raise ValueError("Invalid matched DMD perceptual/diversity study settings")
                if record.get("continuation_policy"):
                    lineage = record.get("continuation", {})
                    if (record["continuation_policy"] != "fresh-coco-one-step-v1" or
                            lineage.get("policy") != record["continuation_policy"] or
                            lineage.get("training_image_overlap") != 0 or
                            record.get("continuation_only_steps") != 1 or record["sampling_steps"] != 1 or
                            lineage.get("fresh_training_prompt_hash") != record["distillation_prompt_hash"] or
                            lineage.get("parent_training_prompt_hash") != record.get("parent_training_prompt_hash") or
                            lineage.get("parent_training_updates") != record.get("parent_training_updates") or
                            lineage.get("additional_updates") != record["training_updates"] or
                            record.get("cumulative_training_updates") != record["parent_training_updates"]+record["training_updates"] or
                            lineage.get("combined_unique_training_images") !=
                            lineage.get("parent_training_prompt_count", 0)+lineage.get("fresh_training_prompt_count", 0)):
                        raise ValueError("Invalid fresh-data one-step continuation lineage or cumulative budget")
                if record["guidance_scale"] != (record["teacher_guidance_scale"] if branches == 2 else 1):
                    raise ValueError("Teacher CFG or baked-guidance student configuration is invalid")
                if record["denoiser_branches_per_step"] != branches:
                    raise ValueError("Invalid denoiser branch count")
                if any(record[k] is None or not math.isfinite(record[k]) for k in
                       ("clip_score_mean", "clip_score_std")):
                    raise ValueError("Complete text-to-image results need a finite CLIP alignment score")
                if record["method"] == "instaflow":
                    if record["reflow_updates"]+record["flow_distillation_updates"] != record["training_updates"]:
                        raise ValueError("InstaFlow's two phases must sum to the matched generator budget")
                    endpoint_only = record.get("continuation_policy") == "fresh-coco-one-step-v1"
                    expected_teacher_nfe = 0 if endpoint_only else record["teacher_pair_count"]*record["initial_steps"]*2
                    if endpoint_only and record["reflow_updates"] != 0:
                        raise ValueError("Endpoint continuation must not restart reflow training")
                    if record["teacher_pair_nfe"] != expected_teacher_nfe:
                        raise ValueError("Invalid guided teacher-pair NFE")
                    if record["reflow_pair_nfe"] != record["teacher_pair_count"]*record["initial_steps"]:
                        raise ValueError("Invalid reflow teacher-pair NFE")
            grid = record["timesteps"]
            if len(grid) != record["sampling_steps"] or any(a <= b for a, b in zip(grid, grid[1:])):
                raise ValueError("Invalid inference grid")
            sizes = {"CIFAR-10": (32, {"train": 50000, "test": 10000}),
                     "CelebA-HQ": (256, {"train": 28000, "validation": 2000}),
                     "COCO-2017": (512, {"validation": 5000})}
            size, splits = sizes[record["dataset"]]
            if record["image_size"] != size or record["reference_split"] not in splits:
                raise ValueError("Dataset, resolution, and reference split do not match")
            if record["num_real"] > splits[record["reference_split"]]:
                raise ValueError("Reference count exceeds the dataset split size")
            key = (record["method"], record["seed"], record["sampling_steps"],
                   tuple(protocol_value(record, k) for k in PROTOCOL_KEYS))
            if key in seen:
                previous, original_path = seen[key]
                if record != previous:
                    raise ValueError(f"Conflicting duplicate run; also found in {original_path}")
                skipped.append({"file": str(path), "reason": "identical duplicate"})
                continue
            seen[key] = (record, path)
            records.append({**record, "source_file": str(path)})
        except (ValueError, jsonschema.ValidationError) as exc:
            raise ValueError(f"Invalid result {path}: {exc}") from exc
    protocols = {tuple(protocol_value(r, k) for k in PROTOCOL_KEYS) for r in records}
    if len(protocols) > 1:
        differences = [k for k in PROTOCOL_KEYS if len({protocol_value(r, k) for r in records}) > 1]
        raise ValueError("Incompatible evaluation protocols: " + ", ".join(differences) +
                         ". Put pilot and final results in separate folders.")
    budgets = defaultdict(set)
    grids = defaultdict(set)
    capacities = {r.get("student_parameterization", "lora") for r in records if r["method"] != "teacher"}
    if len(capacities) > 1:
        raise ValueError("Student parameterizations differ; keep full-U-Net source recipes separate from LoRA studies")
    for r in records:
        if r["method"] != "teacher":
            budgets[r["sampling_steps"]].add((r["training_updates"], r.get("effective_batch_size")))
        grids[r["sampling_steps"]].add(tuple(r["timesteps"]))
    if any(len(v) > 1 for v in budgets.values()):
        raise ValueError("Training budgets differ at the same sampling step count")
    if any(len(v) > 1 for v in grids.values()):
        raise ValueError("Inference grids differ at the same sampling step count")
    for key in ("teacher_identity", "initial_steps"):
        if len({r.get(key) for r in records}) > 1:
            raise ValueError(f"Incompatible {key} values across results")
    for method, fields in (("consistency", ("consistency_sigma_data", "student_prediction_type")),
                           ("dmd", ("dmd_regression_weight", "teacher_pair_count", "paired_regression_batch_size")),
                           ("instaflow", ("teacher_pair_count", "reflow_updates", "flow_distillation_updates", "flow_time_mapping", "student_prediction_type"))):
        trials = [r for r in records if r["method"] == method]
        for field in fields:
            if len({r.get(field) for r in trials}) > 1:
                raise ValueError(f"Incompatible {field} within {method}")
    return records, skipped


def mean_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None


def formatted(values):
    mean, std = mean_std(values)
    return f"{mean:.2f}" if std is None else f"${mean:.2f} \\pm {std:.2f}$"


def write_checkpoint_evidence(folder, primary_records, output_dir):
    """Import an intermediate cohort separately from the final one-step tables."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out/"checkpoint_evidence.tex"
    if not folder or not Path(folder).is_dir():
        target.write_text("% No intermediate checkpoint cohort available.\n")
        (out/"checkpoint_evidence.json").unlink(missing_ok=True)
        return []
    schema = Path(__file__).resolve().parents[2]/"schemas/result.schema.json"
    records, _ = collect_results(folder, schema)
    selected = {r["method"]: r for r in records if r["method"] in ("baseline", "proposed") and r["seed"] == 42}
    if set(selected) != {"baseline", "proposed"}:
        target.write_text("% Matched intermediate baseline/proposed results pending.\n")
        (out/"checkpoint_evidence.json").unlink(missing_ok=True)
        return []
    shared = [key for key in PROTOCOL_KEYS if key != "comparison_steps"]+["teacher_identity", "initial_steps"]
    for checkpoint in selected.values():
        if checkpoint["sampling_steps"] != checkpoint.get("comparison_steps"):
            raise ValueError("Intermediate checkpoint budget is not explicitly recorded")
        for primary in primary_records:
            if any(protocol_value(checkpoint, key) != protocol_value(primary, key) for key in shared):
                raise ValueError("Intermediate checkpoint protocol differs from the main study")
    baseline, proposed = selected["baseline"], selected["proposed"]
    steps, nfe, updates = baseline["sampling_steps"], baseline["nfe"], baseline["training_updates"]
    text = (r"\paragraph{Intermediate checkpoints.} "
            f"At {steps} steps ({nfe} NFE), after {updates:,} generator updates, "
            f"FID is {baseline['fid']:.2f} for PD and {proposed['fid']:.2f} for the proposed loss "
            f"on the same {baseline['num_samples']:,} COCO prompts and {baseline['num_real']:,} references "
            f"(training seed 42). These intermediate results are separate from the one-step comparison.\n")
    target.write_text(text)
    from .runtime import atomic_json
    atomic_json(out/"checkpoint_evidence.json", {
        "steps": steps, "nfe": nfe, "training_updates": updates,
        "num_samples": baseline["num_samples"], "num_real": baseline["num_real"],
        "seed": 42, "results": [{"method": r["method"], "fid": r["fid"],
            "source_file": r["source_file"], "model_identity": r["model_identity"]}
            for r in (baseline, proposed)]})
    return [baseline, proposed]


def write_study_config(records, output_dir, config=None):
    from .config import dataset_name, image_size, load_config

    if config is None:
        profile = {"COCO-2017": "sd15", "CelebA-HQ": "celebahq"}.get(
            records[0]["dataset"] if records else "CIFAR-10", "base")
        root = Path(__file__).resolve().parents[2]
        config = load_config(root / f"configs/{profile}.yaml")
    name, size = dataset_name(config), image_size(config)
    celeba = name == "CelebA-HQ"
    sd = name == "COCO-2017"
    tr = config["train"]
    continuation_record = next((r for r in records if r.get("continuation")), {})
    continuation = bool(continuation_record or config.get("continuation"))
    study_seeds = sorted({r["seed"] for r in records if r["method"] != "teacher"}) or config.get("study_seeds", [42, 43, 44])
    mantissa, exponent = f"{tr['learning_rate']:.1e}".split("e")
    macros = {"StudyDataset": name, "StudyResolution": size,
              "StudyTeacher": config["model"]["source"],
              "StudyTrainCount": f"{config['data']['train_limit']:,}" if sd else "28,000" if celeba else "50,000",
              "StudyReferenceCount": f"{records[0]['num_real']:,}" if records else
                                     f"{config['eval']['num_real']:,}" if sd else "2,000" if celeba else "10,000",
              "StudyReferenceSplit": records[0]["reference_split"] if records else "validation" if sd or celeba else "test",
              "StudyInitialSteps": records[0]["initial_steps"] if records else tr["initial_steps"],
              "StudySeeds": ", ".join(map(str, study_seeds)),
              "StudyStageUpdates": f"{tr['updates_per_stage']:,}",
              "StudyVSDUpdates": f"{tr['updates_per_stage']*(1 if continuation else len(tr['stages'])):,}",
              "StudyTrainOffset": f"{config['data'].get('train_offset', 0):,}",
              "StudyParentUpdates": f"{continuation_record.get('parent_training_updates', 0):,}",
              "StudyTotalUpdates": f"{continuation_record.get('cumulative_training_updates', 0):,}",
              "StudyBatchSize": config["data"]["batch_size"]*tr["grad_accumulation"],
              "StudyMicroBatch": config["data"]["batch_size"],
              "StudyAccumulation": tr["grad_accumulation"],
              "StudyEMADecay": tr["ema_decay"], "StudyLoRARank": config["model"].get("lora_rank", 0),
              "StudyGuidance": config["model"].get("guidance_scale", 1)}
    lines = [r"\newif\ifStudySD", r"\StudySDtrue" if sd else r"\StudySDfalse"]
    lines += [r"\newif\ifStudyContinuation", r"\StudyContinuationtrue" if continuation else r"\StudyContinuationfalse"]
    flow_record = next((r for r in records if r["method"] == "instaflow"), {})
    dmd_record = next((r for r in records if r["method"] == "dmd"), {})
    converted_flow = flow_record.get("student_prediction_type", tr.get("instaflow_prediction_type")) == "epsilon-converted-velocity"
    lines += [r"\newif\ifStudyFlowVP", r"\StudyFlowVPtrue" if converted_flow else r"\StudyFlowVPfalse"]
    macros["StudyDMDPairs"] = dmd_record.get("teacher_pair_count", tr.get("dmd_pair_count", 256))
    macros["StudyFlowPairs"] = flow_record.get("teacher_pair_count", tr.get("instaflow_pair_count", 256))
    lines.extend(f"\\newcommand{{\\{key}}}{{{tex_escape(value)}}}" for key, value in macros.items())
    curriculum = [tr["initial_steps"]] + tr["stages"]
    lines.append(r"\newcommand{\StudyCurriculum}{" + r"\rightarrow".join(map(str, curriculum)) + "}")
    lines.append(f"\\newcommand{{\\StudyLearningRate}}{{{float(mantissa):g}\\times10^{{{int(exponent)}}}}}")
    note = ("COCO validation is held out from student training; overlap with the teacher's "
            "LAION pretraining data is not established. This subset protocol is not the "
            "COCO-2014 benchmark used in other papers." if sd else
            "The pinned dataset mirror provides a fixed train/validation split. Validation is "
            "held out from our progressive student training; its exposure during teacher pretraining "
            "is not established. This is not an official unseen-teacher test protocol."
            if celeba else "The pretrained teacher has already seen the CIFAR-10 training split.")
    lines.append(f"\\newcommand{{\\StudySplitNote}}{{{tex_escape(note)}}}")
    Path(output_dir, "study.tex").write_text("\n".join(lines)+"\n")


def write_report_inputs(records, skipped, output_dir, sample_steps=1, sample_seed=42, teacher_preview=(),
                        study_config=None, qualitative_records=()):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_study_config(records, out, study_config)
    sd = bool(records and records[0]["dataset"] == "COCO-2017")
    if not records and study_config:
        from .config import is_stable_diffusion
        sd = is_stable_diffusion(study_config)
    groups = defaultdict(list)
    for record in records:
        groups[(record["method"], record["sampling_steps"])].append(record)
    rows, teacher_rows, summary = [], [], []
    for (method, steps), trials in sorted(groups.items(), key=lambda x: (x[0][0], -x[0][1])):
        nfe = trials[0]["nfe"]
        if any(r["nfe"] != nfe for r in trials):
            raise ValueError("NFE differs across seeds of the same method")
        main_budget = (not sd or steps == 1 or
                       method == "teacher" and steps == max(r["sampling_steps"] for r in records if r["method"] == "teacher"))
        if method not in ABLATION_METHODS and main_budget:
            target_rows = teacher_rows if method == "teacher" else rows
            target_rows.append(f"{tex_escape(METHOD_LABELS.get(method, method))} & {steps}/{nfe} & {len(trials)} & " +
                        formatted([r["fid"] for r in trials]) + " & " +
                        formatted([r["inception_score_mean"] for r in trials]) +
                        (" & " + formatted([r["clip_score_mean"] for r in trials]) if sd else "") + r" \\")
        fid_mean, fid_std = mean_std([r["fid"] for r in trials])
        is_mean, is_std = mean_std([r["inception_score_mean"] for r in trials])
        clip_mean, clip_std = mean_std([r["clip_score_mean"] for r in trials]) if sd else (None, None)
        summary.append({"method": method, "steps": steps, "nfe": nfe, "seeds": len(trials),
                        "fid_mean": fid_mean, "fid_std": fid_std, "is_mean": is_mean,
                        "is_std": is_std, "clip_mean": clip_mean, "clip_std": clip_std})
    if rows:
        rows.insert(0, f"\\multicolumn{{{6 if sd else 5}}}{{l}}{{Trained students: compare equal Steps/NFE}}" + r" \\")
    if teacher_rows:
        rows += ([r"\midrule"] if rows else []) + [f"\\multicolumn{{{6 if sd else 5}}}{{l}}{{Teacher references: different inference budgets}}" + r" \\", *teacher_rows]
    if not rows:
        rows = [f"\\multicolumn{{{6 if sd else 5}}}{{c}}{{No completed research results yet.}}" + r" \\"]
    table = [r"\begin{tabular}{lrrrrr}" if sd else r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Method & Steps/NFE & Seeds & FID $\downarrow$ & IS $\uparrow$" +
             (r" & CLIP $\uparrow$" if sd else "") + r" \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (out / "quality_table.tex").write_text("\n".join(table) + "\n")
    timing_groups = defaultdict(list)
    for r in records:
        if r["method"] in ABLATION_METHODS:
            continue
        if sd and r["sampling_steps"] != 1 and not (r["method"] == "teacher" and
                r["sampling_steps"] == max(t["sampling_steps"] for t in records if t["method"] == "teacher")):
            continue
        key = (r["environment"]["hardware"], r["benchmark_batch_size"], r["method"], r["nfe"])
        timing_groups[key].append(r["latency_ms_per_image"])
    timing_rows = [f"{tex_escape(hw)} & {batch} & {tex_escape(method)} & {steps} & {formatted(values)}" + r" \\"
                   for (hw, batch, method, steps), values in sorted(timing_groups.items())]
    if not timing_rows:
        timing_rows = [r"\multicolumn{5}{c}{No runtime measurements yet.} \\"]
    (out / "latency_table.tex").write_text("\n".join([
        r"\begin{tabular}{llrrr}", r"\toprule", r"Hardware & Batch & Method & NFE & ms/image \\",
        r"\midrule", *timing_rows, r"\bottomrule", r"\end{tabular}"]) + "\n")
    cost_groups = defaultdict(list)
    for r in records:
        if r["method"] != "teacher" and r["method"] not in ABLATION_METHODS and r["sampling_steps"] == 1:
            cost_groups[(r["method"], r.get("training_hardware") or "unrecorded")].append(r)
    cost_rows = []
    for (method, hardware), trials in sorted(cost_groups.items()):
        updates = {r["training_updates"] for r in trials}
        auxiliary = {r.get("auxiliary_updates", 0) for r in trials}
        if len(auxiliary) != 1:
            raise ValueError(f"Auxiliary update budgets differ within {method}")
        hours = formatted([r.get("training_seconds", 0)/3600 for r in trials])
        pair_nfe = {r.get("teacher_pair_nfe", 0)+r.get("reflow_pair_nfe", 0) for r in trials}
        if len(pair_nfe) != 1:
            raise ValueError(f"Teacher-pair generation budgets differ within {method}")
        cost_rows.append(f"{tex_escape(METHOD_LABELS.get(method, method))} & "
                         f"{tex_escape(hardware)} & {next(iter(updates))} & "
                         f"{next(iter(auxiliary))} & {next(iter(pair_nfe))} & {hours}" + r" \\")
    if not cost_rows:
        cost_rows = [r"\multicolumn{6}{c}{One-step training cost results pending.} \\"]
    (out / "cost_table.tex").write_text("\n".join([
        r"\begin{tabular}{llrrrr}", r"\toprule", r"Method & Train hardware & G & F & Pair NFE & Hours \\",
        r"\midrule", *cost_rows, r"\bottomrule", r"\end{tabular}"]) + "\n")
    if records:
        r = records[0]
        protocol = (f"{r['dataset']}, {r['image_size']} by {r['image_size']} RGB; "
                    f"{r['num_samples']:,} generated images versus {r['num_real']:,} "
                    f"reference images from the {r['reference_split']} split; evaluation seed "
                    f"{r['eval_seed']}; {r['precision']}; {r['metric_implementation']}. "
                    "Error bars and plus/minus values indicate sample standard deviation across "
                    "training seeds, not Inception Score split variation. Single-seed entries have no error bar.")
        if sd:
            protocol += " Fixed COCO captions and reference preprocessing use the recorded manifest hashes."
        if r.get("comparison_steps") is not None:
            protocol += (f" The trained-method comparison uses exactly {r['comparison_steps']} sampling "
                         "step(s) per image; teacher rows are separate references.")
        if r.get("continuation"):
            lineage = r["continuation"]
            protocol += (f" This cohort continues completed one-step checkpoints with "
                f"{r['training_updates']:,} additional updates on {lineage['fresh_training_prompt_count']:,} "
                f"fresh training images; {lineage['training_image_overlap']} overlap with the original "
                f"{lineage['parent_training_prompt_count']:,}. Cumulative generator updates: "
                f"{r['cumulative_training_updates']:,}.")
    else:
        protocol = "No completed research results have been imported. Smoke and example outputs are excluded."
    (out / "protocol.tex").write_text(tex_escape(protocol) + "\n")
    (out / "result_status.tex").write_text(
        f"{len(records)} completed evaluation records; {len(skipped)} excluded nonmetric/duplicate files.\n")
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "steps", "nfe", "seeds", "fid_mean", "fid_std", "is_mean", "is_std", "clip_mean", "clip_std"])
        writer.writeheader()
        writer.writerows(summary)
    samples = make_sample_figure(records, out, sample_steps, sample_seed, teacher_preview)
    controlled_samples = make_qualitative_panels(qualitative_records, out, sample_seed)
    ablations = write_ablation_inputs(groups, out, sd)
    manifest = {"records": records, "skipped": skipped, "summary": summary, "qualitative_rows": samples,
                "ablations": ablations, "controlled_qualitative": controlled_samples}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    make_plots([r for r in summary if r["method"] not in ABLATION_METHODS], timing_groups, out)
    return summary


def write_ablation_inputs(groups, out, sd):
    section = out/"ablation_section.tex"
    if not any(method in ABLATION_METHODS for method, _ in groups):
        section.write_text("% No completed edge-weight sweep results yet.\n")
        (out/"ablation_table.tex").write_text("% Ablations pending.\n")
        return {}
    steps_by_method = {method: {steps for name, steps in groups if name == method}
                       for method in EDGE_SETTINGS}
    common_steps = set.intersection(*steps_by_method.values())
    selected_steps, seeds = None, set()
    for steps in sorted(common_steps):
        common_seeds = set.intersection(*({r["seed"] for r in groups[(method, steps)]}
                                         for method in EDGE_SETTINGS))
        if common_seeds:
            selected_steps, seeds = steps, common_seeds
            break
    if selected_steps is None:
        section.write_text(r"\paragraph{Ablations.} Controlled edge-loss comparisons await matching "
                           "steps and training seeds for all five settings.\n")
        return {"status": "pending", "reason": "missing matched steps or seeds"}
    rows, summaries = [], []
    for method, (weight, gate) in EDGE_SETTINGS.items():
        trials = [r for r in groups[(method, selected_steps)] if r["seed"] in seeds]
        for record in trials:
            if "edge_weight" in record and not math.isclose(record["edge_weight"], weight):
                raise ValueError(f"Edge-weight metadata differs from the named ablation: {method}")
            if gate != "off" and record.get("edge_schedule", gate) != gate:
                raise ValueError(f"Edge gate metadata differs from the named ablation: {method}")
        gate_label = {"off": "off", "constant": "constant", "noise_aware": r"$\alpha_t^2$"}[gate]
        label = {"baseline": "PD", "edge_constant": "Constant edge"}.get(method, "Noise-aware edge")
        score = "clip_score_mean" if sd else "inception_score_mean"
        rows.append(f"{label} & {weight:g} & {gate_label} & {len(seeds)} & " +
                    formatted([r["fid"] for r in trials]) + " & " +
                    formatted([r[score] for r in trials]) + r" \\")
        summaries.append({"method": method, "edge_weight": weight, "edge_schedule": gate,
                          "fid_mean": float(np.mean([r["fid"] for r in trials]))})
    (out/"ablation_table.tex").write_text("\n".join([
        r"\begin{tabular}{lrlrrr}", r"\toprule",
        r"Variant & $\lambda$ & Gate & Seeds & FID $\downarrow$ & " +
        (r"CLIP $\uparrow$" if sd else r"IS $\uparrow$") + r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])+"\n")
    section.write_text(
        "\\begin{table}[t]\n\\centering\\scriptsize\n"
        "\\resizebox{\\linewidth}{!}{\\input{generated/ablation_table.tex}}\n"
        f"\\caption{{Controlled edge-loss ablations at {selected_steps} sampling step(s). "
        "All rows use the same training seeds, generator update budget, and evaluation inputs. "
        "Only edge weight and gate vary.}\n\\label{tab:ablations}\n\\end{table}\n")
    return {"status": "complete", "sampling_steps": selected_steps,
            "training_seeds": sorted(seeds), "variants": summaries}


def make_plots(summary, timing_groups, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"teacher": "#505050", "baseline": "#2878b5", "edge_constant": "#9a69b5",
              "proposed": "#c43832", "swiftbrush_vsd": "#36975b", "consistency": "#db8d1a", "dmd": "#248c95"}
    with plt.rc_context({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(5.5, 3))
        for method in sorted({r["method"] for r in summary}):
            values = sorted((r for r in summary if r["method"] == method), key=lambda x: x["steps"])
            ax.errorbar([r["nfe"] for r in values], [r["fid_mean"] for r in values],
                        yerr=[r["fid_std"] or 0 for r in values], marker="o", capsize=3,
                        color=colors.get(method), label=METHOD_LABELS.get(method, method))
        ax.set(xlabel="Denoiser evaluations (NFE)", ylabel="FID (lower is better)")
        if summary:
            ax.set_xscale("log", base=2)
            ax.legend(fontsize=7)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "Results pending", ha="center", transform=ax.transAxes)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "quality_vs_steps.pdf")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.5, 3))
        for hw, batch, method in sorted({key[:3] for key in timing_groups}):
            values = sorted((steps, mean_std(v)[0]) for (h, b, m, steps), v in timing_groups.items()
                            if (h, b, m) == (hw, batch, method))
            ax.plot([s for s, _ in values], [v for _, v in values], "o-",
                    label=f"{method}, {hw}, B={batch}")
        ax.set(xlabel="Denoiser evaluations (NFE)", ylabel="Median milliseconds / image")
        if timing_groups:
            ax.legend(fontsize=6)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "Runtime measurements pending", ha="center", transform=ax.transAxes)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out / "latency_vs_steps.pdf")
        plt.close(fig)
