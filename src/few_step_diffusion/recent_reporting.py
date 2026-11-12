"""Only completed, protocol-checked recent baselines may enter the report."""
import json
import math

from .config import config_fingerprint, load_config
from .recent_objectives import SOURCE_COMMIT
from .reporting import collect_results
from .runtime import sha256
from .sd_data import caption_records, manifest_hash


def validate_result(root, path, job):
    config = load_config(root/job["config"])
    records, _ = collect_results(root, root/"schemas/result.schema.json", result_files=[path])
    if len(records) != 1:
        raise ValueError("Recent baseline needs a completed research evaluation")
    result = records[0]
    expected = {"method": job["method"], "seed": 42, "status": "complete",
                "num_samples": 5000, "num_real": 5000, "reference_split": "validation",
                "sampling_steps": 1, "nfe": 1, "lora_rank": 32, "effective_batch_size": 16,
                "cumulative_training_updates": 36000, "training_updates": 36000,
                "auxiliary_updates": 324000, "discriminator_updates": 324000,
                "source_commit": SOURCE_COMMIT, "teacher_guidance_scale": 1.75,
                "guidance_scale": 1., "precision": "bf16", "student_update_frequency": 10,
                "gan_weight": .001, "training_uses_real_images": True,
                "auxiliary_parameterization": "full-unet", "config_hash": job["config_hash"]}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Recent baseline protocol mismatch: {key}")
    if config_fingerprint(config) != job["config_hash"] or config_fingerprint(result["resolved_config"]) != job["config_hash"]:
        raise ValueError("Recent baseline config identity mismatch")
    if result["model_identity"] != sha256(root/job["run_dir"]/"stage_1/adapter.safetensors"):
        raise ValueError("Recent baseline model identity mismatch")
    train = caption_records(config)
    validation = caption_records(config, "validation", 5000)
    if (result["distillation_prompt_hash"] != manifest_hash(train) or
            result["evaluation_prompt_hash"] != manifest_hash(validation) or
            result["reference_manifest_hash"] != manifest_hash(validation)):
        raise ValueError("Recent baseline uses a different dataset or validation protocol")
    if result.get("f_divergence") != ("js" if job["method"] == "f_distill" else "reverse-kl"):
        raise ValueError("Recent baseline divergence differs from its method label")
    phases = result.get("training_data_phases", [])
    if len(phases) != 2 or any(p.get("updates") != 18000 for p in phases):
        raise ValueError("Recent baseline does not match the two 18,000-update data phases")
    for phase, rows, sampling in zip(phases, (train[:50000], train[50000:]),
                                    ("replacement", "shuffled_epochs"), strict=True):
        if phase.get("manifest_hash") != manifest_hash(rows) or phase.get("count") != len(rows) or phase.get("sampling") != sampling:
            raise ValueError("Recent baseline phase data mismatch")
    if any(not math.isfinite(result[k]) for k in ("fid", "inception_score_mean", "clip_score_mean")):
        raise ValueError("Nonfinite full-validation metric")
    return result


def write_recent_results(root, out):
    target = out/"recent_baselines.tex"
    plan_path = root/"configs/recent_baselines_36k.json"
    if not plan_path.exists():
        target.write_text("% No completed recent-baseline evaluations.\n")
        return []
    plan = json.loads(plan_path.read_text())
    completed = []
    for job in plan["jobs"]:
        path = root/job["result_dir"]/f"{job['method']}_seed42_1steps.result.json"
        if path.exists():
            completed.append(validate_result(root, path, job))
    if not completed:
        target.write_text("% No completed recent-baseline evaluations.\n")
        return []
    rows = [r"\section{Additional recent baselines}",
        r"The additional baselines use the published FastGen SD1.5 recipes for "
        r"\href{https://arxiv.org/abs/2405.14867}{DMD2} and "
        r"\href{https://arxiv.org/abs/2502.15681}{Jensen--Shannon f-distill}. "
        r"These runs use the same 50,000 and 68,287 disjoint COCO training pools, with 18,000 student updates per pool, "
        r"a rank-32 student, effective batch 16 and one-step inference. The fake scorer is fully trained. "
        r"Each student update follows nine scorer/discriminator updates (324,000 of each overall); "
        r"the discriminator uses real training images. The source settings use teacher guidance 1.75 "
        r"and adversarial weight 0.001, without paired regression. Thus data, student capacity and student updates "
        r"are matched, while teacher guidance and auxiliary compute differ from the main comparison. "
        r"These are local adaptations, not reproductions at the original paper's scale.",
        r"\begin{table}[t]\centering\small", r"\begin{tabular}{lrrr}\toprule",
        r"Method & FID $\downarrow$ & IS $\uparrow$ & CLIP $\uparrow$ \\", r"\midrule"]
    for record in completed:
        label = "DMD2" if record["method"] == "dmd2" else r"f-distill (JS)"
        rows.append(label+" & "+" & ".join(f"{record[k]:.2f}" for k in
                    ("fid", "inception_score_mean", "clip_score_mean"))+r" \\")
    rows.extend([r"\bottomrule\end{tabular}",
        r"\caption{Additional one-step baselines: 36,000 student updates and all 5,000 held-out COCO captions and references.}",
        r"\label{tab:recent}\end{table}"])
    target.write_text("\n".join(rows)+"\n")
    return [r["method"] for r in completed]
