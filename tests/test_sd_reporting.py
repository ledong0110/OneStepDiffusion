import copy
import json
from pathlib import Path

import pytest

from few_step_diffusion.config import load_config
from few_step_diffusion.distillation_objectives import PRECONDITIONED_FLOW_SAMPLER
from few_step_diffusion.reporting import collect_results, write_checkpoint_evidence, write_report_inputs


def sd_result():
    record = json.loads(Path("examples/result.template.json").read_text())
    return {**record, "status": "complete", "method": "proposed", "sampling_steps": 1, "nfe": 1,
            "timesteps": [999], "dataset": "COCO-2017", "image_size": 512,
            "num_samples": 1000, "num_real": 1000, "reference_split": "validation",
            "sampler": "SD15-DDIM-eta0-unclipped-nested-v1", "training_updates": 3000,
            "effective_batch_size": 4, "teacher_guidance_scale": 7.5, "guidance_scale": 1,
            "denoiser_branches_per_step": 1, "lora_rank": 32,
            "prompt_protocol": "coco2017-first-caption-by-id-v1",
            "evaluation_prompt_hash": "a"*64, "reference_manifest_hash": "a"*64,
            "distillation_prompt_hash": "b"*64, "sample_prompts": ["TEST FIXTURE"],
            "sample_image_ids": [1], "latency_scope": "prompt-encoding+denoising+VAE-decoding",
            "clip_model_identity": "openai/clip-vit-base-patch32@3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
            "clip_score_definition": "100*max(cosine(image,text),0); mean over caption-image pairs",
            "clip_score_mean": 25, "clip_score_std": 1,
            "fid": 20, "inception_score_mean": 7, "inception_score_std": .1}


def test_continuation_reports_phase_budget_and_rejects_mixed_training_histories(tmp_path):
    record = sd_result()
    record.update(continuation_policy="fresh-coco-one-step-v1", continuation_only_steps=1,
        parent_training_prompt_hash="d"*64, parent_training_updates=18000,
        cumulative_training_updates=21000, continuation={"policy": "fresh-coco-one-step-v1",
            "parent_training_prompt_hash": "d"*64, "parent_training_updates": 18000,
            "parent_training_prompt_count": 50000, "fresh_training_prompt_count": 68287,
            "fresh_training_prompt_hash": record["distillation_prompt_hash"],
            "combined_unique_training_images": 118287, "training_image_overlap": 0,
            "additional_updates": 3000})
    path = tmp_path/"continued.result.json"
    path.write_text(json.dumps(record))
    records, skipped = collect_results(tmp_path, "schemas/result.schema.json")
    config = load_config("configs/sd15_50k.yaml")
    config["train"]["updates_per_stage"] = 3000
    config["data"].update(train_offset=50000, train_limit=68287)
    generated = tmp_path.parent/(tmp_path.name+"-generated")
    write_report_inputs(records, skipped, generated, study_config=config)
    assert "StudyContinuationtrue" in (generated/"study.tex").read_text()
    assert r"\StudyVSDUpdates}{3,000}" in (generated/"study.tex").read_text()
    assert "68,287 fresh training images" in (generated/"protocol.tex").read_text()
    baseline = sd_result()
    baseline["method"] = "baseline"
    (tmp_path/"scratch.result.json").write_text(json.dumps(baseline))
    with pytest.raises(ValueError, match="Incompatible"):
        collect_results(tmp_path, "schemas/result.schema.json")
    (tmp_path/"scratch.result.json").unlink()
    record["cumulative_training_updates"] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="continuation"):
        collect_results(tmp_path, "schemas/result.schema.json")


def test_continued_instaflow_counts_only_new_endpoint_couplings(tmp_path):
    record = sd_result()
    record.update(method="instaflow", sampler=PRECONDITIONED_FLOW_SAMPLER,
        student_prediction_type="epsilon-converted-velocity", teacher_pair_count=68287,
        reflow_updates=0, flow_distillation_updates=3000, teacher_pair_nfe=0,
        reflow_pair_nfe=68287*record["initial_steps"], flow_time_mapping="nearest-VP-SNR",
        continuation_policy="fresh-coco-one-step-v1", continuation_only_steps=1,
        parent_training_prompt_hash="d"*64, parent_training_updates=18000,
        cumulative_training_updates=21000, continuation={"policy": "fresh-coco-one-step-v1",
            "parent_training_prompt_hash": "d"*64, "parent_training_updates": 18000,
            "parent_training_prompt_count": 50000, "fresh_training_prompt_count": 68287,
            "fresh_training_prompt_hash": record["distillation_prompt_hash"],
            "combined_unique_training_images": 118287, "training_image_overlap": 0,
            "additional_updates": 3000})
    (tmp_path/"flow.result.json").write_text(json.dumps(record))
    records, _ = collect_results(tmp_path, "schemas/result.schema.json")
    assert len(records) == 1


def test_sd_import_counts_cfg_and_renders_alignment(tmp_path):
    student = sd_result()
    teacher = {**student, "method": "teacher", "seed": 0, "nfe": 2, "training_updates": 0,
               "effective_batch_size": 0, "denoiser_branches_per_step": 2, "guidance_scale": 7.5}
    (tmp_path/"friend.json").write_text(json.dumps(student))
    (tmp_path/"teacher.json").write_text(json.dumps(teacher))
    records, skipped = collect_results(tmp_path, "schemas/result.schema.json")
    output = tmp_path.parent / (tmp_path.name+"-generated")
    summary = write_report_inputs(records, skipped, output)
    assert {r["nfe"] for r in summary} == {1, 2}
    table = (output/"quality_table.tex").read_text()
    assert "CLIP" in table and "1/2" in table and "1/1" in table
    teacher["nfe"] = 1
    (tmp_path/"teacher.json").write_text(json.dumps(teacher))
    with pytest.raises(ValueError, match="NFE"):
        collect_results(tmp_path, "schemas/result.schema.json")


@pytest.mark.parametrize("field,value", [("evaluation_prompt_hash", "c"*64),
                                        ("teacher_guidance_scale", 6),
                                        ("lora_rank", 8), ("distillation_prompt_hash", "c"*64)])
def test_sd_comparison_rejects_changed_prompts_guidance_or_adapters(tmp_path, field, value):
    original = sd_result()
    changed = copy.deepcopy(original)
    changed.update(seed=43)
    changed[field] = value
    (tmp_path/"a.json").write_text(json.dumps(original))
    (tmp_path/"b.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Incompatible"):
        collect_results(tmp_path, "schemas/result.schema.json")


def test_sd_does_not_silently_mix_full_generators_with_lora_study(tmp_path):
    original = sd_result()
    full = {**original, "method": "swiftbrush_source", "student_parameterization": "full-unet",
            "lora_rank": 0, "trainable_parameters": 859520964,
            "student_prediction_type": "swiftbrush-source-x0", "sampler": "SD15-SwiftBrush-source-x0-v1",
            "source_commit": "a346fec56484a5de54e062b1b4251f1af1e0bce0", "terminal_alpha_cumprod": .0047}
    (tmp_path/"lora.json").write_text(json.dumps(original))
    (tmp_path/"full.json").write_text(json.dumps(full))
    with pytest.raises(ValueError, match="Incompatible|parameterization"):
        collect_results(tmp_path, "schemas/result.schema.json")


def test_intermediate_checkpoint_evidence_keeps_budgets_separate_and_rejects_data_mismatch(tmp_path):
    from few_step_diffusion.diffusion import make_grid, nested_grid
    cohort = tmp_path/"checkpoint-results"
    cohort.mkdir()
    one_step = sd_result()
    intermediate = {**one_step, "sampling_steps": 8, "nfe": 8, "comparison_steps": 8,
                    "training_updates": 9000, "timesteps": nested_grid(make_grid(1000, 64), 8).tolist()}
    for method in ("baseline", "proposed"):
        (cohort/f"{method}.json").write_text(json.dumps({**intermediate, "method": method}))
    output = tmp_path/"generated"
    imported = write_checkpoint_evidence(cohort, [one_step], output)
    assert len(imported) == 2
    text = (output/"checkpoint_evidence.tex").read_text()
    assert "8 steps (8 NFE)" in text and "9,000 generator updates" in text
    assert "separate from the one-step" in text
    with pytest.raises(ValueError, match="differs from the main study"):
        write_checkpoint_evidence(cohort, [{**one_step, "evaluation_prompt_hash": "c"*64}], output)
    (cohort/"proposed.json").unlink()
    assert write_checkpoint_evidence(cohort, [one_step], output) == []
    assert not (output/"checkpoint_evidence.json").exists()


def test_consistency_uses_its_own_sampler_under_the_shared_data_protocol(tmp_path):
    original = sd_result()
    consistency = {**original, "method": "consistency", "consistency_sigma_data": .5,
                   "student_prediction_type": "consistency-x0", "sampler": "SD15-consistency-boundary-v1"}
    (tmp_path/"pd.json").write_text(json.dumps(original))
    (tmp_path/"cd.json").write_text(json.dumps(consistency))
    records, _ = collect_results(tmp_path, "schemas/result.schema.json")
    assert len(records) == 2
    consistency["sampler"] = original["sampler"]
    (tmp_path/"cd.json").write_text(json.dumps(consistency))
    with pytest.raises(ValueError):
        collect_results(tmp_path, "schemas/result.schema.json")


def test_dmd_needs_regression_provenance_and_consistent_pair_budget(tmp_path):
    dmd = {**sd_result(), "method": "dmd", "dmd_regression_weight": .25,
           "teacher_pair_count": 256, "teacher_pair_nfe": 32768,
           "teacher_pair_generation_seconds": 20, "paired_regression_batch_size": 4}
    (tmp_path/"dmd.json").write_text(json.dumps(dmd))
    records, _ = collect_results(tmp_path, "schemas/result.schema.json")
    assert records[0]["teacher_pair_nfe"] == 32768
    dmd.pop("teacher_pair_count")
    (tmp_path/"dmd.json").write_text(json.dumps(dmd))
    with pytest.raises(ValueError):
        collect_results(tmp_path, "schemas/result.schema.json")


def test_instaflow_requires_both_phase_budgets_and_pair_generation_costs(tmp_path):
    flow = {**sd_result(), "method": "instaflow", "student_prediction_type": "rectified-flow-velocity",
            "sampler": "SD15-rectified-flow-euler-v1", "teacher_pair_count": 256,
            "teacher_pair_nfe": 32768, "reflow_pair_nfe": 16384,
            "reflow_updates": 1500, "flow_distillation_updates": 1500,
            "flow_time_mapping": "round(999*(1-progress))"}
    (tmp_path/"flow.json").write_text(json.dumps(flow))
    (tmp_path/"pd.json").write_text(json.dumps(sd_result()))
    records, _ = collect_results(tmp_path, "schemas/result.schema.json")
    assert len(records) == 2
    flow["reflow_updates"] = 3000
    (tmp_path/"flow.json").write_text(json.dumps(flow))
    with pytest.raises(ValueError, match="two phases"):
        collect_results(tmp_path, "schemas/result.schema.json")


def test_matched_comparison_rejects_different_student_generation_steps(tmp_path):
    record = {**sd_result(), "comparison_steps": 1, "sampling_steps": 4, "nfe": 4,
              "timesteps": [999, 745, 492, 238]}
    (tmp_path/"wrong_steps.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="matched-comparison"):
        collect_results(tmp_path, "schemas/result.schema.json")


def test_ablation_table_uses_matched_seeds_and_keeps_teacher_as_reference(tmp_path):
    from few_step_diffusion.reporting import EDGE_SETTINGS
    records = []
    for i, (method, (weight, gate)) in enumerate(EDGE_SETTINGS.items()):
        records.append({**sd_result(), "method": method, "edge_weight": weight,
                        "edge_schedule": "noise_aware" if gate == "off" else gate,
                        "comparison_steps": 1, "fid": 20+i, "seed": 42})
    records.append({**records[-1], "seed": 43, "fid": 100})
    teacher = {**sd_result(), "method": "teacher", "seed": 0, "nfe": 2, "training_updates": 0,
               "effective_batch_size": 0, "denoiser_branches_per_step": 2,
               "guidance_scale": 7.5, "comparison_steps": 1}
    records.append(teacher)
    for i, record in enumerate(records):
        (tmp_path/f"result_{i}.json").write_text(json.dumps(record))
    imported, skipped = collect_results(tmp_path, "schemas/result.schema.json")
    output = tmp_path.parent/(tmp_path.name+"-generated")
    write_report_inputs(imported, skipped, output)
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["ablations"]["training_seeds"] == [42]
    assert manifest["ablations"]["variants"][-1]["fid_mean"] == 24
    table = (output/"ablation_table.tex").read_text()
    assert "0.03" in table and "0.3" in table and "100.00" not in table
    assert "different inference budgets" in (output/"quality_table.tex").read_text()
