import copy
import json

import pytest
import torch
from safetensors.torch import load_file

from few_step_diffusion import dmd_conflict as conflict
from few_step_diffusion import dmd_diversity, sd_training, training
from few_step_diffusion import dmd_noise_agreement as noise
from few_step_diffusion import dmd_response as response
from few_step_diffusion.config import load_config, validate
from few_step_diffusion.runtime import sha256
from test_continuation import annotations
from test_dmd_diversity import TinyDistance
from test_sd_training import TinyBundle, tiny_unet


def combine(score, perceptual, mode):
    p = torch.nn.Parameter(torch.zeros(len(score)))
    acc = conflict.GradientAccumulator([p], ["test.to_q.lora_A.default.weight"])
    acc.add((p*torch.tensor(score)).sum(), (p*torch.tensor(perceptual)).sum(), 1)
    stats = acc.assign(mode, .25)
    return p.grad, stats


def test_project_removes_only_opposing_component_and_norm_control_matches_size():
    projected, stats = combine([2., 0.], [-3., 4.], "project")
    torch.testing.assert_close(projected, torch.tensor([2., 1.]))
    assert stats["gradient_conflict"] == 1
    assert stats["gradient_cosine"] == pytest.approx(-.6)
    shrunk, _ = combine([2., 0.], [-3., 4.], "norm_match")
    torch.testing.assert_close(shrunk, torch.tensor([1.4, .8]))
    torch.testing.assert_close((projected-torch.tensor([2., 0.])).norm(),
                               (shrunk-torch.tensor([2., 0.])).norm())
    for mode in conflict.MODES:
        aligned, _ = combine([2., 0.], [3., 4.], mode)
        torch.testing.assert_close(aligned, torch.tensor([2.75, 1.]))
        zero, _ = combine([0., 0.], [-3., 4.], mode)
        torch.testing.assert_close(zero, torch.tensor([-.75, 1.]))
        empty, _ = combine([0., 0.], [0., 0.], mode)
        assert torch.isfinite(empty).all()
    with pytest.raises(FloatingPointError):
        combine([float("nan"), 1.], [1., 1.], "project")


def test_project_uses_whole_batch_and_global_parameter_dot_product():
    a, b = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    acc = conflict.GradientAccumulator([a, b])
    # First microbatch conflicts, second outweighs it. Full-batch dot is positive.
    acc.add(2*a.sum()+0*b.sum(), -3*a.sum()+4*b.sum(), 2)
    acc.add(2*a.sum()+0*b.sum(), 5*a.sum()-2*b.sum(), 2)
    stats = acc.assign("project", .25)
    assert stats["gradient_conflict"] == 0
    torch.testing.assert_close(a.grad, torch.tensor([2.25]))
    torch.testing.assert_close(b.grad, torch.tensor([.25]))


def test_sum_matches_weighted_loss_backward_and_skips_unused_parameters():
    p = torch.nn.Parameter(torch.tensor([.3, -.2]))
    unused = torch.nn.Parameter(torch.ones(1))
    acc = conflict.GradientAccumulator([p, unused])
    acc.add(p.square().sum(), p.sin().sum(), 1)
    acc.assign("sum", .25)
    expected = p.grad.clone()
    p.grad = None
    (p.square().sum()+.25*p.sin().sum()).backward()
    torch.testing.assert_close(p.grad, expected)
    assert unused.grad is None


def test_module_projection_preserves_aligned_modules_and_groups_lora_factors():
    names = ["a.to_q.lora_A.default.weight", "a.to_q.lora_B.default.weight",
             "b.to_v.lora_A.default.weight"]
    def apply(mode):
        parameters = [torch.nn.Parameter(torch.zeros(1)) for _ in names]
        acc = conflict.GradientAccumulator(parameters, names)
        # Module a: dot=-4, ||gD||^2=2. Module b: dot=+6.
        a, b, c = parameters
        acc.add(a.sum()+b.sum()+2*c.sum(), -2*a.sum()-2*b.sum()+3*c.sum(), 1)
        statistics = acc.assign(mode, .25)
        return torch.cat([p.grad for p in parameters]), statistics
    projected, statistics = apply("block_project")
    torch.testing.assert_close(projected, torch.tensor([1., 1., 2.75]))
    assert statistics["gradient_conflict"] == 0
    assert statistics["gradient_block_count"] == 2
    assert statistics["gradient_block_conflict_fraction"] == .5
    # Module a is exactly opposed, so both projection and matched norm give zero LPIPS.
    shrunk, _ = apply("block_norm_match")
    torch.testing.assert_close(shrunk, projected)
    global_update, _ = apply("project")
    torch.testing.assert_close(global_update, torch.tensor([.5, .5, 2.75]))


def test_module_projection_refuses_unnamed_groups():
    p = torch.nn.Parameter(torch.zeros(2))
    acc = conflict.GradientAccumulator([p])
    acc.add(p.sum(), -p.sum(), 1)
    with pytest.raises(ValueError, match="named LoRA"):
        acc.assign("block_project", .25)


def test_result_import_requires_matched_parent_and_real_refinement_budget(tmp_path):
    from few_step_diffusion.reporting import collect_results
    from test_sd_reporting import sd_result
    records = []
    for mode, method in conflict.MODES.items():
        record = sd_result()
        record.update(method=method, training_updates=500, cumulative_training_updates=36500,
            dmd_gradient_mode=mode, dmd_regression_type="lpips_vgg", dmd_diversity_weight=0.,
            refinement_policy=conflict.POLICY, refinement_updates=500,
            refinement_parent_checkpoint_sha256="c"*64, teacher_pair_protocol_hash="d"*64,
            teacher_pair_weights_sha256="e"*64,
            dmd_refinement={"policy": conflict.POLICY, "parent_training_updates": 36000,
                "parent_checkpoint_sha256": "c"*64, "additional_updates": 500,
                "training_prompt_hash": record["distillation_prompt_hash"]})
        (tmp_path/f"{method}.result.json").write_text(json.dumps(record))
        records.append(record)
    imported, skipped = collect_results(tmp_path, "schemas/result.schema.json")
    assert len(imported) == len(conflict.MODES) and not skipped
    record = records[1]
    record["cumulative_training_updates"] = 36000
    path = tmp_path/f"{record['method']}.result.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="refinement"):
        collect_results(tmp_path, "schemas/result.schema.json")
    record["cumulative_training_updates"] = 36500
    record["refinement_parent_checkpoint_sha256"] = "f"*64
    record["dmd_refinement"]["parent_checkpoint_sha256"] = "f"*64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Incompatible evaluation protocols"):
        collect_results(tmp_path, "schemas/result.schema.json")


@pytest.mark.parametrize("family,mode", [("gradient", mode) for mode in conflict.MODES]+
                         [("noise", mode) for mode in noise.MODES]+[("pair", mode) for mode in noise.PAIR_MODES]+
                         [("response", mode) for mode in ("angular_30", "magnitude_30", "vector_30")])
def test_refinement_preserves_parent_and_resumes_exactly(tmp_path, monkeypatch, family, mode):
    annotations(tmp_path/"data")
    parent = tmp_path/"parent"
    config = load_config("configs/dmd.yaml")
    config.update(device="cpu", precision="fp32", progress_bar=False, output_dir=str(parent))
    config["model"].update(lora_rank=2, gradient_checkpointing=True)
    config["data"].update(root=str(tmp_path/"data"), train_limit=2, batch_size=2, download=False)
    config["train"].update(initial_steps=4, updates_per_stage=2, grad_accumulation=2,
        save_every=1, log_every=1, dmd_pair_count=2, dmd_pair_seeds=2, warmup_steps=0,
        dmd_regression_type="lpips_vgg", dmd_diversity_weight=0.,
        dmd_pair_cache_dir=str(tmp_path/"pairs"))
    config["wandb"]["mode"] = "disabled"
    monkeypatch.setattr(sd_training, "SDBundle", TinyBundle)
    monkeypatch.setattr(sd_training, "new_unet", tiny_unet)
    monkeypatch.setattr(sd_training, "preview", lambda *args: None)
    monkeypatch.setattr(dmd_diversity, "load_lpips", lambda *args: TinyDistance())
    training.train(config)
    parent_hash = sha256(parent/"last.pt")
    is_noise = family in ("noise", "pair")
    module = response if family == "response" else noise if is_noise else conflict
    options = {"noise_pairs": 2} if family == "pair" else {}
    if family == "response":
        calibration = tmp_path/"calibration.json"
        calibration.write_text(json.dumps({"policy": "frozen-teacher-finite-response-diagnostic-v1",
            "parent_checkpoint_sha256": parent_hash, "training_only": True, "weights_unchanged": True,
            "optimizer_updates": 0, "gradient_calibration": {"coefficients": {
                kind: {"weights": {"0.3": .2}} for kind in ("angular", "magnitude", "vector")}}}))
        options.update(calibration_file=str(calibration), calibration_sha256=sha256(calibration))
    fresh = module.refinement_config(config, parent, tmp_path/"fresh", mode, parent_hash,
                                     updates=2, warmup_steps=0, **options)
    (tmp_path/"fresh").mkdir()
    state = conflict.initialize_refinement(fresh, tmp_path/"fresh")
    original = torch.load(parent/"last.pt", weights_only=False)
    for key in ("student", "ema", "fake"):
        for name in original[key]:
            torch.testing.assert_close(state[key][name], original[key][name], rtol=0, atol=0)
    for key in ("optimizer", "fake_optimizer"):
        assert state[key]["param_groups"] == original[key]["param_groups"]
        for index, values in original[key]["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(state[key]["state"][index][name], value, rtol=0, atol=0)
    torch.testing.assert_close(state["torch_rng"], original["torch_rng"], rtol=0, atol=0)
    assert fresh["dmd_refinement"]["data_update_offset"] == 2
    training.train(fresh)
    metadata = json.loads((tmp_path/"fresh/stage_1/metadata.json").read_text())
    assert metadata["cumulative_training_updates"] == 4
    assert metadata["cumulative_auxiliary_updates"] == 4
    if family == "response":
        assert metadata["dmd_response_variant"] == mode
        assert metadata["dmd_response_weight"] == .2
        assert metadata["dmd_pair_seeds"] == 2
        assert metadata["teacher_pair_unique_prompts"] == 1
        logs = [json.loads(line) for line in (tmp_path/"fresh/train.jsonl").read_text().splitlines()]
        assert all(row["response_valid_pairs"] == 1 for row in logs)
        assert any(row["dmd_response_loss"] > 0 for row in logs)
    else:
        assert metadata["dmd_noise_mode" if is_noise else "dmd_gradient_mode"] == mode
    if is_noise:
        assert metadata["dmd_score_probes"] == (4 if family == "pair" else 2)
        logs = [json.loads(line) for line in (tmp_path/"fresh/train.jsonl").read_text().splitlines()]
        assert all(row["score_filtered_norm"] == pytest.approx(row["score_paired_mean_norm"], rel=1e-5)
                   for row in logs)
        if mode != "paired_mean":
            assert any(row["score_filter_relative_change"] > 0 for row in logs)
    assert metadata["dmd_regression_type"] == "lpips_vgg"
    assert "continuation" not in metadata
    resumed = copy.deepcopy(fresh)
    resumed["output_dir"] = str(tmp_path/"resumed")
    save = sd_training.save_state
    # Shorten only the save interval for the interruption test; training is identical.
    factory = module.refinement_config
    def test_factory(*args, **kwargs):
        result = factory(*args, **kwargs)
        result["train"]["save_every"] = 1
        return result
    monkeypatch.setattr(module, "refinement_config", test_factory)
    resumed["train"]["save_every"] = 1
    def save_and_stop(*args, **kwargs):
        save(*args, **kwargs)
        if args[7] == 1:
            raise InterruptedError("interrupted after first saved update")
    monkeypatch.setattr(sd_training, "save_state", save_and_stop)
    with pytest.raises(InterruptedError):
        training.train(resumed)
    monkeypatch.setattr(sd_training, "save_state", save)
    training.train(resumed, resume=True)
    expected = load_file(str(tmp_path/"fresh/stage_1/adapter.safetensors"))
    actual = load_file(str(tmp_path/"resumed/stage_1/adapter.safetensors"))
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    final = torch.load(tmp_path/"fresh/last.pt", weights_only=False)
    restarted = torch.load(tmp_path/"resumed/last.pt", weights_only=False)
    for key in ("student", "fake"):
        for name in final[key]:
            torch.testing.assert_close(restarted[key][name], final[key][name], rtol=0, atol=0)
    assert sha256(parent/"last.pt") == parent_hash
    altered = copy.deepcopy(fresh)
    altered["train"]["dmd_diversity_weight"] = .1
    with pytest.raises(ValueError, match="Response alignment" if family == "response" else
                       "Noise-agreement" if is_noise else "gradient combination"):
        validate(altered)
