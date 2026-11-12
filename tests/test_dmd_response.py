import pytest
import torch
import copy
import json

from few_step_diffusion.dmd_response import response_loss


def test_calibration_pinning_and_response_configuration(tmp_path):
    from few_step_diffusion.dmd_response import refinement_config
    from few_step_diffusion.config import load_config, validate
    from few_step_diffusion.runtime import sha256
    parent = load_config("configs/dmd_perceptual.yaml")
    calibration = tmp_path/"calibration.json"
    calibration.write_text(json.dumps({"policy": "frozen-teacher-finite-response-diagnostic-v1",
        "parent_checkpoint_sha256": "c"*64, "training_only": True, "weights_unchanged": True,
        "optimizer_updates": 0, "gradient_calibration": {"coefficients": {
            "angular": {"weights": {"0.3": .18}}}}}))
    digest = sha256(calibration)
    config = refinement_config(parent, tmp_path/"parent", str(tmp_path/"child"), "angular_30", "c"*64,
        calibration_file=calibration, calibration_sha256=digest)
    assert config["train"]["dmd_response_weight"] == .18
    for field, value in (("dmd_pair_seeds", 1), ("dmd_response_weight", float("nan")),
                         ("dmd_noise_mode", "agreement"), ("dmd_gradient_mode", "sum")):
        altered = copy.deepcopy(config)
        altered["train"][field] = value
        with pytest.raises(ValueError):
            validate(altered)
    calibration.write_text(calibration.read_text()+"\n")
    with pytest.raises(ValueError, match="calibration changed"):
        refinement_config(parent, tmp_path/"parent", str(tmp_path/"child"), "angular_30", "c"*64,
            calibration_file=calibration, calibration_sha256=digest)


def test_response_result_import_preserves_loss_and_budget_identity(tmp_path):
    from few_step_diffusion.dmd_response import POLICY
    from few_step_diffusion.config import config_fingerprint
    from few_step_diffusion.reporting import collect_results
    from test_active_report import result_fixture
    record = result_fixture("dmd_perceptual")
    for field in ("continuation", "continuation_policy", "continuation_only_steps", "dmd_study_policy",
                  "dmd_study_parent_config_hash", "parent_training_updates", "parent_training_prompt_hash"):
        record.pop(field, None)
    fields = dict(dmd_response_mode="angular", dmd_response_variant="angular_30",
                  dmd_response_weight=.18, dmd_response_gradient_fraction=.3)
    record["resolved_config"]["name"] = "dmd_response_angular_30"
    record["resolved_config"]["train"].update(fields, mode="dmd", updates_per_stage=500,
        dmd_pair_seeds=2, dmd_regression_type="lpips_vgg", dmd_diversity_weight=0.)
    record.update(fields, method="dmd_response_angular_30", training_updates=500,
        cumulative_training_updates=36500, refinement_policy=POLICY, refinement_updates=500,
        refinement_parent_checkpoint_sha256="c"*64,
        config_hash=config_fingerprint(record["resolved_config"]),
        dmd_refinement={"policy": POLICY, "parent_training_updates": 36000, "additional_updates": 500,
            "parent_checkpoint_sha256": "c"*64, "calibration_sha256": "e"*64,
            "training_prompt_hash": record["distillation_prompt_hash"]})
    path = tmp_path/"response.result.json"
    path.write_text(json.dumps(record))
    results, skipped = collect_results(tmp_path, "schemas/result.schema.json")
    assert len(results) == 1 and not skipped
    for field, value in (("dmd_response_weight", .19), ("cumulative_training_updates", 36000)):
        changed = copy.deepcopy(record)
        changed[field] = value
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="response-alignment"):
            collect_results(tmp_path, "schemas/result.schema.json")


def test_legacy_response_export_pair_fields_require_recorded_protocol():
    from few_step_diffusion.config import load_config
    from few_step_diffusion.dmd_response import response_pair_metadata
    config = load_config("configs/dmd_perceptual.yaml")
    assert response_pair_metadata(config, {}) == {}
    config["name"] = "dmd_response_angular_30"
    config["train"].update(dmd_response_mode="angular", dmd_response_variant="angular_30",
        dmd_response_weight=.18, dmd_response_gradient_fraction=.3,
        dmd_pair_seeds=2, dmd_pair_count=5000)
    metadata = {"method": config["name"], "teacher_pair_count": 5000, "protocol": {
        "seeds_per_prompt": 2, "count": 5000, "unique_prompts": 2500,
        "grouping": "adjacent-images-share-caption"}}
    fields = response_pair_metadata(config, metadata)
    assert fields == {"dmd_pair_seeds": 2, "teacher_pair_unique_prompts": 2500}
    assert "dmd_pair_seeds" not in metadata
    assert response_pair_metadata(config, {**metadata, **fields}) == fields
    for changed in ({**metadata, "dmd_pair_seeds": 1}, {**metadata, "protocol": {}},
                    {**metadata, "teacher_pair_count": 1000},
                    {**metadata, "protocol": {**metadata["protocol"], "grouping": "unpaired"}}):
        with pytest.raises(ValueError, match="Response export"):
            response_pair_metadata(config, changed)


def inputs():
    generator = torch.Generator().manual_seed(193)
    return (torch.randn(4, 4, 8, 8, generator=generator),
            torch.randn(4, 4, 8, 8, generator=generator))


def test_angular_gradient_preserves_pair_center_and_is_tangent():
    student, teacher = inputs()
    student.requires_grad_()
    teacher.requires_grad_()
    loss, _ = response_loss(student, teacher)
    loss.backward()
    assert teacher.grad is None
    gradient = student.grad.reshape(2, 2, -1)
    displacement = student.detach().reshape(2, 2, -1).diff(dim=1)[:, 0]
    torch.testing.assert_close(gradient.sum(1), torch.zeros_like(gradient[:, 0]))
    radial_derivative = ((gradient[:, 1]-gradient[:, 0])*displacement).sum(1)
    torch.testing.assert_close(radial_derivative, torch.zeros_like(radial_derivative), atol=1e-7, rtol=0)
    assert gradient.norm() > 0


def test_angular_loss_ignores_response_size_and_pair_center():
    student, teacher = inputs()
    expected, _ = response_loss(student, teacher)
    for scale in (.2, 2., 5.):
        actual, _ = response_loss(student*scale+1.7, teacher*3-2)
        torch.testing.assert_close(actual, expected)
    teacher_center = teacher.reshape(2, 2, 4, 8, 8).mean(1, keepdim=True)
    teacher_response = teacher.reshape(2, 2, 4, 8, 8)-teacher_center
    aligned = (teacher_center+3*teacher_response).reshape_as(teacher)
    angular, _ = response_loss(aligned, teacher)
    magnitude, _ = response_loss(aligned, teacher, "magnitude")
    torch.testing.assert_close(angular, torch.tensor(0.), atol=1e-7, rtol=0)
    assert magnitude > 3.9


def test_full_response_decomposes_into_magnitude_and_angle():
    student, teacher = inputs()
    for start in (0, 2):
        s, t = student[start:start+2], teacher[start:start+2]
        angular, info = response_loss(s, t)
        magnitude, _ = response_loss(s, t, "magnitude")
        vector, _ = response_loss(s, t, "vector")
        torch.testing.assert_close(vector, magnitude+2*info["response_norm_ratio"]*angular)


def test_degenerate_and_invalid_pairs_are_handled():
    student, teacher = inputs()
    student.zero_().requires_grad_()
    loss, _ = response_loss(student, teacher)
    gradient, = torch.autograd.grad(loss, student)
    assert torch.isfinite(gradient).all()
    teacher.zero_()
    loss, stats = response_loss(student, teacher)
    assert loss == 0 and stats["response_valid_pairs"] == 0
    with pytest.raises(ValueError):
        response_loss(student[:3], teacher[:3])
    with pytest.raises(FloatingPointError):
        response_loss(student.detach()+float("nan"), teacher)
