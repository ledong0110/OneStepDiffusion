import copy
import json
from types import SimpleNamespace

import pytest
from safetensors.torch import load_file
import torch

from few_step_diffusion import dmd_diversity, sd_training, training
from few_step_diffusion.config import load_config, validate
from few_step_diffusion.continuation import continuation_config
from few_step_diffusion.runtime import sha256
from few_step_diffusion.sd_data import caption_records
from few_step_diffusion.teacher_pairs import TeacherPairs
from test_continuation import annotations
from test_sd_training import TinyBundle, tiny_unet


class TinyDistance(torch.nn.Module):
    def forward(self, x, y):
        return (x-y).square().mean((1, 2, 3), keepdim=True)


def test_perceptual_and_diversity_gradients_reach_both_samples_with_frozen_targets():
    config = load_config("configs/dmd_perceptual_diversity.yaml")
    bundle = SimpleNamespace(decode=lambda x: x[:, :3].tanh())
    generated = torch.stack([torch.full((4, 8, 8), .2), torch.full((4, 8, 8), .3)]).requires_grad_()
    target = torch.stack([torch.full((4, 8, 8), -.6), torch.full((4, 8, 8), .6)]).requires_grad_()
    metric = TinyDistance().eval().requires_grad_(False)
    reg, diversity, parts = dmd_diversity.regression_and_diversity(bundle, metric, generated, target, config)
    diversity.backward(retain_graph=True)
    assert generated.grad[0, :3].sum() > 0 and generated.grad[1, :3].sum() < 0
    assert target.grad is None
    generated.grad = None
    reg.backward()
    assert generated.grad[0].abs().sum() > 0 and generated.grad[1].abs().sum() > 0
    assert target.grad is None and parts["teacher_pair_lpips_vgg"] > parts["student_pair_lpips_vgg"]
    same = target.detach().clone().requires_grad_()
    reg, diversity, _ = dmd_diversity.regression_and_diversity(bundle, metric, same, target, config)
    assert reg.item() == diversity.item() == 0


def test_grouped_teacher_cache_pairs_same_prompt_and_verifies_protocol(tmp_path, monkeypatch):
    config = load_config("configs/dmd_perceptual_diversity.yaml")
    config["train"].update(dmd_pair_count=4, initial_steps=4)
    config["data"]["sampling"] = "shuffled_epochs"
    rows = [{"image_id": i, "caption": f"prompt {i}"} for i in (10, 20, 30)]
    from few_step_diffusion import teacher_pairs
    calls = []

    def fake_sample(bundle, model, grid, prompts, seed, guidance, noise):
        calls.extend(prompts)
        return noise+.2

    monkeypatch.setattr(teacher_pairs, "generate_sd_latents", fake_sample)
    bundle = SimpleNamespace(unet=None, device=torch.device("cpu"))
    pairs = TeacherPairs(config, bundle, torch.arange(4), tmp_path, rows)
    assert calls == ["prompt 10", "prompt 10", "prompt 20", "prompt 20"]
    assert pairs.metadata["teacher_pair_count"] == 4
    assert pairs.metadata["teacher_pair_nfe"] == 32
    seen = []
    for microstep in range(2):
        noise, _, prompts = pairs.batch(microstep, "cpu")
        assert prompts[0] == prompts[1]
        assert not torch.equal(noise[0], noise[1])
        seen.append(prompts[0])
    assert set(seen) == {"prompt 10", "prompt 20"}
    restored = TeacherPairs(config, bundle, torch.arange(4), tmp_path, rows)
    assert calls == ["prompt 10", "prompt 10", "prompt 20", "prompt 20"]
    assert restored.batch(1, "cpu")[2] == pairs.batch(1, "cpu")[2]
    altered = copy.deepcopy(config)
    altered["model"]["guidance_scale"] = 4.5
    with pytest.raises(ValueError, match="protocol"):
        TeacherPairs(altered, bundle, torch.arange(4), tmp_path, rows)


def test_interrupted_pair_preprocessing_reuses_verified_completed_chunks(tmp_path, monkeypatch):
    from few_step_diffusion import teacher_pairs
    config = load_config("configs/dmd_perceptual_diversity.yaml")
    config["train"]["dmd_pair_count"] = 4
    config["eval"]["batch_size"] = 2
    rows = [{"image_id": i, "caption": f"prompt {i}"} for i in (10, 20)]
    calls = []
    should_fail = True

    def sample(bundle, model, grid, prompts, seed, guidance, noise):
        calls.append(prompts)
        if len(calls) == 2 and should_fail:
            raise InterruptedError("Interrupted preprocessing")
        return noise+.2

    monkeypatch.setattr(teacher_pairs, "generate_sd_latents", sample)
    bundle = SimpleNamespace(unet=None, device=torch.device("cpu"))
    with pytest.raises(InterruptedError):
        TeacherPairs(config, bundle, torch.arange(4), tmp_path, rows)
    assert (tmp_path/"teacher_pair_chunks/000000.safetensors").exists()
    should_fail = False
    pairs = TeacherPairs(config, bundle, torch.arange(4), tmp_path, rows)
    assert calls == [["prompt 10"]*2, ["prompt 20"]*2, ["prompt 20"]*2]
    assert torch.equal(pairs.tensors["latents"], pairs.tensors["noise"]+.2)


def test_diversity_evaluation_uses_shared_seeds_and_held_out_prompts(tmp_path, monkeypatch):
    from few_step_diffusion import diversity_evaluation as evaluator
    annotations(tmp_path/"data")
    config = load_config("configs/dmd.yaml")
    config.update(device="cpu", precision="fp32", progress_bar=False)
    config["data"].update(root=str(tmp_path/"data"), train_limit=2, train_offset=2, download=False)
    config["train"]["initial_steps"] = 4
    calls = []

    def generate(bundle, model, grid, prompts, seed, guidance, noise):
        calls.append((prompts, noise.clone()))
        return noise[:, :3].tanh()

    monkeypatch.setattr(evaluator, "SDBundle", TinyBundle)
    monkeypatch.setattr(evaluator, "generate_sd", generate)
    monkeypatch.setattr(evaluator, "load_lpips", lambda net, device: TinyDistance())
    first = evaluator.evaluate_diversity(config, "teacher", tmp_path/"first", prompt_count=2)
    second = evaluator.evaluate_diversity(config, "teacher", tmp_path/"second", prompt_count=2)
    assert first["diversity_lpips_alex_mean"] == second["diversity_lpips_alex_mean"]
    assert first["diversity_generation_nfe"] == 2*4*4*2
    assert first["diversity_samples_per_prompt"] == 4
    assert len(first["diversity_pairwise_scores_per_prompt"]) == 2
    for i in range(4):
        assert calls[i][0] == calls[i+4][0]
        torch.testing.assert_close(calls[i][1], calls[i+4][1], rtol=0, atol=0)
    assert all("20" in prompts[0] or "22" in prompts[0] for prompts, _ in calls)
    assert not torch.equal(calls[0][1][0], calls[0][1][1])


@pytest.mark.parametrize("variant", list(dmd_diversity.VARIANTS))
def test_direct_dmd_variant_forks_parent_and_resumes_exactly(tmp_path, monkeypatch, variant):
    data = tmp_path/"data"
    annotations(data)
    config = load_config("configs/dmd.yaml")
    config.update(device="cpu", precision="fp32", progress_bar=False, output_dir=str(tmp_path/"parent"))
    config["model"].update(lora_rank=2, gradient_checkpointing=False)
    config["data"].update(root=str(data), train_limit=2, download=False)
    config["train"].update(initial_steps=4, updates_per_stage=2, grad_accumulation=2,
        save_every=1, log_every=1, dmd_pair_count=2, warmup_steps=0)
    config["wandb"]["mode"] = "disabled"
    monkeypatch.setattr(sd_training, "SDBundle", TinyBundle)
    monkeypatch.setattr(sd_training, "new_unet", tiny_unet)
    monkeypatch.setattr(sd_training, "preview", lambda *args: None)
    monkeypatch.setattr(dmd_diversity, "load_lpips", lambda *args: TinyDistance())
    training.train(config)
    parent_sha = sha256(tmp_path/"parent/last.pt")
    cache = tmp_path/"shared_pairs"
    fresh = continuation_config(config, tmp_path/"parent", tmp_path/"continued", 2, 2, 2)
    fresh = dmd_diversity.apply_variant(fresh, variant, cache)
    fresh["continuation"]["parent_checkpoint_sha256"] = parent_sha
    validate(fresh)
    training.train(fresh)
    metadata = json.loads((tmp_path/"continued/stage_1/metadata.json").read_text())
    assert metadata["global_step"] == metadata["auxiliary_updates"] == 2
    assert metadata["effective_batch_size"] == metadata["paired_regression_batch_size"] == 2
    assert metadata["cumulative_training_updates"] == 4 and metadata["dmd_pair_seeds"] == 2
    assert metadata["teacher_pair_unique_prompts"] == 1
    assert metadata["continuation"]["parent_checkpoint_sha256"] == parent_sha
    assert not (tmp_path/"continued/teacher_pairs.safetensors").exists()
    resumed = copy.deepcopy(fresh)
    resumed["output_dir"] = str(tmp_path/"resumed")
    save = sd_training.save_state

    def interrupt(*args, **kwargs):
        save(*args, **kwargs)
        if args[7] == 1:
            raise InterruptedError("Interrupt new DMD objective")

    monkeypatch.setattr(sd_training, "save_state", interrupt)
    with pytest.raises(InterruptedError):
        training.train(resumed)
    monkeypatch.setattr(sd_training, "save_state", save)
    training.train(resumed, resume=True)
    reference = load_file(str(tmp_path/"continued/stage_1/adapter.safetensors"))
    actual = load_file(str(tmp_path/"resumed/stage_1/adapter.safetensors"))
    for key in reference:
        torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
    assert sha256(tmp_path/"parent/last.pt") == parent_sha
    assert caption_records(fresh)[0]["image_id"] == 5
