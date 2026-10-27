import copy
import json

import pytest
from safetensors.torch import load_file
import torch

from few_step_diffusion import sd_swiftbrush, sd_training, training
from few_step_diffusion.config import load_config
from few_step_diffusion.continuation import continuation_config, initialize_continuation
from few_step_diffusion.runtime import sha256
from few_step_diffusion.sd_data import CaptionStream, caption_records
from test_sd_training import TinyBundle, TinyStream, tiny_unet


def annotations(root):
    (root/"annotations").mkdir(parents=True)
    for split, ids in (("train2017", [1, 3, 5, 7]), ("val2017", [20, 22])):
        contents = {"images": [{"id": i, "file_name": f"{i}.jpg"} for i in ids],
            "annotations": [{"id": i, "image_id": i, "caption": f"a quiet city street {i}"}
                            for i in ids]}
        (root/"annotations"/f"captions_{split}.json").write_text(json.dumps(contents))


def test_fresh_subset_is_disjoint_validation_ignores_offset_and_epochs_resume(tmp_path):
    annotations(tmp_path)
    config = load_config("configs/sd15.yaml")
    config["data"].update(root=str(tmp_path), download=False, train_limit=2)
    old = caption_records(config)
    fresh_config = copy.deepcopy(config)
    fresh_config["data"].update(train_offset=2, sampling="shuffled_epochs", batch_size=3)
    fresh = caption_records(fresh_config)
    assert [r["image_id"] for r in old] == [1, 3]
    assert [r["image_id"] for r in fresh] == [5, 7]
    assert [r["image_id"] for r in caption_records(fresh_config, "validation", 2)] == [20, 22]
    assert (tmp_path/"train2017_2.manifest.json").exists()
    assert (tmp_path/"train2017_2_offset2.manifest.json").exists()
    stream = CaptionStream(fresh_config, images=False)
    # Odd-sized batches straddle epoch boundaries, with every image visited per epoch.
    sequence = [i for micro in range(4) for i in stream.indices(micro).tolist()]
    assert all(sorted(sequence[start:start+2]) == [0, 1] for start in range(0, 12, 2))
    resumed = CaptionStream(fresh_config, images=False)
    assert stream.batch(3, "cpu")[1] == resumed.batch(3, "cpu")[1]


@pytest.mark.parametrize("mode", ["distill", "consistency", "dmd", "instaflow", "swiftbrush"])
def test_completed_one_step_continuation_preserves_state_parent_and_exact_resume(
        tmp_path, monkeypatch, mode):
    data_root = tmp_path/"data"
    annotations(data_root)
    source = mode == "swiftbrush"
    config = load_config("configs/sd15_swiftbrush_lora.yaml" if source else "configs/sd15_proposed.yaml")
    config.update(device="cpu", precision="fp32", progress_bar=False, output_dir=str(tmp_path/"parent"))
    config["model"].update(lora_rank=2, lora_alpha=2, gradient_checkpointing=False)
    config["data"].update(root=str(data_root), download=False, train_limit=2)
    if source:
        config["train"]["fake_lora_rank"] = 2
    else:
        config["train"]["mode"] = mode
    config["train"].update(initial_steps=4, stages=[2, 1] if mode == "distill" else [1],
        updates_per_stage=2 if mode == "distill" else 4, grad_accumulation=1, save_every=1,
        log_every=1, warmup_steps=0, dmd_pair_count=2, dmd_regression_weight=.25,
        instaflow_pair_count=2, instaflow_prediction_type="epsilon-converted-velocity")
    config["wandb"]["mode"] = "disabled"

    class FreshTinyStream(TinyStream):
        def __init__(self, c, images=True):
            super().__init__(c, images)
            self.records = caption_records(c)

    for module in (sd_training, sd_swiftbrush):
        monkeypatch.setattr(module, "SDBundle", TinyBundle)
        monkeypatch.setattr(module, "CaptionStream", FreshTinyStream)
        monkeypatch.setattr(module, "new_unet", tiny_unet)
    monkeypatch.setattr(sd_training, "preview", lambda *args: None)
    monkeypatch.setattr(sd_swiftbrush, "preview_source", lambda *args: None)
    training.train(config)
    parent = tmp_path/"parent"
    parent_sha = sha256(parent/"last.pt")
    original = torch.load(parent/"last.pt", weights_only=False)
    continued = continuation_config(config, parent, tmp_path/"continued", offset=2, count=2, updates=2)
    (tmp_path/"continued").mkdir()
    initial = initialize_continuation(continued, tmp_path/"continued")
    assert initial["global_step"] == initial["stage_step"] == 0
    assert initial["stage_index"] == (1 if mode == "distill" else 0)
    for key in ("student", "fake") if mode in ("dmd", "swiftbrush") else ("student",):
        for name in original[key]:
            torch.testing.assert_close(initial[key][name], original[key][name], rtol=0, atol=0)
    assert initial["optimizer"]["state"]
    if source:
        assert initial["ema"]["optimization_step"] == 4
    training.train(continued)
    metadata = json.loads((tmp_path/"continued/stage_1/metadata.json").read_text())
    assert metadata["global_step"] == 2
    assert metadata["cumulative_training_updates"] == 6
    assert metadata["continuation"]["training_image_overlap"] == 0
    assert metadata["continuation"]["combined_unique_training_images"] == 4
    if mode == "instaflow":
        assert metadata["reflow_updates"] == metadata["teacher_pair_nfe"] == 0
        assert metadata["flow_distillation_updates"] == 2
        assert not (tmp_path/"continued/teacher_pairs.safetensors").exists()
    records = [json.loads(row) for row in (tmp_path/"continued/train.jsonl").read_text().splitlines()]
    assert len(records) == 2 and all(r["stage_steps"] == 1 for r in records)
    resumed = continuation_config(config, parent, tmp_path/"resumed", offset=2, count=2, updates=2)
    module, name = (sd_swiftbrush, "save_source_state") if source else (sd_training, "save_state")
    save = getattr(module, name)

    def interrupt(*args, **kwargs):
        save(*args, **kwargs)
        if args[7] == 1:
            raise InterruptedError("Interruption after first continuation update")

    monkeypatch.setattr(module, name, interrupt)
    with pytest.raises(InterruptedError):
        training.train(resumed)
    monkeypatch.setattr(module, name, save)
    training.train(resumed, resume=True)
    expected = load_file(str(tmp_path/"continued/stage_1/adapter.safetensors"))
    actual = load_file(str(tmp_path/"resumed/stage_1/adapter.safetensors"))
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert sha256(parent/"last.pt") == parent_sha
    overlap = continuation_config(config, parent, tmp_path/"overlap", offset=0, count=2, updates=2)
    (tmp_path/"overlap").mkdir()
    with pytest.raises(ValueError, match="overlap"):
        initialize_continuation(overlap, tmp_path/"overlap")
