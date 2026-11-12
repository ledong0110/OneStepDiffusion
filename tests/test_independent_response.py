from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from few_step_diffusion.config import load_config, validate
from few_step_diffusion.independent_response import (
    PhasedCaptions, WEIGHT, bind_initialization, validate_independent,
)
from few_step_diffusion.runtime import sha256


@pytest.mark.parametrize("method", ["response_scratch", "dmd_lpips_scratch"])
def test_independent_config_and_no_inherited_state(method):
    cfg = load_config(f"configs/sd15_{method}_36k.yaml")
    assert cfg["train"]["updates_per_stage"] == 36000
    assert cfg["data"]["train_limit"] == 118287
    assert cfg["model"]["lora_rank"] == 32
    assert cfg["train"].get("dmd_response_weight", 0) == (WEIGHT if method == "response_scratch" else 0)
    for field in ("continuation", "dmd_refinement", "dmd_study"):
        altered = deepcopy(cfg)
        altered[field] = {"parent_run": "old_distilled_run"}
        with pytest.raises(ValueError, match="inherit"):
            validate_independent(altered)
    altered = deepcopy(cfg)
    altered["train"]["updates_per_stage"] = 500
    with pytest.raises(ValueError):
        validate(altered)


def test_exact_phase_boundary_and_local_indices(monkeypatch):
    import few_step_diffusion.independent_response as study
    records = [{"id": i} for i in range(4)]

    class Stream:
        def __init__(self, config, images):
            assert not images
            self.config = config
            self.records = records[:2] if config["data"]["train_offset"] == 0 else records[2:]

        def batch(self, index, device):
            return self.config["data"]["sampling"], index

    monkeypatch.setattr(study, "CaptionStream", Stream)
    monkeypatch.setattr(study, "caption_records", lambda c: records)
    cfg = load_config("configs/sd15_response_scratch_36k.yaml")
    data = PhasedCaptions(cfg)
    data.select(17999)
    assert data.batch(17999*8+7, None) == ("replacement", 17999*8+7)
    data.select(18000)
    assert data.batch(18000*8, None) == ("shuffled_epochs", 0)
    data.select(35999)
    assert data.batch(35999*8+7, None) == ("shuffled_epochs", 17999*8+7)


def test_initialization_binding_rejects_warm_start_and_foreign_resume(tmp_path):
    cfg = load_config("configs/sd15_response_scratch_36k.yaml")
    p = nn.Parameter(torch.tensor(1.))
    opt = torch.optim.AdamW([p])
    aux = torch.optim.AdamW([nn.Parameter(torch.tensor(2.))])
    weights = {"lora": torch.zeros(2, 2)}
    bind_initialization(cfg, tmp_path, weights, weights, opt, aux, None)
    record = json.loads((tmp_path/"initialization.json").read_text())
    assert record["initial_updates"] == 0
    assert record["response_enabled_from_update"] == 0
    assert record["inherited_distillation_checkpoint"] is None
    checkpoint = {"independent_initialization_sha256": sha256(tmp_path/"initialization.json")}
    bind_initialization(cfg, tmp_path, weights, weights, opt, aux, checkpoint)
    with pytest.raises(ValueError, match="fresh initialization"):
        bind_initialization(cfg, tmp_path, weights, weights, opt, aux, {})
    p.square().backward()
    opt.step()
    with pytest.raises(ValueError, match="empty optimizer"):
        bind_initialization(cfg, tmp_path, weights, weights, opt, aux, None)


def test_fresh_loop_resume_and_phase_switch(monkeypatch, tmp_path):
    """Actual DMD/response loop, fresh binding and RNG resume; tiny models only."""
    import few_step_diffusion.sd_training as train
    import few_step_diffusion.independent_response as study
    import few_step_diffusion.dmd_diversity as diversity

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 1)

        def forward(self, x, t, encoder_hidden_states):
            return SimpleNamespace(sample=self.conv(x))

    def base():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9)
            return Tiny().eval().requires_grad_(False)

    class Bundle:
        def __init__(self, config, device):
            self.unet, self.alphas = base(), torch.linspace(.99, .1, 1000)

        def embeddings(self, prompts):
            return torch.ones(len(prompts), 2, 2)

    class Data:
        def __init__(self, cfg):
            self.records, self.phase = [{"id": 1}, {"id": 2}], 0

        def batch(self, index, device):
            return None, ["first" if self.phase == 0 else "second"]*2

    class Pairs:
        def __init__(self, cfg, bundle, grid, data):
            self.data, self.rows = data, [{"caption": "a"}, {"caption": "a"}]
            self.metadata = {"teacher_pair_count": 5000, "weights_sha256": "b"*64, "protocol_hash": "c"*64,
                "protocol": {"count": 5000, "unique_prompts": 2500, "seeds_per_prompt": 2,
                             "grouping": "adjacent-images-share-caption"}}

        def select(self, step):
            self.data.phase = int(step >= 1)

        def batch(self, index, device):
            g = torch.Generator().manual_seed(100+index)
            z = torch.randn(2, 4, 64, 64, generator=g)
            return z, z*.3, ["a", "a"]

    monkeypatch.setattr(study, "PhasedCaptions", Data)
    monkeypatch.setattr(study, "PhasedPairs", Pairs)
    monkeypatch.setattr(study, "checkpoint_preview", lambda *a: None)
    monkeypatch.setattr(study, "independent_metadata", lambda *a: {})
    monkeypatch.setattr(train, "SDBundle", Bundle)
    monkeypatch.setattr(train, "new_unet", lambda *a: base())
    monkeypatch.setattr(train, "add_lora", lambda m, c, **kw: m.train().requires_grad_(True))
    monkeypatch.setattr(train, "adapter_state", lambda m: {k: v.detach().clone() for k, v in m.state_dict().items()})
    monkeypatch.setattr(train, "load_adapter", lambda m, s: m.load_state_dict(s))
    monkeypatch.setattr(train, "update_adapter_ema", lambda ema, m, decay:
                        [ema[k].lerp_(v.detach(), 1-decay) for k, v in m.state_dict().items()])
    monkeypatch.setattr(train, "preview", lambda *a: None)
    monkeypatch.setattr(train, "export_adapter", lambda *a: None)
    monkeypatch.setattr(diversity, "load_lpips", lambda *a: None)
    monkeypatch.setattr(diversity, "lpips_identity", lambda *a: {})
    monkeypatch.setattr(diversity, "regression_and_diversity", lambda b, m, x, y, c:
                        ((x-y).square().mean(), x.new_zeros(()), {}))
    cfg = load_config("configs/sd15_response_scratch_36k.yaml")
    cfg["precision"], cfg["progress_bar"] = "fp32", False
    cfg["train"].update(updates_per_stage=3, save_every=1, preview_every=10, grad_accumulation=1, log_every=1)
    tracker = SimpleNamespace(log=lambda *a: None, summarize=lambda *a: None)
    full, split = tmp_path/"full", tmp_path/"split"
    for folder in (full, split):
        folder.mkdir()
        torch.manual_seed(21)
        train.train_sd(cfg, None, torch.device("cpu"), folder, folder/"last.pt", {"hardware": {}},
                       tracker, stop_after=1 if folder == split else None)
    saved = torch.load(split/"last.pt", weights_only=False)
    assert saved["global_step"] == 1 and saved["auxiliary_updates"] == 1
    assert not (split/"run.json").exists()
    train.train_sd(cfg, saved, torch.device("cpu"), split, split/"last.pt", {"hardware": {}}, tracker)
    a, b = [torch.load(p/"last.pt", weights_only=False) for p in (full, split)]
    for key in ("student", "ema", "fake"):
        for name in a[key]:
            torch.testing.assert_close(a[key][name], b[key][name], rtol=0, atol=0)
    assert a["global_step"] == b["global_step"] == 3
    for key in ("optimizer", "fake_optimizer"):
        for parameter, values in a[key]["state"].items():
            for name, tensor in values.items():
                torch.testing.assert_close(tensor, b[key]["state"][parameter][name], rtol=0, atol=0)
    rows = [json.loads(line) for line in (full/"train.jsonl").read_text().splitlines()]
    assert rows[0]["global_step"] == 1 and "dmd_response_loss" in rows[0]
    assert rows[0]["dmd_response_weight"] == WEIGHT
    assert json.loads((split/"run.json").read_text())["cumulative_training_updates"] == 3


def test_queue_phase_resource_requirements(monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    import launch_independent_response as queue
    value = {"gpus": [0, 1, 2], "phase_limits": {
        "prepare": {"gpu_gib": 9, "disk_gib": 2}, "train": {"gpu_gib": 16, "disk_gib": 6}}}
    monkeypatch.setattr(queue, "gpu_free", lambda: {0: 1, 1: 13, 2: 7, 3: 100})
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda p: SimpleNamespace(free=10*2**30))
    assert queue.available_gpu(value, "prepare")[0] == 1
    assert queue.available_gpu(value, "train")[0] is None


@pytest.mark.parametrize("method", ["response_scratch", "dmd_lpips_scratch"])
def test_completed_independent_result_import_and_warmstart_rejection(tmp_path, method):
    from test_active_report import result_fixture
    from few_step_diffusion.config import config_fingerprint
    from few_step_diffusion.reporting import collect_results
    cfg = load_config(f"configs/sd15_{method}_36k.yaml")
    record = result_fixture("dmd_perceptual")
    for key in list(record):
        if key.startswith(("continuation", "parent_", "dmd_study", "refinement")):
            record.pop(key)
    record.update(method=method, resolved_config=cfg, config_hash=config_fingerprint(cfg),
        training_updates=36000, cumulative_training_updates=36000, auxiliary_updates=36000,
        independent_training=cfg["independent_training"], independent_initialization_sha256="d"*64,
        initialization={"initial_updates": 0, "inherited_distillation_checkpoint": None,
                        "optimizer_state_entries": [0, 0], "config_hash": config_fingerprint(cfg)},
        dmd_regression_type="lpips_vgg", dmd_regression_weight=.25,
        teacher_pair_phases=[{}, {}], cumulative_teacher_pair_count=10000, cumulative_teacher_pair_nfe=1280000,
        training_data_phases=[{"count": 50000, "updates": 18000, "sampling": "replacement"},
                              {"count": 68287, "updates": 18000, "sampling": "shuffled_epochs"}])
    if method == "response_scratch":
        record.update({k:v for k,v in cfg["train"].items() if k.startswith("dmd_response_")})
    path = tmp_path/"independent.result.json"
    path.write_text(json.dumps(record))
    assert len(collect_results(tmp_path, "schemas/result.schema.json")[0]) == 1
    record["initialization"]["inherited_distillation_checkpoint"] = "old_dmd.pt"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="independent"):
        collect_results(tmp_path, "schemas/result.schema.json")
