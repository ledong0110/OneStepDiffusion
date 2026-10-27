import json
from types import SimpleNamespace

import pytest

from few_step_diffusion.config import config_fingerprint, load_config
from few_step_diffusion import tracking


def test_auto_mode_uses_credentials_without_prompting(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("NETRC", str(tmp_path / "missing-netrc"))
    assert tracking.resolve_mode({"wandb": {"mode": "auto"}}) == "offline"
    monkeypatch.setenv("WANDB_API_KEY", "fake-key-used-only-by-this-test")
    assert tracking.resolve_mode({"wandb": {"mode": "auto"}}) == "online"
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert tracking.resolve_mode({"wandb": {"mode": "online"}}) == "disabled"


def test_logging_settings_do_not_invalidate_checkpoint_fingerprint():
    config = load_config("configs/smoke.yaml")
    original = config_fingerprint(config)
    config["wandb"].update(mode="offline", project="another-project")
    config["progress_bar"] = False
    assert config_fingerprint(config) == original
    config["train"]["learning_rate"] *= 2
    assert config_fingerprint(config) != original


def test_online_resume_and_failed_run_cleanup(tmp_path, monkeypatch):
    import sys

    config = load_config("configs/smoke.yaml")
    config["wandb"]["mode"] = "online"
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setattr(tracking, "has_credentials", lambda: True)
    calls, finished = [], []

    def init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id=kwargs.get("id") or "unit-test-id", project="test",
                               dir=str(tmp_path), summary={}, define_metric=lambda *a, **k: None,
                               log=lambda values: None,
                               finish=lambda **kwargs: finished.append(kwargs["exit_code"]))

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **kw: kw))
    with tracking.TrainingTracker(config, {}, tmp_path):
        pass
    with pytest.raises(InterruptedError):
        with tracking.TrainingTracker(config, {}, tmp_path, resume=True):
            raise InterruptedError("interrupted test")
    assert calls[1]["id"] == "unit-test-id" and calls[1]["resume"] == "allow"
    assert finished == [0, 1]
    assert json.loads((tmp_path / "wandb_run.json").read_text())["mode"] == "online"
