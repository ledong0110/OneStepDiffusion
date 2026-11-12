import torch


def test_future_checkpoints_only_and_repeated_atomic_save(monkeypatch, tmp_path):
    from few_step_diffusion import checkpoint_storage as storage
    project, ssd = tmp_path/"project", tmp_path/"ssd"
    monkeypatch.setattr(storage, "ROOT", project)
    monkeypatch.setenv("FSD_CHECKPOINT_ROOT", str(ssd))
    output = project/"runs/new"
    storage.prepare_output_storage(output)
    assert not (output/"config.yaml").is_symlink()
    for name in ("last.pt", "checkpoints", "stage_1"):
        assert (output/name).is_symlink()
    for step in (1, 2):
        path = storage.save_destination(output/"last.pt")
        temp = path.with_suffix(".tmp")
        torch.save({"step": step}, temp)
        temp.replace(path)
        assert (output/"last.pt").is_symlink()
        assert torch.load(output/"last.pt", weights_only=True)["step"] == step
    assert (ssd/"runs/new/last.pt").is_file()


def test_existing_live_run_is_never_relocated(monkeypatch, tmp_path):
    from few_step_diffusion import checkpoint_storage as storage
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setenv("FSD_CHECKPOINT_ROOT", str(tmp_path/"ssd"))
    output = tmp_path/"runs/live"
    output.mkdir(parents=True)
    (output/"initialization.json").write_text("{}")
    (output/"last.pt").write_bytes(b"existing")
    storage.prepare_output_storage(output)
    assert (output/"last.pt").read_bytes() == b"existing"
    assert not any(p.is_symlink() for p in output.iterdir())
    assert not (output/"checkpoints").exists()


def test_real_save_state_keeps_external_link(monkeypatch, tmp_path):
    import few_step_diffusion.sd_training as training
    from few_step_diffusion import checkpoint_storage as storage
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setenv("FSD_CHECKPOINT_ROOT", str(tmp_path/"ssd"))
    output = tmp_path/"runs/new"
    storage.prepare_output_storage(output)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    monkeypatch.setattr(training, "adapter_state", lambda m: m.state_dict())
    for step in (1, 2):
        training.save_state(output/"last.pt", {}, model, model.state_dict(), optimizer,
                            0, step, step, 1.)
        assert (output/"last.pt").is_symlink()
        assert torch.load(output/"last.pt", weights_only=False)["global_step"] == step
    assert not (output/"last.tmp").exists()
