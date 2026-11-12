from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import accelerate_recent_baselines as speedup
from accelerate_recent_baselines import gate_step, resource_gate


def test_only_native_checkpoint_preview_is_a_safe_boundary():
    output = Path('/project/runs/method')
    prefix = ['python', '-m', 'few_step_diffusion.cli', 'evaluate', '--checkpoint']
    assert gate_step(prefix+[str(output/'checkpoints/step_004000')], output) == 4000
    assert gate_step(prefix+[str(output/'stage_1')], output) is None
    assert gate_step(prefix+[str(Path('/other/checkpoints/step_004000'))], output) is None
    assert gate_step(['python', 'train', '--checkpoint', str(output/'checkpoints/step_004000')], output) is None
    assert gate_step(prefix, output) is None


def test_relative_checkpoint_path_used_by_training_is_recognized(monkeypatch, tmp_path):
    monkeypatch.setattr(speedup, "ROOT", tmp_path)
    output = tmp_path/'runs/method'
    command = ['python', '-m', 'few_step_diffusion.cli', 'evaluate', '--checkpoint',
               'runs/method/checkpoints/step_006000']
    assert gate_step(command, output) == 6000


def test_rollout_preserves_live_run_when_replacement_cannot_fit():
    assert not resource_gate(1.5, 22, 6.7)
    assert not resource_gate(11.5, 22, 6.7)
    assert resource_gate(12, 22, 6.7)
    assert not resource_gate(12, 22, 2.7)


def test_placement_moves_only_the_requested_method():
    args = SimpleNamespace(train_gpu=3, helper_gpu=2, micro_factor=4, training_memory_gib=32,
        placements={"dmd2": {"train_gpu": 0, "micro_factor": 1, "training_memory_gib": 24,
                              "optimizations": ["fused_adamw", "vae_helper"]}})
    moved = speedup.execution_for_job(args, "dmd2")
    retained = speedup.execution_for_job(args, "f_distill")
    assert (moved.train_gpu, moved.helper_gpu, moved.micro_factor, moved.training_memory_gib) == (0, 2, 1, 24)
    assert moved.optimizations == ["fused_adamw", "vae_helper"]
    assert (retained.train_gpu, retained.micro_factor, retained.training_memory_gib) == (3, 4, 32)
    assert retained.optimizations == speedup.FLAGS
    assert args.train_gpu == 3


def test_destination_capacity_never_counts_memory_owned_on_source_gpu(monkeypatch):
    def output(command, **kwargs):
        if "--query-gpu=index,uuid,memory.free" in command:
            return "0, destination, 18432\n2, helper, 7168\n3, source, 9216\n"
        return "123, source, 22528\n"
    monkeypatch.setattr(speedup.subprocess, "check_output", output)
    args = SimpleNamespace(train_gpu=0, helper_gpu=2, training_memory_gib=24,
                           optimizations=["fused_adamw", "vae_helper"])
    budget = speedup.capacity({"pid": 123}, args)
    assert budget["owned_training_gpu_gib"] == 0
    assert budget["free_after_exit_gib"] == 18
    assert budget["required_after_exit_gib"] == 26
    assert budget["ready"] is False


@pytest.mark.parametrize("preferred_ready,source_ready,allow,expected", [
    (True, True, True, 0), (False, True, True, 3),
    (False, True, False, 0), (False, False, True, 0),
])
def test_source_fallback_keeps_the_preferred_destination_request(monkeypatch, preferred_ready, source_ready, allow, expected):
    preferred = SimpleNamespace(train_gpu=0, helper_gpu=2, micro_factor=1,
                                training_memory_gib=24, optimizations=["fused_adamw", "no_checkpoint"])
    monkeypatch.setattr(speedup, "capacity", lambda record, setting:
        {"ready": preferred_ready if setting.train_gpu == 0 else source_ready})
    selected, budget = speedup.choose_execution({"pid": 1}, preferred, 3, allow)
    assert selected.train_gpu == expected
    assert selected.optimizations == preferred.optimizations
    assert preferred.train_gpu == 0
    assert budget["ready"] == (preferred_ready if expected == 0 else source_ready)


@pytest.mark.parametrize("override", [
    {"train_gpu": 2}, {"micro_factor": 3}, {"training_memory_gib": 16},
    {"optimizations": ["unvalidated_kernel"]}, {"unknown_option": True},
])
def test_placement_rejects_unsafe_or_unknown_settings(override):
    args = SimpleNamespace(train_gpu=3, helper_gpu=2, micro_factor=4, training_memory_gib=32,
                           placements={"dmd2": override})
    with pytest.raises(ValueError):
        speedup.execution_for_job(args, "dmd2")
