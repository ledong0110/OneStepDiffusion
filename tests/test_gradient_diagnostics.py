import pytest
import torch

from few_step_diffusion.gradient_diagnostics import alignment_statistics


def test_block_conflict_can_be_hidden_by_positive_global_dot_product():
    names = ["one.to_q.lora_A.default.weight", "one.to_q.lora_B.default.weight",
             "two.to_q.lora_A.default.weight"]
    score = [torch.tensor([1.]), torch.tensor([1.]), torch.tensor([2.])]
    perceptual = [torch.tensor([-2.]), torch.tensor([-2.]), torch.tensor([3.])]
    originals = [p.clone() for p in [*score, *perceptual]]
    row = alignment_statistics(names, score, perceptual, .25)
    assert row["global_cosine"] > 0 and not row["global_conflict"]
    assert len(row["blocks"]) == 2 and row["conflicting_block_fraction"] == .5
    assert row["global_removed_norm"] == 0
    assert row["block_removed_norm"] == pytest.approx(.25*(8**.5))
    for actual, original in zip([*score, *perceptual], originals, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_zero_and_nonfinite_diagnostic_gradients():
    name = ["one.to_q.lora_A.default.weight"]
    row = alignment_statistics(name, [torch.zeros(2)], [torch.ones(2)], .25)
    assert row["global_relative_correction"] == row["block_relative_correction"] == 0
    with pytest.raises(FloatingPointError):
        alignment_statistics(name, [torch.tensor([float("nan")])], [torch.ones(1)], .25)
