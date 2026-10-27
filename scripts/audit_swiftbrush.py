#!/usr/bin/env python3
"""Compare the port against expressions extracted from pinned official source."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", action="store_true", help="Use the verified cached reference")
    p.add_argument("--output", default="artifacts/audit/swiftbrush/parity.json")
    args = p.parse_args()
    import torch
    import torch.nn.functional as F
    from few_step_diffusion.sd_swiftbrush import (SOURCE_COMMIT, SOURCE_URL, source_original,
                                                 source_scorer_loss)
    from few_step_diffusion.vsd import vsd_surrogate
    from few_step_diffusion.runtime import atomic_json
    path = Path("artifacts/audit/swiftbrush")/SOURCE_COMMIT/"train_swiftbrush.py"
    if not args.offline:
        url = f"https://raw.githubusercontent.com/VinAIResearch/SwiftBrush/{SOURCE_COMMIT}/train_swiftbrush.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=30).read())
    expected = "008f8703794ba91f3282b43c22d96a0722de2ce6ee0590711f8bbec8cfdc9343"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Official reference source hash mismatch")
    tree = ast.parse(path.read_text())
    # Execute only the pinned pure predictor and loss assignments; never its
    # imports, training entry point, downloads, or arbitrary repository code.
    predictor = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "predict_original")
    env = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[predictor], type_ignores=[]), str(path), "exec"), env)
    torch.manual_seed(412)
    z, prediction = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    reference_original = env["predict_original"](lambda *a: SimpleNamespace(sample=prediction),
        SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)), z, None)
    torch.testing.assert_close(source_original(z, prediction), reference_original, rtol=0, atol=0)
    assignments = sorted((node for node in ast.walk(tree) if isinstance(node, ast.Assign)),
                         key=lambda node: node.lineno)

    def expression_block(first, last):
        def defines(node, name):
            return any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        start = next(node.lineno for node in assignments if defines(node, first))
        stop = next(node.lineno for node in assignments if defines(node, last))
        return ast.Module(body=[node for node in assignments if start <= node.lineno <= stop], type_ignores=[])

    images = reference_original.clone().requires_grad_()
    real, fake = torch.randn_like(images), torch.randn_like(images)
    sigma = torch.tensor([.2, .8])[:, None, None, None]
    env.update(sigma_t=sigma, teacher_pred=real, lora_pred=fake, pred_original_samples=images)
    exec(compile(expression_block("score_gradient", "loss_vsd"), str(path), "exec"), env)
    reference_gradient = torch.autograd.grad(env["loss_vsd"], images)[0]
    local_images = images.detach().clone().requires_grad_()
    local_loss = vsd_surrogate(local_images, real, fake, sigma)
    local_gradient = torch.autograd.grad(local_loss, local_images)[0]
    torch.testing.assert_close(local_loss, env["loss_vsd"], rtol=0, atol=0)
    torch.testing.assert_close(local_gradient, reference_gradient, rtol=0, atol=0)
    alpha = torch.tensor([.3, .9])[:, None, None, None]
    noise = torch.randn_like(images)
    scorer = torch.randn_like(images, requires_grad=True)
    env.update(alpha_t=alpha, lora_pred=scorer, noise=noise)
    # alpha_t's upstream assignment reads the scheduler. Supply it directly and
    # extract the subsequent prediction/target/loss assignments verbatim.
    last = next(node for node in assignments if any(isinstance(t, ast.Name) and t.id == "loss_lora"
                                                   for t in node.targets))
    block = ast.Module(body=[node for node in assignments if last.lineno-4 <= node.lineno <= last.lineno],
                       type_ignores=[])
    exec(compile(block, str(path), "exec"), env)
    reference_scorer_gradient = torch.autograd.grad(env["loss_lora"], scorer)[0]
    local_scorer = scorer.detach().clone().requires_grad_()
    local_scorer_loss = source_scorer_loss(local_scorer, noise, alpha)
    local_scorer_gradient = torch.autograd.grad(local_scorer_loss, local_scorer)[0]
    torch.testing.assert_close(local_scorer_loss, env["loss_lora"], rtol=0, atol=0)
    torch.testing.assert_close(local_scorer_gradient, reference_scorer_gradient, rtol=0, atol=0)
    result = {"status": "passed", "source_commit": SOURCE_COMMIT, "source_url": SOURCE_URL,
              "source_sha256": expected, "precision": "fp32", "finite_inputs": True,
              "predict_original_max_error": float((source_original(z, prediction)-reference_original).abs().max()),
              "vsd_gradient_max_error": float((local_gradient-reference_gradient).abs().max()),
              "scorer_gradient_max_error": float((local_scorer_gradient-reference_scorer_gradient).abs().max()),
              "note": "Pure-expression parity only; not end-to-end equivalence or generation quality. "
                      "Our trainer raises on nonfinite scores; upstream applies nan_to_num."}
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
