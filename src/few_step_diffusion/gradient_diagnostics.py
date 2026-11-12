"""Read-only alignment diagnostics for accumulated LoRA objective gradients."""

from collections import defaultdict
import math

import torch


def alignment_statistics(names, score, perceptual, weight):
    """Group A/B adapter matrices by their linear module; do not modify gradients."""
    groups = defaultdict(lambda: [0., 0., 0.])
    for name, a, b in zip(names, score, perceptual, strict=True):
        if ".lora_" not in name:
            raise ValueError("Diagnostics expect named LoRA parameters")
        values = torch.stack([(a*b).sum(), a.square().sum(), b.square().sum()]).detach()
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite diagnostic gradients")
        row = groups[name.split(".lora_", 1)[0]]
        for i, value in enumerate(values.cpu().tolist()):
            row[i] += value
    if not groups or not math.isfinite(weight) or weight < 0:
        raise ValueError("Diagnostics require gradients and a finite nonnegative coefficient")
    dot, score_sq, perceptual_sq = (math.fsum(v[i] for v in groups.values()) for i in range(3))
    denominator = math.sqrt(score_sq*perceptual_sq)
    cosine = max(-1., min(1., dot/denominator)) if denominator else 0.
    combined = math.sqrt(max(0., score_sq+weight**2*perceptual_sq+2*weight*dot))
    global_removed = weight*max(-dot, 0.)/math.sqrt(score_sq) if score_sq else 0.
    block_removed_sq = 0.
    blocks = []
    for name, (inner, a2, b2) in groups.items():
        denom = math.sqrt(a2*b2)
        conflict = inner < 0 and denom > 0
        if conflict:
            block_removed_sq += inner**2/a2
        blocks.append({"module": name, "cosine": max(-1., min(1., inner/denom)) if denom else 0.,
                       "conflict": conflict, "score_norm": math.sqrt(a2),
                       "perceptual_norm": math.sqrt(b2)})
    block_removed = weight*math.sqrt(block_removed_sq)
    return {"global_cosine": cosine, "global_conflict": dot < 0,
        "score_norm": math.sqrt(score_sq), "perceptual_norm": math.sqrt(perceptual_sq),
        "unmodified_combined_norm": combined, "global_removed_norm": global_removed,
        "block_removed_norm": block_removed,
        "global_relative_correction": global_removed/combined if combined else 0.,
        "block_relative_correction": block_removed/combined if combined else 0.,
        "conflicting_block_fraction": sum(b["conflict"] for b in blocks)/len(blocks),
        "blocks": blocks}
