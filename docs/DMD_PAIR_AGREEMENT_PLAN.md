# Independent-pair spatial agreement: follow-up experiment

The first spatial-agreement pilot reached FID 30.0153, compared with 30.0894
for its paired mean, 30.0622 for its shifted-map control, and 29.9376 for the
frozen DMD+LPIPS reference. It was not selected as an improved final method.
This follow-up revises the statistic used for spatial weighting. DMD, LPIPS,
antithetic averaging and normalization remain attributed to prior work.

## Why change the estimator?

The opposite probes within one antithetic pair are dependent. They may
disagree even when their average has already cancelled the noise. For example,
let the desired direction be `(1,1)` and the two estimates be `(1,3)` and
`(1,-1)`. Their average is exactly correct, but the old agreement map assigns
different weights to its two entries and changes that direction.

Instead, draw two independent Gaussian noises and compute four normalized
DMD directions, `u+`, `u-`, `v+`, `v-`, using the same generated latent, caption
and timestep. Form independent antithetic means:

```text
p = (u+ + u-) / 2
q = (v+ + v-) / 2
m = (p + q) / 2
r_patch = clip(2 * inner(p_patch,q_patch) / (norm(p_patch)^2 + norm(q_patch)^2), 0, 1)
weighted = broadcast(1+r_patch) * m
update = weighted * norm(m) / norm(weighted)
```

The filter now measures disagreement remaining after antithetic averaging.
In the linear cancellation example, both pair means equal the desired signal;
their weights are uniform and normalization leaves it unchanged. This is an
elementary motivation, not a new mathematical theorem or a quality guarantee.
The update remains biased, and both pair means can share teacher/scorer error.

## Training-only diagnostic

A fixed 32-caption diagnostic with three timesteps and a separate 16-probe
reference examined strengths 0, 0.25, 0.5, 1, 2 and 4. Strength 1 gave the
largest average alignment gain among tested settings in both formulations;
larger strengths worsened it. The four-probe pair-agreement gain over its own
four-probe mean was +0.00391, with a caption-bootstrap interval
[0.00271, 0.00511]. Its shifted map gave -0.00766. These development measurements
do not prove that the revision improves images or is superior to the earlier
formulation. The same development captions were reused for sensitivity analysis.

No diagnostic optimizer updates were made; frozen model weights were verified
unchanged. The diagnostic source is
`artifacts/dmd_noise_agreement/probe_diagnostic/strength_and_independent_pairs.json`.

## Matched follow-up arms

| Arm | Direction used | Score probes |
|---|---|---:|
| Pair agreement | Spatial filtering of the mean of two independent antithetic means | 4 |
| Four-probe mean | Mean only | 4 |
| Shifted pairs | Same agreement weights, shifted by half the patch-map height/width | 4 |

All three fork the same completed 36,000-update DMD+LPIPS full state, preserving
generator, fake scorer, EMA, optimizers and RNG. Each gets 500 new updates,
36,500 cumulative, rank 32, effective batch 16, BF16, teacher CFG 7.5, generator
learning rate 1e-5 with 25-update warmup and cosine decay, and one auxiliary
scorer update per generator update. LPIPS-VGG stays at 0.25; no edge or distance
loss is added. Patch size 8 and filter strength 1 are fixed for this study.

The fixed DMD reference is not retrained. Four score probes add training work
relative to the earlier two-probe study and the single-probe reference. Only
the three new arms match one another in probe count. Inference remains one
conditional denoiser evaluation. Full 5,000-image FID/IS/CLIP, 128-caption × four
seed diversity, and the preset native sample comparison run automatically.

```bash
.venv/bin/python scripts/launch_dmd_conflict.py --study dmd_pair_agreement \
  --gpus 0 2 3 --max-parallel 3 --min-free-gib 16
```

## Completed outcome

All three runs completed 500 additional updates and the full evaluation pipeline.

| Method | FID | IS | CLIP | Within-caption LPIPS-Alex |
|---|---:|---:|---:|---:|
| Fixed DMD+LPIPS | 29.9376 | 34.1214 | 31.0986 | 0.708632 |
| Independent-pair agreement | 30.0342 | 34.2591 | 31.0895 | 0.709620 |
| Four-probe mean | 30.0627 | 34.2916 | 31.0876 | 0.709396 |
| Shifted pair agreement | 30.0396 | 34.1526 | 31.0901 | 0.709605 |

The candidate is 0.0966 worse in FID than the frozen reference. Its FID advantage
over the shifted-map control is only 0.0054. Slightly greater image variation
and IS do not establish an overall improvement; CLIP also decreases slightly.
The preset native images show small appearance changes without a clear visual
advantage. The training-only alignment diagnostic did not predict a FID gain.

This pilot is closed without promotion to a final method. The report has imported
all completed results. A selected final version still requires a supported new
candidate, final training, matched ablations and full evaluation. The earlier
checkpoints, diagnostics and negative results remain preserved.
