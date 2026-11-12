# Candidate: spatial noise agreement for one-step distillation

This candidate changes the DMD latent update. **LPIPS regression is inherited
from DMD, not our contribution.** It is an experimental DMD extension, not an
established improvement or a claim of worldwide novelty. The earlier global
projection pilot reached FID 30.0592, worse than the fixed DMD+LPIPS reference
29.9376. Projection is not the primary proposal.

## Mechanism

For one generated latent, caption and timestep, evaluate the teacher and fake
scorer twice: once with Gaussian noise epsilon, once with its negative. Compute
the usual normalized DMD update for each probe, u and v. Their mean is m.

For each non-overlapping 8 by 8 latent patch, over all four channels:

```text
r = clip(2 * inner(u_patch, v_patch) / max(norm(u_patch)^2 + norm(v_patch)^2, tiny), 0, 1)
w = 1 + r
h = broadcast(w) * m
filtered_update = h * norm(m) / max(norm(h), tiny)  # per image
```

Inject the detached filtered update into the generator using the same surrogate
reduction as the existing DMD implementation. Keep LPIPS-VGG weight 0.25,
teacher CFG 7.5 and the existing fake-scorer training rule. No edge or pairwise
distance loss is used. Inference still needs one conditional denoiser call.

This tests whether spatial agreement helps select a useful update direction.
It does not estimate calibrated uncertainty. Two probes can agree on the same
error, and weighting introduces bias. Norm preservation holds in latent space,
not in LoRA parameter space after the generator Jacobian or optimizer.
Weights remain in [1,2]; zero updates remain zero. Patch size 8 is an exploratory
choice, not a value justified by a completed sensitivity study.

The weights impose a precise limit on the change in direction. For a nonzero
paired mean m, let p_i = m_i^2 / ||m||^2 and mu = sum_i p_i w_i. Then
cos(angle(m, filtered)) = mu / sqrt(sum_i p_i w_i^2). Since 1 <= w_i <= 2,
w_i^2 <= 3w_i - 2, giving cos^2(angle) >= mu^2 / (3mu - 2) >= 8/9.
Thus the latent rotation is at most acos(sqrt(8/9)), approximately 19.47 degrees.
The bound is attained by m=(sqrt(2),1) and weights (1,2), and applies to the
shifted-map control too. This elementary bound explains the restricted
perturbation strength, not optimality of the weights or better FID. It does
not bound parameter-gradient angles after the generator Jacobian or AdamW.

## Three matched experiments

| Variant | Score update | Purpose |
|---|---|---|
| Spatial agreement | Norm-preserving spatial filter of m | Candidate mechanism |
| Paired mean | m directly | Separate filtering from averaging two probes |
| Shifted agreement | Roll the patch map by half its height and width, then filter m | Test whether spatial association matters |

All three use two opposite-noise probes. They fork the same frozen completed
36k DMD+LPIPS state, preserving G, fake scorer, EMA, both optimizers and RNG.
They use the same continued training-caption stream, fixed cached teacher
anchors, rank 32, effective batch 16 and BF16. Each pilot receives 500 new
updates: **36,500 cumulative updates**, with generator LR 1e-5, 25-update warmup
and cosine decay. Scorer LR stays 1e-4, one scorer update per generator update.
The fixed baseline is not retrained. Consequently a gain over that reference
alone does not isolate the mechanism from additional training or probe compute.

```bash
.venv/bin/python scripts/launch_dmd_conflict.py --study dmd_noise_agreement \
  --gpus 2 3 0 1 --max-parallel 3 --min-free-gib 16
```

Training is detached, with tqdm and inherited W&B auto mode. Each process has a
15 GiB PyTorch allocation cap; the queue waits for at least 16 GiB free and
8 GiB free disk. The three folders are under `runs/sd15/dmd_noise_agreement/`.
The manifest and status preserve actual configuration, parent identity and
process identity. Resume the same command rather than starting duplicate jobs.

## Evidence and selection

The 500-update pilot is complete. Full 5,000-image quality, 128-caption/four-seed
diversity, and final checkpoint samples are available for all three arms:

| Method | Total updates | FID | IS | CLIP | Pairwise LPIPS |
|---|---:|---:|---:|---:|---:|
| Fixed DMD+LPIPS | 36,000 | 29.9376 | 34.1214 | 31.0986 | 0.708632 |
| Spatial agreement | 36,500 | 30.0153 | 34.0819 | 31.1006 | 0.708718 |
| Paired mean | 36,500 | 30.0894 | 34.0455 | 31.1032 | 0.708975 |
| Shifted agreement | 36,500 | 30.0622 | 34.1749 | 31.1009 | 0.708524 |

Spatial agreement reduces observed FID by 0.0741 versus paired mean and 0.0469
versus the shifted map, but is 0.0777 worse than the frozen reference. These
single-seed differences do not establish significance or a general improvement.
**Do not promote this checkpoint as an improved final method.** Its full results
and native samples remain in the report while further refinement is investigated.

The next read-only diagnostic examines strengths 0, 0.25, 0.5, 1, 2 and 4 on the
existing training-caption diagnostic set. It also tests agreement between two
independent antithetic-pair means, with a matched four-probe mean as its control.
The latter addresses a limitation of the original statistic: opposite probes
can disagree even when averaging already cancels their noise. Neither diagnostic
trains a model, proves improved FID, or changes the completed pilot's recipe.

Every completed arm automatically evaluates all 5,000 COCO validation captions
against all 5,000 real validation images, using the fixed reference's noise
stream and metric protocol. It then evaluates 128 held-out prompts with four
seeds each for within-caption LPIPS-Alex variation and renders the same three
prespecified qualitative prompts. The report imports native checkpoint samples
only when their identity matches the full evaluation. No fabricated scores or
imagegen examples enter empirical figures.

Inspect score-probe cosine, patch agreement and the relative change from the
paired mean during training; finite losses alone do not demonstrate image
quality. Full FID/IS/CLIP and diversity plus samples determine whether to extend
training. The spatial rule should show evidence against **both** equal-probe
ablations, not only the lower-budget frozen reference. Repeated validation
selection and one training seed limit any causal or significance claim.

The launcher refreshes `DMD_NOISE_AGREEMENT_RESULTS.md`, then rebuilds the Group 11
Assignment 2 PDF and source package after completed evaluation. Pilot launch
does not mean the selected final version has completed.

### Guarded main-figure selection

The Assignment 2 builder can replace the earlier distance-extension column
with an evaluated candidate using an explicit
`configs/assignment2_selected_method.json` selection. No selection is active
while these pilots run. The selection records policy
`assignment2-evaluated-main-candidate-v1`, study, method, a written reason,
the SHA-256 of the completed result JSON, the evaluated model identity, and
the SHA-256 of its archived checkpoint. All study arms must first have full
quality and diversity measurements. Native final images must match that
checkpoint and the existing main figure's prompts, noise and baseline images.

This only selects the main comparison column; it does not turn an exploratory
result into a demonstrated improvement. The caption states the selected
candidate's actual cumulative updates alongside the 36,000-update baselines.
Earlier distance-study figures retain their original model labels and images.
The existing selection remains unchanged when the configuration is absent.

## Attribution and boundaries

- Original [DMD](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html)
  supplies distribution matching, the auxiliary scorer and perceptual regression.
- Antithetic sampling and norm normalization are existing numerical ideas.
- [CARV](https://arxiv.org/html/2605.21489v1) evaluates diffusion-teacher estimator
  variance reduction, including re-noising, and reports no downstream FID gain
  in its single-step experiments despite lower gradient variance. Averaging
  two probes alone is therefore neither the claimed contribution nor proof
  of likely improvement.
- Our candidate tests the particular spatial cross-energy weighting and its
  norm-preserving application to a paired DMD update. The prior-art review
  does not establish that this combination is globally novel.

### Closer prior-art check

The following primary sources further narrow what this project can claim:

| Prior work | Existing mechanism | Difference in this candidate |
|---|---|---|
| [SFERD](https://arxiv.org/html/2311.03830v2) | Teacher attention guidance and a learned semantic gradient predictor for spatial fitting error | Agreement is computed from two DMD directions; no attention mask or new predictor |
| [SOLACE: intrinsic self-confidence rewards](https://arxiv.org/html/2603.00918v3) (formerly ARC) | Antithetic re-noising probes produce a scalar noise-reconstruction reward for RL | Direct patch weighting of the DMD update, without a reconstruction reward or RL objective |
| [GSD](https://arxiv.org/html/2406.16695v1) | Geometry-warped score gradients and a cross-view consistency loss for text-to-3D | Same-latent, opposite-noise probes; no camera geometry or cross-view loss |
| [sDMD](https://arxiv.org/html/2610.39889v1) | Discriminator-gated Sinkhorn correction of distribution matching | No transport term or discriminator; weights derive from local probe agreement |

Thus neither using two opposite noises, spatial guidance, nor comparing score
gradients is a defensible standalone novelty claim. The empirical question is
whether this specific stopped spatial filter helps beyond both equal-probe
controls. This limited search does not prove absence of an identical prior method.

### Early native check

The candidate and paired-mean arm have preserved full-state snapshots at 50
updates. The three preset dog, teddy-bear and lighthouse prompts retain
recognizable subjects and very similar compositions to the fixed reference.
No obvious collapse appears in these three images, and no visible advantage
is established. These early previews are diagnostics only; the paper's final
sample importer still requires a fully evaluated 500-update checkpoint.

Implementation: `src/few_step_diffusion/dmd_noise_agreement.py`; integration:
`src/few_step_diffusion/sd_training.py`. Tests cover the baseline surrogate
identity, stopped scorer gradients, norm preservation, shifted-map control,
numerical guards, metadata import and full-state checkpoint/resume behavior.

### Independent-probe direction diagnostic

A read-only diagnostic uses the frozen DMD training state (raw generator and
fake-scorer weights) and training captions, without optimizer updates. A first
four-caption check motivated a repeat with 32 additional captions and a separate
noise seed. Each caption shares one generated latent across timesteps 100, 500
and 900. At each point, eight additional independent antithetic pairs (16
probes) form a finite reference direction. Pairs are independent of each other
and of the candidate pair; the two probes within a pair are dependent. The
filter and patch size remain unchanged.

Over the 96 repeated cases, the candidate's mean cosine gain over the paired
mean is **+0.00349**; shifting the spatial map gives **-0.00541**. The respective
95% caption-bootstrap intervals are [0.00195, 0.00509] and
[-0.00670, -0.00419]. Caption resampling keeps all three timesteps together.
At t=900, the candidate's interval includes zero. This is modest evidence that
spatial association affects direction estimation in this setting; it is not
ground-truth gradient accuracy, training-seed uncertainty, or image-quality
improvement. The reference direction itself remains noisy.

`scripts/analyze_noise_agreement.py` records measurements and verifies unchanged
weights. `scripts/draw_probe_diagnostic.py` renders all individual cases and
caption-level intervals. The report package includes the raw measurements.
Results are under `artifacts/dmd_noise_agreement/probe_diagnostic/`; no diagnostic
replaces the full 5,000-image quality evaluation or the matched training ablations.
