# Earlier candidate: conflict-aware perceptual optimization for DMD

The primary candidate is now [spatial noise agreement](DMD_NOISE_AGREEMENT_PLAN.md),
following the request for a distinct mechanism beyond inherited DMD+LPIPS.
All four projection-study arms have completed full quality and diversity
evaluation. None reduced measured FID below the fixed reference:

| Configuration | Full 5,000-image FID | Pairwise LPIPS-Alex |
|---|---:|---:|
| Fixed DMD+LPIPS, 36k | 29.9376 | 0.708632 |
| Global projection, 36.5k | 30.0592 | 0.708339 |
| Global norm control, 36.5k | 30.0591 | 0.708129 |
| Module projection, 36.5k | 30.1596 | 0.708113 |
| Module norm control, 36.5k | 29.9960 | 0.707939 |

Global projection and its norm control are effectively tied at the reported
FID precision. Module projection is worse than its norm control in this run.
These single-seed results provide no basis for extending projection training;
the experiments remain exploratory evidence, not a claimed novel method or
the selected final model. This does not prove a universal negative result.

Status: implemented and launched as a 500-update pilot on 4 November 2026.
The user cancelled additional DMD control training and chose to retain the
existing best 36k DMD+LPIPS result. Global and per-module projection each have a
norm-matched ablation, making four exploratory experiments. Quality and diversity
evaluations are complete. The diversity-weight sweep remains paused.
This is an experimental DMD extension, not an established improvement.

Implementation: `src/few_step_diffusion/dmd_conflict.py`. Launch/resume the
detached queue with:

```bash
.venv/bin/python scripts/launch_dmd_conflict.py --gpus 2 3 --max-parallel 2
.venv/bin/python scripts/launch_dmd_conflict.py --study dmd_block_conflict --gpus 0 1 2 3 --max-parallel 2 --min-free-gib 16
```

Each queue runs projection and the corresponding norm-matched ablation. Each starts
from the same complete 36k DMD+LPIPS state and
receives 500 updates: **36,500 total**, with a common generator LR of 1e-5,
25-update warmup and cosine decay. The scorer LR remains 1e-4. Rank 32, effective
batch 16, CFG 7.5, BF16, LPIPS coefficient 0.25 and the teacher-pair bank are
unchanged across arms. Both backward passes are computed in every arm.

The module variant groups each adapted linear layer's LoRA A and B tensors,
then applies the projection independently in each group after accumulation over
the complete effective batch. Its ablation scales each group's perceptual
gradient to the same norm without changing its direction. A read-only diagnostic
on eight training batches at the global proposal's 100-update checkpoint found
opposing gradients in 66.9% of groups. The correction relative to the combined
gradient was 1.53% globally and 4.58% with module grouping. These measurements
motivate testing the finer partition; they do not demonstrate better images.
GPUs 0 and 1 use a 15 GiB PyTorch allocation cap and require 16 GiB free at launch.
The queue now permits GPUs 0--3 for phase scheduling. After module-norm training
completed, its first evaluation failed the memory admission check on GPU 1;
the checkpoint was retained and evaluation was rescheduled. The launcher checks
memory again between phases and can move evaluation without retraining.

Checkpoints are saved every 50 updates; immutable full-state archives preserve
updates 250 and 500. Fixed-prompt previews appear at updates 250 and 500.
The queue evaluates all 5,000 validation captions/images and then
128 held-out prompts with four noise seeds for diversity. It refreshes
`DMD_CONFLICT_RESULTS.md` and `report/generated/dmd_conflict_results.tex` after
evaluation. The module queue uses `DMD_BLOCK_CONFLICT_RESULTS.md` and its own
run/results directories. Both queues refresh and compile the Assignment 2
report after evaluation. Pending results remain empty, never copied from the
36k parent. After full evaluation, each queue also renders the same dog,
teddy-bear and lighthouse prompts with the same initial noise as the frozen
reference. The report accepts these panels only when their model identity
matches the evaluated checkpoint. Its diversity column verifies the reference
prompt/seed protocol and all 128 per-prompt measurements.

Status: `runs/sd15/dmd_conflict/status.json`. Active training logs are in `project`
and `norm_match`; module status is in `runs/sd15/dmd_block_conflict/status.json`.
The cancelled control's logs remain in `sum` for history and
will not be used as a completed result. tqdm and W&B auto mode remain enabled;
without credentials, W&B records offline. The user has now authorized continued
development: assess quality, tune or replace an unsuccessful candidate, and
train a final selected version and its ablations. The existing DMD reference
must remain fixed. Do not turn a small validation improvement into a claim of
causal or statistically established superiority.

## Baseline and motivation

Use DMD distribution matching **with LPIPS regression**, not the older latent-MSE
control. The completed matched local control is
`runs/sd15/dmd_diversity_study/dmd_perceptual_seed42`; its measured FID is
29.937644 on 5,000 COCO validation images. Its total 36k-update history includes
the common 18k DMD initialization and the 18k LPIPS phase. It is a local LoRA
adaptation, not an exact reproduction of the published full-model experiment.

The implemented generator objective is a weighted sum of distribution matching
and LPIPS regression. `src/few_step_diffusion/sd_training.py` constructs the two
terms on separate generated and cached-anchor streams and backpropagates their
sum in the original path. The new path records full-batch gradient cosine,
component norms and whether projection is active at each logged update. Our hypothesis is that some
updates may make these objectives oppose each other in the small trainable LoRA
parameter space. The first logged update had positive cosine (approximately
0.05) in both arms. At update 10, cosine was -0.00996 in the projection arm and
-0.02941 in the standard arm; the projection branch activated. These are early
training diagnostics, not evaluation results. **They do not establish that
gradient conflict causes image errors or that removing it improves FID.**

## Proposed change

Let `g_D` be the gradient of the DMD surrogate and `g_P` the gradient of unweighted
LPIPS regression with respect to all trainable student adapters. Accumulate each
over the full effective batch before computing their dot product. Keep the DMD
gradient, and modify the perceptual gradient only when the two disagree:

```text
dot = inner(g_D, g_P)
if dot < 0 and squared_norm(g_D) is safely nonzero:
    g_P_projected = g_P - dot / squared_norm(g_D) * g_D
else:
    g_P_projected = g_P

combined_gradient = g_D + 0.25 * g_P_projected
```

This removes the opposing component of perceptual supervision while retaining
its remaining direction. It introduces no diversity loss, no new image model,
no teacher-pair construction and no extra inference step. The two separate
gradient buffers and gradient calculations add training overhead; it must be
measured rather than assumed negligible. Use a zero-gradient guard and explicit
nonfinite checks, not an uncontrolled division.

The projection has a local first-order interpretation in the raw gradient
geometry. It does **not** guarantee better images, lower validation FID, or
monotone loss reduction under AdamW, finite steps, clipping and a moving scorer.
Protecting a biased or stale DMD direction could make results worse.

## Prior art and honest scope

- Original [DMD](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html)
  supplies distribution matching and perceptual regression. Neither is our invention.
- [PCGrad](https://arxiv.org/abs/2001.06782) already projects conflicting task
  gradients. The proposed update is an asymmetric PCGrad-style application,
  not a new general-purpose optimization algorithm.
- [DMD2](https://tianweiy.github.io/dmd2/dmd2.pdf) identifies inadequate fake-score
  tracking and uses a two-time-scale remedy, among other changes. Increasing
  scorer updates alone must be credited to this existing line of work.
- [Flash-DMD](https://arxiv.org/html/2511.20549v1) already discusses conflict
  between distribution and adversarial gradients and separates objectives by
  timestep. [GDMD](https://arxiv.org/abs/2604.19009) addresses distillation/RL
  interactions using gradient-based reward guidance. Those are related but
  different objective pairs from the LPIPS-anchor conflict tested here.
- [Connected Self Forcing, Appendix D.6](https://arxiv.org/html/2610.12156v1)
  already tests asymmetric projection against a DMD gradient, using cross-chunk
  video feedback. Its ablation reports a quality tradeoff. This is a direct
  precedent for the projection form; our different LPIPS objective pair does
  not establish novelty or guarantee improvement.

The possible project contribution is a measured application/analysis of
conflict-aware perceptual optimization for one-step rank-32 DMD. The literature
check does not establish worldwide novelty. A stronger novelty claim would
require a broader prior-art review and a demonstrably distinct method.

## Diagnostics within the short pilot

The first optimizer update in each arm begins from the identical state, training
captions and RNG stream. Its component gradients supply a common initial
diagnostic; later gradients differ as training diverges. Examine the logs:

1. Measure full-batch cosine similarity of `g_D` and `g_P`, the fraction of
   negative dot products, and their norm ratio.
2. Inspect whether apparent conflict repeats across independent batches. Do
   not select a single convenient example. The separate eight-batch diagnostic
   measures per-module conflict, and module-variant training logs additionally
   record the fraction of conflicting LoRA modules.
3. Inspect aggregate scorer denoising error as context. Noise-range breakdowns
   are not currently logged; aggregate MSE alone is not a calibrated estimate
   of score error because irreducible noise varies with the timestep.

If conflict is negligible or unstable across batches, do not extend training
beyond this short pilot. A training-gradient diagnostic is not positive evidence
of image-quality improvement by itself. Logged conflict indicators are sampled
every ten updates, not an exact count over all optimizer updates.

## Pilot and cancelled matched control

All arms fork the identical completed LPIPS-control state, including generator,
fake scorer, EMA and optimizer states. They use the same captions, teacher bank,
seeds, rank, batch, guidance, precision and explicit learning-rate schedule.

| Arm | Update | Purpose |
|---|---|---|
| A (cancelled) | Standard `g_D + 0.25 g_P` | Originally planned equal-budget control |
| B | Conditional projection above | Test directional conflict handling |
| C | Rescale `g_P` to the projected norm but preserve its direction | Separate direction correction from simply weakening LPIPS |
| D | Apply B independently within each LoRA module | Test the effect of parameter grouping |
| E | Apply C independently within each LoRA module | Magnitude ablation for D |

For C, apply the norm rescaling only on the same negative-dot-product updates;
use unchanged `g_P` otherwise. Compute both component gradients in every arm so
the diagnostic/control overhead is matched. Measure actual GPU time as well as
generator updates. Keep the scorer update ratio fixed across arms.

Start with 500 additional updates per arm, checking at 250 and 500; choose a
larger budget only after measuring throughput and the actual remaining time.
These would be **36,500 total updates**, not 36,000. A shorter screening sample
set is only a pilot; use all 5,000 validation images for final comparable
FID/IS/CLIP, and include the existing within-prompt diversity evaluation.
Validation-based selection remains exploratory because this benchmark has
already been inspected during development.

Arm A's extra training is now cancelled. The retained DMD+LPIPS reference has
36,000 updates while the proposal experiments have 36,500. A gain over that
reference cannot isolate the new rule from the additional training budget.
All four proposal experiments share an equal budget. Consider FID together
with CLIP, diversity, and fixed-prompt images. One training seed cannot establish statistical
significance. If time permits, repeating the winning comparison with another
seed is more informative than adding several unmotivated loss terms.

## What can be claimed

Before quality measurements: an implemented mechanism with training diagnostics,
not an established improvement. Compare B with C and D with E to assess direction
correction at equal update budgets. Comparison with the fixed DMD reference has
an additional-training confound because arm A was cancelled. If projection does
not help, report that outcome and revise the candidate. Existing results from
other objectives must not be relabelled as evidence for this method.
