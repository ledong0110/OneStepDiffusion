# Assignment 2 discussion notes — Group 11

The main report is a review and empirical comparison of **progressive
distillation, consistency distillation, DMD, SwiftBrush and InstaFlow**.
The selected response-alignment proposal is included as **Ours** in the main
table, matched sample comparison and conditioning figures. Its detailed design
and ablations are in the appendix. The main
report, including references, is limited to eight pages; the appendix is unlimited.

## What is compared fairly?

All five students use the SD1.5 backbone, rank-32 generator adapters,
36,000 total generator updates, effective batch 16, and one-step / one-NFE
inference. Ours uses the same generator capacity and inference budget, with
36,500 total updates. Evaluation uses the same 5,000 COCO validation captions, reference
images and initial noise stream. The teacher uses 64 DDIM steps / 128 NFE.

The recipes do not have equal total training compute. Teacher-pair generation,
scorer updates and decoder/perceptual backward passes have different costs.
SwiftBrush retains teacher CFG 4.5, FP32 and source-style optimization; the
other students use CFG 7.5 and BF16. These are local LoRA adaptations, not
exact reproductions of the published full-model systems.

Training supervision also differs. PD and consistency encode real training
images; DMD and SwiftBrush sample captions and generated images. DMD score
matching and fake-scorer updates use the broader caption stream. Its paired
regression and the appendix response loss use the smaller bank of 5,000
teacher targets from 2,500 captions. InstaFlow learns from generated couplings.
The 5,000 validation images are held out from student training in every case.

## Which standard pipeline is strongest here?

DMD with its original LPIPS regression has the lowest student FID, **29.94**,
and the best IS and CLIP among the five standard pipelines in this study.
The teacher has FID **24.10**. This describes the recorded setting and one
training seed, not a universal ordering of the original methods.

## Why LPIPS and why 0.25?

LPIPS regression and its default weight 0.25 are ingredients of original
DMD. They are not claimed as our invention. In the matched regression-space
ablation, replacing latent MSE with LPIPS reduces FID from **34.20 to 29.94**.
Both variants have the same initialization, paired bank, caption stream and
36,000-update budget. Equal scalar coefficients do not imply equal gradient
norms, so this tests the resulting regression recipe.

## Does low loss establish good image generation?

No. A surrogate can have a small numeric value while its gradient estimates,
image structure or text adherence remain poor. Conclusions use full-set
FID/IS/CLIP and direct sample inspection. Loss scales are not comparable
across different objectives.

## What does the appendix add?

The response extension keeps DMD and its LPIPS term, and adds
`lambda * (1 - cosine(G(z2,c)-G(z1,c), T(z2,c)-T(z1,c)))` for independent cached
noises sharing a caption. It tests direction agreement without explicitly
matching displacement magnitude. Existing paired-regression forwards are
reused; inference remains one NFE.

Noise-response matching already has prior work, notably GAD, and cosine
alignment is established. The appendix tests a specific angular formulation
on independent cached pairs; it does not claim to invent response matching.
Independent noises give finite displacements, not local Jacobian estimates.

## Why those response weights?

A separate 32-caption training-only calibration measures latent-gradient
norms at the unchanged DMD checkpoint. Fixed angular coefficients
0.05970, 0.17910 and 0.29849 target initial gradient ratios of 10%, 30% and
50% relative to DMD. The ratios are not enforced dynamically. No validation
images enter coefficient calibration.

## Is the response result better?

The completed 500-update pilots have full-validation FID **29.63, 29.07 and
28.95** at the three strengths, versus **29.94** for the frozen DMD reference.
The 50% setting has lower IS and CLIP than 30%, so it is not a win on every
metric. Its pairwise LPIPS is **0.7101**, versus **0.7086** for DMD. The
unadjusted paired-caption interval for that change is approximately
**[0.00011, 0.00297]**; this is a small perceptual-variation change, not proof
of improved semantic diversity or training-seed robustness.

These pilots have **36,500 total updates**, 500 more than the reference.
The user requested keeping the DMD reference fixed. Therefore the difference
cannot isolate additional optimization from the added response term.
Selection used repeated validation inspection and is exploratory.

A separate final study has completed **1,000 additional updates
(37,000 total)** for angular, magnitude-only and full-vector objectives,
all from the same fixed 36k checkpoint. Their FIDs are **29.27, 29.84 and
29.96**, respectively. Angular has the lowest FID and highest measured
within-caption LPIPS among those controls, while magnitude has the highest
IS and CLIP. The vector control has the lowest training-bank latent MSE,
but that does not translate into the lowest FID.

The selected checkpoint remains the 500-update angular model at FID **28.95**.
The completed 1,000-update study supports comparing the response components,
but its extra training does not beat that checkpoint's FID. The main comparison
and conditioning figures use the selected 36.5k model; the appendix distinguishes
these from the 37k component results.
All final arms have full quality, diversity, matched native samples and frozen
diagnostics; no further baseline training was performed.

## How are samples and uncertainty interpreted?

The main figure contains the five standard pipelines, Ours and the teacher at
matched captions and noise. Ours uses authentic images from the selected
checkpoint. Its prompt-control and interpolation seeds reuse the earlier
report's selection without reselection. Images are illustrative; all 5,000
validation outputs enter the quality metrics. Text interpolation mixes CLIP
hidden states, not pixels.

Caption-bootstrap intervals keep models and image seeds fixed. One training
seed does not establish run-to-run uncertainty. Future work needs repeated
seeds, matched compute/guidance settings, an untouched test set, and object,
counting and spatial-relation evaluation.
