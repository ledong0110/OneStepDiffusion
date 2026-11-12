# Direction-only finite-response pilot

The fixed DMD+LPIPS reference has FID **29.9376** on all 5,000 COCO validation
images. The completed two- and four-probe score-agreement pilots did not beat
it. They are closed experiments, not the selected final method.

This experiment changes the training target. For two independent noises with
the same caption, form student and cached teacher latent displacements
`dS = G(z2,c) - G(z1,c)` and `dT = T(z2,c) - T(z1,c)`. Add
`lambda * (1 - cosine(dS, dT))` to original DMD with its existing 0.25 LPIPS
term. LPIPS belongs to DMD and is not our contribution. No edge loss is used.

The added term aligns direction without penalizing displacement magnitude or
pair centre. For nonzero displacements, its gradients sum to zero across the
pair and are perpendicular to the student displacement. These are first-order
latent-space properties, not guarantees about the combined optimizer update.
Already-computed paired-regression outputs are reused: no additional generator
or teacher forward passes, model parameters or inference steps are introduced.

## Evidence and attribution

A frozen diagnostic on 128 training captions found mean finite-response cosine
0.7263 and mean student/teacher norm ratio 1.0697. This supports testing direction
correction; it does not demonstrate image-quality improvement or semantic
diversity collapse. The independent noises are not nearby perturbations, so
these measurements are finite secants, not local Jacobian estimates.

[GAD](https://arxiv.org/html/2606.01651v1) matches local noise responses through
squared vector differences using nearby noises. Response matching, relational
distillation and cosine alignment have prior art. This project tests a specific
direction-only extension on cached independent pairs; it does not claim to
invent those ingredients or establish global novelty.

## Predeclared strengths and budget

A separate 32-caption training-only calibration measures the median ratio of
added latent-gradient norm to the original DMD gradient at the frozen EMA
reference. The three target ratios and angular weights are:

| Initial gradient fraction | Angular loss weight |
|---:|---:|
| 0.10 | 0.059698801989226266 |
| 0.30 | 0.17909640596767878 |
| 0.50 | 0.2984940099461313 |

These are fixed weights, not an adaptive guarantee about gradients later in
training. Calibration performs no optimizer updates and verifies unchanged
reference weights. No validation images are used in calibration.

Each arm forks the same completed 36,000-update state, preserving student,
EMA, scorer, optimizers and RNG. It receives 500 additional generator/scorer
updates (36,500 total), rank 32, batch 2 x accumulation 8, learning rate 1e-5,
25-step warmup, cosine decay, teacher CFG 7.5 and inherited LPIPS coefficient
0.25. Original single-probe DMD remains intact. The fixed baseline is not
retrained, so a gain over it alone cannot isolate extra training from the new
loss. All three strengths have the same budget.

## Evaluation and decisions

Every arm receives the existing full 5,000-image FID/IS/CLIP protocol, 128-caption
four-seed diversity evaluation and native images at the same prespecified
prompts/noises. The detached supervisor refreshes the report after completed
evaluation. Low training loss alone is not selection evidence.

After full evaluation, the same frozen 128-caption diagnostic can be applied
to each completed EMA export:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/analyze_teacher_responses.py \
  --study dmd_response_strength --variant angular_30 --gpu 0
```

The command refuses live training runs and verifies that the export is the
model used for the full 5,000-image evaluation. It saves a separate
`response_diagnostic.json` in that run's folder and cannot overwrite a
different model's diagnostic. These captions and noises belong to the
regression training bank: alignment changes diagnose the optimized mechanism,
not generalization or image quality. All three pilot checkpoint diagnostics
are complete; this command does not start another training run.
The report automatically includes completed diagnostics as a separate table of
response cosine, response-norm ratio and pointwise latent MSE. It checks the
evaluated model, source checkpoint, teacher bank and exact caption/noise protocol
before displaying measurements, and includes the source evidence in the code
submission. Pending diagnostics do not produce a results table.

The first two pilot evaluations required a metadata-only repair: their exports
recorded the complete two-seed teacher-bank protocol but omitted the flattened
pair-count fields expected by the result importer. The original result JSONs
and repair audit are preserved under `artifacts/dmd_response_strength/metadata_repair`.
All measured scores and model weights were unchanged. New exports include the
fields, and evaluation can recover them from a matching recorded protocol.

All three 500-update pilots have completed full quality, diversity and native
sample evaluation. The 50% angular setting has the lowest measured pilot FID
(28.9540, versus 29.9376 for fixed DMD), with IS 33.8479, CLIP 31.0402 and mean
within-caption LPIPS-Alex 0.710145. Its IS and CLIP are slightly lower than at
30%; the choice prioritizes FID. These measurements support an exploratory
follow-up, not a general novelty or causal improvement claim.

The prepared `dmd_response_final` study compares angular, magnitude-only and
full-vector losses at a training-calibrated 50% initial gradient ratio. Each
receives 1,000 additional updates (37,000 total), starting independently from
the fixed 36k parent. They do not inherit differently tuned pilot checkpoints.
The manifest pins the completed pilot results and the selection rationale.
The detached supervisor's per-arm state is recorded in
`runs/sd15/dmd_response_final/status.json`. The 1,000-update angular arm has
completed full quality, diversity, native samples and the frozen diagnostic:
FID 29.2656, IS 33.5017, CLIP 30.9993 and within-caption LPIPS-Alex 0.710336.
Its FID remains below the fixed reference's 29.9376, but is above the
500-update pilot's 28.9540; IS and CLIP are also lower than the pilot's.
All eight retained validation samples preserve baseline-like compositions,
with local texture changes and existing attribute errors. They do not establish
general visual superiority. The magnitude control has also completed quality,
diversity, native samples and its diagnostic: FID 29.8384, IS 34.4229, CLIP
31.0981 and LPIPS-Alex 0.704859. Angular has lower FID and higher within-caption
perceptual variation; magnitude has higher IS and CLIP. The frozen training-bank
diagnostic reflects the intended components: angular raises response cosine
to 0.7788, while magnitude brings the response-norm ratio to 1.0089.
These mechanism checks do not establish held-out quality or semantic diversity.
The vector control is also complete: FID 29.9622, IS 33.3914, CLIP 30.9782
and LPIPS-Alex 0.705259. Its response cosine is 0.7742, response-norm ratio
0.9711 and training-bank latent MSE 0.3318. It has the smallest pointwise
training-bank error, but the highest FID among the final response controls.
All three final arms have completed full quality, diversity, matched native
samples and frozen diagnostics. Their checkpoints and 500-update snapshots
are preserved.

After reviewing the complete component study, retain the 500-update angular
checkpoint as the selected model: its FID 28.9540 is below every 1,000-update
arm. At the equal 37k budget, angular alignment has the lowest FID and highest
measured within-caption LPIPS, while magnitude has the highest IS and CLIP.
Matched native samples preserve coherent subjects but do not establish general
visual superiority. No further training is justified by these results alone.
Selection remains exploratory and the fixed DMD reference remains unchanged.
The selected inference export is
`runs/sd15/dmd_response_strength/angular_50/stage_1`, with its original
configuration at `configs/generated/dmd_response_strength/angular_50.yaml`.
`configs/assignment2_selected_method.json` pins this model and the completed
final-study evidence. The 37k comparison exports remain under
`runs/sd15/dmd_response_final/{angular_50,magnitude_50,vector_50}/stage_1`.

The Assignment 2 main report compares the five standard pipelines and the selected
response proposal (Ours), including its measured quality and native samples.
Detailed response design and ablations stay in its appendix. Publication limits the main
report, including references, to eight pages; the appendix has no page limit. The final
component section appears only after all three arms finish full quality,
diversity, matched native samples and response diagnostics.
For the final study, the supervisor also runs the frozen response diagnostic
after quality, diversity and matched samples finish, before its report refresh.

Prepare or resume the pilot with:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python scripts/launch_dmd_conflict.py \
  --study dmd_response_strength --gpus 0 1 --max-parallel 2 --min-free-gib 16
```

The launcher verifies the pinned calibration, parent checkpoint and teacher
bank. Results appear in `DMD_RESPONSE_STRENGTH_RESULTS.md`; the supervisor state
is under `runs/sd15/dmd_response_strength/status.json`. The code package includes
the frozen diagnostic and calibration evidence, but excludes model weights and
datasets.
