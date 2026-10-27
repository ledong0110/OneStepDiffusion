# Assessment requirements and evidence plan

Source of truth: `docs/project_2_handout.docx`, CE7454 Project 2. The local
slide deck is background material from CE6190, not the current marking rubric.

The handout sets **15 November 2026, 11:59 PM** as the due date, groups of 4–6,
and submission of both a **four-page CVPR-format report** and a presentation
video. It says one group member submits on behalf of the team. The handout
states a registration deadline of 7 October, 11:59 PM; the team should check
its own registration status. It specifies a 5%-per-day late penalty for up
to three days. These are transcribed from the provided file, not independently
verified against later announcements.

The report contributes 40% of the course grade; the oral presentation contributes
10%. Within the report, novelty/difficulty, technical soundness, experimental
analysis, and clarity each receive 25%. A grade cannot be guaranteed by setup,
automation, or any particular model result.

| Criterion | Supplied support | Work still needed before submission |
| --- | --- | --- |
| Novelty and difficulty, 25% | Testable noise-aware edge-loss hypothesis, two-to-one distillation, constant-edge control | Review existing gradient/frequency-aware losses; state the precise contribution without claiming first use; explain why this intervention is worthwhile |
| Technical soundness, 25% | Explicit VP/DDIM equations, inverse transition target, frozen SD teacher/VAE/text encoder, LoRA EMA, baked CFG, exact grids, adapter/scorer resume tests | Audit formulas against code; record final commit/configs; disclose differences from the original distillation papers |
| Experimental analysis, 25% | Teacher shortcuts, progressive controls, conditional VSD, consistency distillation, DMD-inspired training, InstaFlow-inspired reflow/distillation, 3-seed full preset, FID/IS/CLIP/latency/memory, protocol validation | Run full experiments; inspect failures and diversity; discuss uncertainty and negative outcomes; analyze lambda sensitivity |
| Report clarity, 25% | Official CVPR style, structured draft, automatic tables/plots, citation list, four-page check | Complete author list and interpretations; replace draft statements; maintain readable figures and four-page length; proofread all claims against JSON |

The current baseline is an adaptation with an epsilon backbone, nonzero terminal
signal inherited from the pretrained scheduler, and clipped-SNR weighting.
The original progressive-distillation paper studies different parameterizations
and schedules. Do not describe this code as a faithful reproduction of its FID.
The proposed gate is small at the highest-noise input; this predicts weaker
direct influence in the one-step stage and should be analyzed, not hidden.

Recommended final evidence:

- The additional direct one-step proposal is a controlled DMD extension with
  LPIPS-VGG regression and teacher-calibrated same-prompt pair-distance matching.
  Four variants isolate latent/perceptual regression and the diversity term.
  All start from the same completed DMD checkpoint and share 5,000 generated
  anchors (2,500 fresh prompts with two noise seeds), 18,000 additional G/F
  updates, rank-32 students, effective batch 16 and one-step/one-NFE inference.
  This extends established DMD/RDD/CoDMD ideas; do not claim first use of
  perceptual or relational distillation. The new question is whether this
  within-prompt objective improves the restricted direct one-step setting.
  FID/IS/CLIP use 5,000 held-out pairs; diversity uses LPIPS-Alex across four
  seeds per prompt for 128 held-out prompts, with a teacher reference. Training
  uses a different LPIPS backbone. Larger pairwise distance alone is not better
  quality. Report decoder/perceptual training overhead and shared pair-generation
  cost. The new experiment has a separate CVPR report and single-seed limits;
  measured benefits remain unestablished until the jobs complete.

- A quality-versus-step plot including the teacher, progressive variants, and
  one-step VSD, consistency distillation, and DMD-inspired training.
- A hardware-controlled latency comparison alongside NFE counts.
- Three training seeds per trained method, with mean and sample standard deviation.
- Fixed-prompt, fixed-noise qualitative comparisons at 8, 4, 2, and 1 steps, including failures.
  The report builder supplies a CVPR figure spanning both columns and imports
  encoded PNG samples from team JSONs. The initial teacher-only reference figure
  is an authentic layout preview, not evidence about trained students.
- Equal generator updates and effective batch sizes across the progressive variants.
- Use the matched one-step study for direct cross-method comparison: all trained
  models use 1 step / 1 NFE. The preliminary four-step PD pilot cannot establish
  an equal-step result against one-step VSD/CD/DMD. Teacher references retain
  their separately labelled step/NFE budgets.
- SwiftBrush-inspired VSD as an external one-step objective comparison: match the
  cumulative generator update budget, disclose additional auxiliary updates and
  LoRA scorer compute, and disclose that our generator uses LoRA whereas SwiftBrush
  trains the full generator. The CIFAR/CelebA variants remain unconditional adaptations.
- Consistency distillation: disclose the VP boundary blend, shared DDIM teacher
  grid, latent L2, and LoRA differences from the original EDM/LPIPS recipe.
- DMD-inspired training: include normalized score direction and a separate cached
  teacher-pair regression stream. Report extra scorer updates, paired forward
  passes, pair-generation teacher NFE, and preprocessing time; distinguish latent
  L2 from the original perceptual loss and this baseline from DMD2.
- InstaFlow-inspired training: disclose the matched LoRA rank, direct velocity
  prediction, 999-based time labels, and cached pair subset. Split the matched
  generator budget between uniform-time reflow and endpoint distillation; report
  guided-teacher and frozen reflow-teacher pair-generation NFE and time.
- Paper figures: methods across columns and fixed prompts down rows; a five-way
  edge-loss visual ablation with identical initial noise; and a 3-by-4 animal/action
  prompt-control grid. The latter is a discrete composition study, not continuous
  embedding interpolation. The main PDF keeps its four-page limit; a separate
  optional qualitative gallery holds all six comparison prompts.
- Recent literature: WaDi (CVPR 2026) and Pixel MeanFlow (ICML 2026) are cited as
  context. Different adapter/pixel-model settings prevent treating their published
  benchmark scores as matched SD1.5/LoRA experimental results.
- Edge-weight sensitivity at 0.03, 0.1, and 0.3, with no-edge and constant-edge
  controls. The automatic table pairs the same training seeds and inference
  budget across all five settings. The queued exploratory pilot has one seed,
  128 evaluation pairs, and 600 G updates per method; final three-seed evidence
  remains separate and requires the full sweep.
- The active `scale50k` development study uses 50,000 training pairs, all 5,000
  validation pairs, 18,000 generator updates per variant, effective batch 16,
  matched 1-step / 1-NFE generation, and seed 42. It supersedes the stopped small
  pilot without overwriting its evidence. Larger data alone does not establish
  successful distillation; inspect previews, learning stability and final samples.
  The new InstaFlow variant uses an explicitly disclosed VP epsilon-to-velocity
  conversion and a small-set learning check before scaling. Three-seed final
  evidence remains outstanding.
- Training stability, peak memory, and cumulative distillation time from run logs.
- A clear account of whether decoded-RGB edge supervision helps FID, prompt
  alignment, perceptual appearance, or none. FID/IS/CLIP do not establish complete
  mode coverage or human preference.

The primary profile uses SD 1.5 at 512px with rank-32 student/scorer adapters,
5,000 COCO-2017 training pairs, and 1,000 fixed validation pairs. These subset
results are not the COCO-2014 benchmark reported in SwiftBrush. Teacher pretraining
overlap is unknown. CFG 7.5 costs two denoiser branches per teacher step;
students use one branch with baked guidance. The proposed edge term uses decoded
RGB, while reconstruction uses latent space. Keep legacy results separate.

The presentation rubric is clarity 20%, visualization 20%, logic flow 20%,
overall quality 40%. A useful video sequence is problem → teacher/student
mechanism → proposed loss → controlled evidence → failure cases → limitations.
Use large fixed-noise image grids, the generated quality plot, and actual
latency numbers. The handout permits either one member or all members in the
video. Share roles and record actual individual contributions.

The report builder checks data consistency and PDF length. It cannot judge
novelty, establish causal improvement, verify a friend's claimed measurements,
or replace the team's analysis. Keep original logs and checkpoints as evidence.
