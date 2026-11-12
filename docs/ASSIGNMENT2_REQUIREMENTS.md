# CE6190 Assignment 2 — Group 11

The controlling handout is `docs/coursework_2026 v2.pdf`, especially slides
6–7 and 11–14. This is a separate assignment from the CE7454 four-page paper.
Assignment 2 requires **at least four pages**, with no maximum stated in this
handout. The user's stricter limit is **eight pages for the main report,
including references; the appendix has no page limit**, enforced before publication. The main
report compares the five standard pipelines and the selected proposal, **Ours**.
The proposal's detailed design and ablations appear in the appendix. The CE7454 paper is published separately as `CE7454_main_report.pdf`.

## Submission

- `CE6190_group_11_report_project2.pdf`
- `group_11_code_project2.zip`

The handout specifies one group submission through the course's NTULearn
Assignment 2 submission area, due **17 November 2026**. Submit these actual
files; a repository/cloud link does not replace the deliverables. The code
archive excludes datasets, model weights, environment folders and runtime
logs. Generated scientific figures and numerical tables are included so the
report is self-contained and its source compiles offline.

The supplied members are Le Duc Dong, Nguyen Huu Thuat, Pham Viet Hoang and
La Nguyen Gia Hy. No unverified contribution roles or student IDs are invented.

## Requirement coverage

| Handout requirement | Evidence in the separate report |
|---|---|
| Visual-generation task and technical review | Introduction: efficient one-step text-to-image generation; Methods: five approaches and a supervision/cost comparison |
| At least one evaluation dataset | COCO-2017; 118,287 training-image captions across two disjoint pools, all 5,000 validation images held out from student training |
| At least two methods | Progressive distillation, consistency distillation, DMD, SwiftBrush and InstaFlow, plus the proposed objective |
| Own experimental results | Five standard-pipeline 36k evaluations and the independent 36k proposal, a matched DMD regression control, and native matched figures; identities and actual budgets checked by the builder |
| Hyperparameter-setting analysis | Optimization settings and regression-space analysis; fixed angular coefficient, gradient geometry and response-loss interpretation in the appendix |
| Key-component ablations | Matched latent-MSE versus original LPIPS regression at 36k; the independent zero-response control is included only after its full evaluation finishes |
| Additional proposal experiments and controls | Direction-only response alignment at three training-calibrated strengths; full quality, paired-caption diversity uncertainty, and frozen response diagnostics. Completed final component comparisons are included when available. |
| Comparison, advantages and disadvantages | Method taxonomy, full-validation metrics, training-cost discussion and local implementation limitations |
| Visualization and discussion | Matched baseline-and-Ours comparison, discrete prompt control and embedding interpolation in the main report; response-method diagram and final component samples in the appendix |
| Future extensions | Repeated seeds, coefficient sweeps, capacity/compute controls and semantic evaluation |
| Introduction / methods / experiments and analysis / conclusion | These are the four numbered top-level sections |
| At least four pages | Enforced by the separate PDF builder |
| Report and code | Group-specific root PDF and source-only ZIP |
| Methods must not overlap with Project 1 | **Team confirmation still required:** Project 1 methods were not supplied. No claim of non-overlap is made. |

Additional datasets are encouraged, but are not required. This report uses one
quantitative dataset and states that limitation. Multiple training seeds and
the proposed broader sweeps are future work, not completed experiments.

The concise main manuscript focuses on progressive distillation, consistency,
DMD, SwiftBrush and InstaFlow, with the selected response method shown as Ours.
Its detailed design and ablations are in the appendix.
The long history of discarded projection and score-probe pilots is omitted
from the paper; its configs, measurements and analysis remain in the project.
The previous complete manuscript is preserved under
`artifacts/report_edits/assignment2_before_pruning_20261105.tex`.

An explicitly selected evaluated response checkpoint replaces the earlier
proposal in the main comparison and conditioning figures. The main figure contains
the five standard pipelines, Ours and the teacher. Prompt-control and interpolation outputs are generated
anew in that run's `qualitative` folder, using the existing displayed prompts
and noise seeds. Figure import checks the selected model and actual training
budget; old model outputs cannot be relabelled as the new proposal.

## Build and refresh

```bash
make assignment2
# Validate native results, regenerate assets, compile, and package code:
.venv/bin/python scripts/build_assignment2_report.py --group-id 11 --package

# Compile the saved tables/figures without reading results or model weights:
.venv/bin/python scripts/build_assignment2_report.py --compile-only --package

# Minimal offline LaTeX build, also supported inside the submission ZIP:
cd report
latexmk -pdf -outdir=build assignment2.tex
```

Sources are `report/assignment2.tex` and `report/assignment2_references.bib`.
Generated assets are isolated in `report/generated/assignment2/`; the build
copy is `report/build/assignment2.pdf`. The standard `make report` and existing
report automation refresh both course-prefixed PDFs. After importing new native
results, run `make assignment2` to regenerate assets and the code archive.

The default refresh requires all nine completed results and both qualitative
panels; it fails rather than filling missing cells with invented values. Metric
tables and native sample figures come from verified result JSONs. `metrics.csv` and
`provenance.json` record the report evidence outside the manuscript. Figure
helpers verify shared captions/noise and the evaluated checkpoint identities.
The submitted archive includes generated assets but not the large native
evaluation payloads; it supports offline compilation, while regenerating all
assets requires the original evaluation JSONs or fresh evaluations.

## Interpretation checks

- Every displayed student has exactly 36,000 cumulative generator updates,
  one-step / one-NFE inference and rank-32 adapters. Additional-update pilot
  results are excluded from both reports.
- Ours starts from pretrained SD1.5 with fresh adapters and obtains FID 28.58,
  IS 33.02 and CLIP 30.91 on all 5,000 validation captions. The original
  DMD+LPIPS reference has FID 29.94, IS 34.12 and CLIP 31.10. Its initialization
  history and paired bank differ, so this is a complete-recipe comparison.
- LPIPS and its 0.25 coefficient come from DMD. The proposed addition is a
  finite-displacement angular constraint. The report does not claim to invent
  response alignment, cosine loss or perceptual regression.
- The method analysis derives the angular gradient, relates it to full vector
  error and distinguishes noise-response constraints from text conditioning.
  The coefficient remains fixed; no unreported 36k weight sweep is claimed.
- Only completed full-validation evaluations receive metric rows. The matched
  independent zero-response control and recent baselines are not assigned
  invented final scores while evaluation is pending.
- The CE7454 paper has a quality comparison, signed metric changes and training
  exposure tables; the earlier regression-control table remains excluded.
  The CE6190 paper retains its valid 36k regression ablation. Selected dog,
  teddy-bear and lighthouse examples use shared prompts and noise; all
  5,000 generated validation images enter the numerical evaluation.
- SwiftBrush's CFG 4.5/FP32 and distinct optimization remain disclosed. Other
  main recipes use CFG 7.5/BF16. Equal student updates do not equalize teacher
  cost or auxiliary compute, and these local recipes are not original-scale
  full-model reproductions.
