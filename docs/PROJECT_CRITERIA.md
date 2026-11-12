# CE7454 Project 2: requirements and current evidence

The source of truth is `docs/project_2_handout.docx`. The CE6190 slide deck
and Assignment 2 report concern a different course.

The CE7454 handout requires a **four-page CVPR-format report** and a
presentation video, submitted by one team member. The due date is
**15 November 2026, 11:59 PM**. Groups have 4–6 members. The report contributes
40% of the course grade, and the presentation contributes 10%. The handout's
late penalty is 5% per day for up to three days.

| Report criterion | Weight | Evidence in the four-page paper |
| --- | --- | --- |
| Novelty and difficulty | 25% | Direction-only alignment between independent same-caption latent displacements; relation to DMD and prior response matching; restricted rank-32, one-NFE setting |
| Technical soundness | 25% | Normalized DMD direction, inherited LPIPS term, angular objective, numerical guards, frozen components, fresh-adapter initialization and explicit teacher-target costs |
| Experimental analysis | 25% | Completed 5,000-image FID/IS/CLIP evaluations, five standard baselines, angular-gradient and vector-error analysis, native matched samples and conditioning probes |
| Clarity | 25% | Official CVPR author-kit style, four named members, actual update budgets, concise interpretations and four pages including references |

The four-page report is `CE7454_main_report.pdf`, with source `report/ce7454.tex`.
Rebuild it with `make report`. The separate CE6190 report is
`CE6190_group_11_report_project2.pdf`; its eight-page main-text limit does not apply
to CE7454.

The current proposal is the independently trained 36,000-update angular
response model. It starts from pretrained SD1.5 and fresh rank-32 adapters,
not random backbone weights. LPIPS and its 0.25 weight are inherited from
DMD. The angular coefficient is fixed throughout the 36,000-update run; this report
does not claim that it is optimal.
The original DMD reference differs in initialization history and paired
teacher bank, so its score difference does not isolate the added loss.

The independent zero-response control is not assigned final metrics until
its 36,000-update training and full validation finish. Both reports exclude earlier additional-update pilots. Only completed
36,000-update student checkpoints and full 5,000-image evaluations are admitted. DMD2 and
f-distill remain additional unfinished experiments. One training seed and
validation reuse limit generalization and causal claims.

The presentation is assessed on clarity (20%), visualization (20%), logic
flow (20%) and overall quality (40%). The handout permits one or all members
to appear in the video. Preparing the video remains a separate submission
task. Report automation validates data identities and page counts; it cannot
guarantee a grade.
