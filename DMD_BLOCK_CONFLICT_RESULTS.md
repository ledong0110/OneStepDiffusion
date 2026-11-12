# Earlier module-projection experiments

These are completed exploratory projection experiments, not the current proposal. The subsequent spatial-noise pilot is recorded in [the spatial noise-agreement study](DMD_NOISE_AGREEMENT_RESULTS.md).

The existing best DMD+LPIPS result is retained as the fixed 36,000-update reference; additional baseline training was cancelled at the user's request. The proposal and its variants start from that same checkpoint and receive 500 additional updates (36,500 total), rank 32, effective batch 16, teacher CFG 7.5 and one-step inference. Evaluation uses all 5,000 COCO validation images with the same prompts and noise. The 500-update budget difference means a gain over the frozen reference alone cannot isolate the effect of the new update rule. Projection is an asymmetric PCGrad-style application; quality evaluation is exploratory.

| Method | Total updates | FID ↓ | IS ↑ | CLIP ↑ |
|---|---:|---:|---:|---:|
| DMD + LPIPS (fixed reference) | 36,000 | 29.94 | 34.12 | 31.10 |
| DMD + module projection | 36,500 | 30.16 | 34.12 | 31.11 |
| Module-norm ablation | 36,500 | 30.00 | 33.95 | 31.12 |
