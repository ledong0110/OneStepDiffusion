# Selected response method and component ablations

The existing best DMD+LPIPS result is retained as the fixed 36,000-update reference; additional baseline training was cancelled at the user's request. The proposal and its variants start from that same checkpoint and receive 1,000 additional updates (37,000 total), rank 32, effective batch 16, teacher CFG 7.5 and one-step inference. Evaluation uses all 5,000 COCO validation images with the same prompts and noise. The 1,000-update budget difference means a gain over the frozen reference alone cannot isolate the effect of the new update rule. The selected angular response loss is compared with magnitude-only and full-vector response objectives. Each starts from the fixed 36k parent, not from a tuned pilot export. All added losses use the same 50% median initial latent-gradient ratio, calibrated on training captions; the ratios are not maintained dynamically. Original DMD and its LPIPS term remain unchanged. Exploratory selection rationale: Select the 50% initial-gradient angular setting for its lowest pilot FID (28.9540 versus 29.0722 and 29.6275) and maintained measured within-caption variation. IS (33.8479) and CLIP (31.0402) are slightly lower than at 30%; the choice prioritizes FID and is exploratory. Matched native samples preserve recognizable subjects and layouts without establishing visual superiority. Use 1,000 additional updates, twice the pilot budget, to test the angular objective against magnitude-only and full-vector controls under the same budget and fixed 36k initialization. The original DMD reference is not retrained.

| Method | Total updates | FID ↓ | IS ↑ | CLIP ↑ |
|---|---:|---:|---:|---:|
| DMD + LPIPS (fixed reference) | 36,000 | 29.94 | 34.12 | 31.10 |
| Angular response (50% initial gradient) | 37,000 | 29.27 | 33.50 | 31.00 |
| Magnitude response (50% initial gradient) | 37,000 | 29.84 | 34.42 | 31.10 |
| Vector response (50% initial gradient) | 37,000 | 29.96 | 33.39 | 30.98 |
