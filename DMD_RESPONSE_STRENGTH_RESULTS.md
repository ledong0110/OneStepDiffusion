# Direction-only response strength pilot

The existing best DMD+LPIPS result is retained as the fixed 36,000-update reference; additional baseline training was cancelled at the user's request. The proposal and its variants start from that same checkpoint and receive 500 additional updates (36,500 total), rank 32, effective batch 16, teacher CFG 7.5 and one-step inference. Evaluation uses all 5,000 COCO validation images with the same prompts and noise. The 500-update budget difference means a gain over the frozen reference alone cannot isolate the effect of the new update rule. The added term aligns the direction of finite latent changes between two independent noises for the same caption. Original DMD and its LPIPS regression remain unchanged. Strengths are calibrated on 32 training captions to median initial latent-gradient ratios of 10%, 30% and 50% relative to DMD, not maintained dynamically during training. Cached teacher pairs and existing regression forwards are reused. Related response-matching work is credited; this is an exploratory extension, not a claim to invent response alignment.

| Method | Total updates | FID ↓ | IS ↑ | CLIP ↑ |
|---|---:|---:|---:|---:|
| DMD + LPIPS (fixed reference) | 36,000 | 29.94 | 34.12 | 31.10 |
| Angular response (10% initial gradient) | 36,500 | 29.63 | 34.27 | 31.11 |
| Angular response (30% initial gradient) | 36,500 | 29.07 | 34.09 | 31.08 |
| Angular response (50% initial gradient) | 36,500 | 28.95 | 33.85 | 31.04 |
