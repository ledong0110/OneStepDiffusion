# Spatial noise-agreement experiments

This completed two-probe pilot did not improve FID over the fixed DMD reference. The follow-up is [independent-pair agreement](DMD_PAIR_AGREEMENT_RESULTS.md).

The existing best DMD+LPIPS result is retained as the fixed 36,000-update reference; additional baseline training was cancelled at the user's request. The proposal and its variants start from that same checkpoint and receive 500 additional updates (36,500 total), rank 32, effective batch 16, teacher CFG 7.5 and one-step inference. Evaluation uses all 5,000 COCO validation images with the same prompts and noise. The 500-update budget difference means a gain over the frozen reference alone cannot isolate the effect of the new update rule. All three variants use two opposite-noise score probes per generated latent. The candidate applies a spatial agreement map and preserves the paired mean's latent-gradient norm; the controls remove or spatially shift that map. LPIPS remains inherited from DMD. Training score-probe compute exceeds the fixed reference; inference remains one step. This is an exploratory, biased update rule, not a proven improvement or novelty claim.

| Method | Total updates | FID ↓ | IS ↑ | CLIP ↑ |
|---|---:|---:|---:|---:|
| DMD + LPIPS (fixed reference) | 36,000 | 29.94 | 34.12 | 31.10 |
| Spatial noise agreement (candidate) | 36,500 | 30.02 | 34.08 | 31.10 |
| Ablation: paired mean only | 36,500 | 30.09 | 34.05 | 31.10 |
| Ablation: spatially shifted agreement | 36,500 | 30.06 | 34.17 | 31.10 |
