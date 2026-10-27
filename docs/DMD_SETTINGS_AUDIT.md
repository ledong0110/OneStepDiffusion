# DMD setting comparison

The running `runs/sd15/scale50k/dmd_seed42` is **DMD-inspired**, not a reproduction
of DMD's SD1.5 experiment. The chosen rank-32 student is a project adaptation,
and other training settings also differ.

Primary reference: [Yin et al., One-step Diffusion with Distribution Matching
Distillation, CVPR 2024, Appendix B.3](https://tianweiy.github.io/dmd/dmd_lowres.pdf).
This table refers to the initial SD1.5 recipe in B.3; the paper also describes
other datasets and additional schedules. It is a paper/configuration audit,
not an official-code equivalence certification.

| Setting | Paper: SD1.5, Appendix B.3 | Running project configuration |
|---|---|---|
| Generator | Full-model training | Rank-32 q/k/v/out LoRA; frozen backbone |
| Fake scorer | Trainable diffusion model | Rank-32 q/k/v/out LoRA |
| Teacher | SD1.5 | Pinned SD1.5 mirror |
| Prompts | LAION-Aesthetic 6.25+ | 50,000 COCO-2017 training captions |
| Optimizer / learning rate | AdamW, 1e-5 | AdamW, G starts at 1e-4 with warmup/cosine; F constant 1e-4 |
| Weight decay / betas | .01 / (.9,.999) | Matches |
| Teacher guidance | 3 | 7.5 for real score and pairs; fake score CFG 1 |
| Regression | VGG LPIPS after VAE decoding | Latent-space squared error |
| Regression weight | .25 | .25, but with a different loss and scale |
| Cached teacher pairs | 500,000 | 5,000 |
| Pair sampling | 50 deterministic sampling steps | 64 DDIM steps |
| Effective G distribution-matching batch | 2,304 | 16 |
| Regression batch | 1,152 | 16 |
| Updates | 20,000 | 18,000 |
| Gradient clipping L2 norm | 10 | 1 |

The project implements the normalized real/fake denoising direction and a
separate fixed noise/teacher-pair regression stream. This establishes the
method's structure; it does not establish implementation equivalence.
Using the same scalar regression weight with LPIPS and latent MSE does not
equalize the regularization strength.

The current DMD process was preserved during the SwiftBrush relaunch. No
learning rate, loss, guidance, dataset, cache, or optimizer state was changed
mid-run. Results remain labelled DMD-inspired. This run cannot support a claim
that published SwiftBrush is intrinsically worse than published DMD.

To build a closer DMD comparison while retaining the user-requested rank-32
student, a separate new run would need the paper's optimizer settings,
decoded-image LPIPS, guidance/pair protocol, and explicitly disclosed
reduced data/compute and LoRA choices. Such a run has not been implemented
or launched by this settings audit.
