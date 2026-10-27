# SwiftBrush implementation audit

The existing `swiftbrush_vsd` experiment is **SwiftBrush-inspired VSD**, not
an implementation-matched reproduction. Its score objective matches the
official finite-input FP32 formula, but its model capacity and training recipe
do not. Poor results from this variant do not establish that published
SwiftBrush performs worse than DMD.

Reference: [official trainer at commit a346fec56484a5de54e062b1b4251f1af1e0bce0](https://github.com/VinAIResearch/SwiftBrush/blob/a346fec56484a5de54e062b1b4251f1af1e0bce0/train_swiftbrush.py),
[official launch recipe](https://github.com/VinAIResearch/SwiftBrush/blob/a346fec56484a5de54e062b1b4251f1af1e0bce0/README.md#training).
The trainer SHA-256 is
`008f8703794ba91f3282b43c22d96a0722de2ce6ee0590711f8bbec8cfdc9343`.

| Component | Official source / README | Existing `swiftbrush_vsd` | Separate `swiftbrush_source` recipe |
|---|---|---|---|
| Generator parameters | Full U-Net | Rank-32 attention LoRA, 6,377,472 trainable parameters | Full SD1.5 U-Net, 859,520,964 trainable parameters |
| Generator optimizer | Adam, LR 1e-6, weight decay 0 | AdamW, LR 1e-4, weight decay .01 | Adam, LR 1e-6, weight decay 0 |
| Scorer optimizer | Adam, LR 1e-3 | AdamW, LR 1e-4 | Adam, LR 1e-3 |
| Learning rate schedule | Constant, no warmup | Generator warmup/cosine; constant scorer LR | Both constant, no warmup |
| Scorer adapter | Rank 64, alpha 108, `to_q`/`to_v`; default PEFT initialization | Rank 32, alpha 32, q/k/v/out; Gaussian initialization | Rank 64, alpha 108, q/v; default PEFT initialization |
| Real teacher | Scorer backbone with adapters disabled | Separate frozen copy | Scorer backbone with adapters disabled |
| Generator score direction | sigma² × (real epsilon − fake epsilon), stopped-target mean MSE | Matches on finite FP32 tensors | Matches on finite FP32 tensors |
| Scorer denoising loss | MSE(alpha × prediction, alpha × noise) | Equivalent alpha²-weighted MSE in FP32 | Same operand scaling as source |
| Scorer samples | Same detached generator outputs made before the generator update | Fresh noise and new generator outputs after its update | Reuses the same detached outputs before its update |
| Text dropout | One 10% dropout decision per batch | Independent decision per image | One decision per effective batch, shared across microbatches |
| Score timesteps | Integers 20 through 980 inclusive | Matches | Matches |
| Scorer timesteps | Integers 0 through 999 inclusive | Matches | Matches |
| EMA | Diffusers `EMAModel`, maximum decay .9999, dynamic initial decay | Fixed .995 | Diffusers `EMAModel`, maximum .9999, dynamic initial decay |
| Terminal clean estimate | Hard-coded alpha-squared .0047 | Scheduler alpha-squared at t=999 | Hard-coded .0047 in training and exported inference |
| CFG | README 4.5, same real/fake CFG | 7.5 | 4.5 by default |
| Teacher / prompts | SD2.1-base / original prompt corpus | SD1.5 / 50k COCO captions | SD1.5 / 50k COCO captions: still a project adaptation |

The new recipe is a **source-aligned training port**, not the original published
experiment. It uses the project's pinned SD1.5 teacher and COCO prompt stream,
18k generator updates, and microbatch accumulation. The README's batch-16
example is simulated by microbatch 1 with accumulation 16. Prompt sampling
uses our deterministic with-replacement stream rather than upstream's shuffled
embedding files. Text is encoded online. FP32 is the default for this recipe;
our locked dependencies and attention kernels differ from upstream's unpinned
environment. For nonfinite scores we fail explicitly; upstream applies
`nan_to_num`. These differences are recorded rather than called exact
end-to-end reproduction.

The source-like generator is about 135 times larger in trainable parameter
count than the project's LoRA students. Keep its results in a separate folder.
The result importer rejects mixing full-generator and LoRA protocols. Its
metadata uses `student_parameterization: full-unet`, student `lora_rank: 0`,
and separate auxiliary rank/alpha/targets. Its native sampling identity also
records the source-specific terminal prediction. Existing runs, checkpoints,
and their optimizer settings have not been rewritten.

## Verification

Run the source-expression comparison:

```bash
.venv/bin/python scripts/audit_swiftbrush.py
# Repeat without network after the pinned source has been cached:
.venv/bin/python scripts/audit_swiftbrush.py --offline
```

This verifies the downloaded source hash and executes only its extracted
predictor and loss expressions on fixed tensors. The initial audit produced
zero maximum error for the clean prediction, generator loss gradient, and
scorer loss gradient. This checks algebra, not generation quality or the entire
training process. The audit result is `artifacts/audit/swiftbrush/parity.json`.

CPU tests check full-generator gradients, q/v-only auxiliary training, exactly
one generator forward per microbatch, fixed learning rates, the EMA schedule,
exact interrupted/resumed generator/scorer/EMA weights, and native exported
inference. A three-update FP32 test on an H200 trained all 859,520,964 generator
parameters, reached a nonzero generator gradient after the expected zero first
update, saved a full checkpoint, exported its EMA, and generated a preview.
This is a functional smoke test, not a trained quality result.

The separate recipe also fixes component logging: numeric losses are averaged
over all accumulated microbatches. Previously `loss` was averaged correctly,
but individual components such as `vsd_loss` showed only the final microbatch.
The same logging correction now applies to future adapted SD runs. It does
not change optimizer updates or the code already loaded by running processes.

## Run separately

```bash
# Inspect the complete configuration without launching:
.venv/bin/python scripts/run_swiftbrush_source.py --gpu 3 --dry-run

# Detached full training followed by the project's 5k-image metric evaluation:
.venv/bin/python scripts/run_swiftbrush_source.py --gpu 3 --background

# Resume the same recipe/checkpoint:
.venv/bin/python scripts/run_swiftbrush_source.py --gpu 3 --background --resume
```

Default output: `runs/sd15/swiftbrush_source_seed42`; metrics:
`results/sd15/swiftbrush_source`. Inspect `launch_status.json`, `launch.log`,
`train.jsonl`, and `preview_1.png` inside the run folder. tqdm and automatic
W&B selection are supported. No existing worker is stopped by this launcher.
The current queue has since been updated: the user requested a rank-32 LoRA
student while keeping the official SwiftBrush optimizer/scorer settings and
CFG 4.5. This is `configs/sd15_swiftbrush_lora.yaml`, active on GPU 2, with
results in `results/sd15/swiftbrush_lora`. Its G uses rank-32 q/k/v/out adapters;
F independently uses rank 64, alpha 108, and q/v targets. The original full-G
profile remains available but is not running. Both native export paths and
exact interrupted/resumed EMA/scorer weights are tested.

The original `swiftbrush_vsd` run was stopped with its step-6000 checkpoint
preserved. The queue controller adopted the existing GPU 0/1/3 processes
without restarting them and retained consistency after SwiftBrush on GPU 2.
The replacement controller records original PID/start-time/argument identities
in its manifest. The shared `runs/sd15/scale50k/launch_status.json` points to it.

Consistency, DMD, and InstaFlow in that study are also documented adaptations.
This audit does not certify them as source-matched implementations.

## Attribution

SwiftBrush is by Thuan Hoang Nguyen and Anh Tran, VinAI Research, CVPR 2024.
The separate trainer ports its algorithm and recipe; this project adds its own
configuration, deterministic COCO stream, microbatching, checkpoint/resume,
tracking, evaluation, and model export integration. The audit executes
expressions extracted from their original trainer. Preserve the upstream
notices when redistributing derived material. An unmodified copy of the
upstream `LICENSE.md` is in `third_party/swiftbrush/LICENSE.md`; its heading is
CC BY-NC-SA 4.0, while the upstream README states CC BY-NC 4.0. The trainer
retains a Hugging Face copyright and Apache 2.0 header. These notices and an
Apache 2.0 license copy are retained without resolving their inconsistency.
Cite [the SwiftBrush paper](https://arxiv.org/abs/2312.05239).
