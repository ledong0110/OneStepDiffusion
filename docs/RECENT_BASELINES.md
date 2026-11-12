# DMD2 and f-distill: matched 36,000-update study

These are **local SD1.5 / rank-32 adaptations**, not claims of reproducing
the papers' full-model benchmarks. The working environment is unchanged.

| Setting | DMD2 | f-distill |
|---|---|---|
| Distribution objective | Reverse-KL score difference | Jensen–Shannon weighted score difference |
| Student | SD1.5 attention LoRA, rank 32 | Same |
| Initialization | Pretrained SD1.5 | Same |
| Student updates | 36,000 | 36,000 |
| Training captions/images | First 50,000 COCO train images for 18,000 updates, then the disjoint remaining 68,287 for 18,000 | Same |
| Student effective batch | 16 (2 × 8 accumulation) | Same |
| Fake scorer | Full U-Net, x0 MSE, 324,000 updates | Same |
| Real-image discriminator | Frozen teacher mid-block + trainable convolutional head, 324,000 updates | Same |
| Teacher / student CFG | 1.75 / 1.0 | Same |
| Optimizers | AdamW, LR 1e-5, betas 0.9/0.999, weight decay .01 | Same |
| Generator GAN weight | .001 | Same |
| LPIPS / paired regression | None | None |
| Inference / final evaluation | 1 step, 1 NFE; all 5,000 COCO validation captions and references | Same |

The caption policy, image ordering, two disjoint training pools and held-out
validation set match the existing project. The first pool uses replacement
sampling; the second uses shuffled epochs. Images are downloaded automatically
and center-fitted to 512×512 without horizontal flips. No validation images
are used for discriminator training.

## Source alignment and differences

The audited implementation reference is
[NVlabs/FastGen](https://github.com/NVlabs/FastGen/tree/3943fd5cfb2a8266ec53e7e541e4b8d2e5b1e94d),
commit `3943fd5cfb2a8266ec53e7e541e4b8d2e5b1e94d` (Apache-2.0).
Pinned files and hashes are in `third_party/fastgen_reference/source.json`.
The SD1.5 configs specify teacher CFG 1.75, GAN weight .001, x0 fake-score
regression, and frequency 10: nine auxiliary updates followed by one student
update. That means **360,000 outer iterations**, not 36,000 outer iterations.
The original DMD2 repository uses a different ratio convention; we follow and
identify the FastGen SD1.5 schedule explicitly.

The loss functions are checked against the actual pinned source. The student
surrogate has detached teacher/fake targets and an outer per-sample weight,
so the f-distill gradient is multiplied by h, not h squared. JS weights use
clipped density ratios, ten timestep bins and batch-global normalization over
all 16 samples; normalizing singleton microbatches would erase the method.
The density history, all three optimizers, fake scorer, discriminator,
student/EMA and RNG are saved for exact resume.

Project adaptations: rank-32 student instead of a full student; COCO instead
of the source training corpus; effective batch 16 instead of 2,048; discrete
VP timesteps 1–999; BF16 compute with FP32 optimized parameters; student EMA
.995 and gradient clipping 1. Thus this matches data, student capacity,
student-update budget and inference cost, **not total training compute**.
Teacher CFG also differs from the older 7.5/4.5 cohorts. The report retains
these distinctions instead of presenting a fully controlled paper ranking.

Papers: [DMD2 (2024)](https://arxiv.org/abs/2405.14867),
[f-distill (2025)](https://arxiv.org/abs/2502.15681).

## Launch and monitor

```bash
.venv/bin/python scripts/launch_recent_baselines.py launch
.venv/bin/python scripts/launch_recent_baselines.py status
tail -f runs/sd15/recent36k/queue.log
```

The detached supervisor survives closing VS Code or an SSH terminal while
the server remains running. It considers GPUs 0, 1 and 2, starts at most two
workers, and requires 64 GiB free GPU memory and 70 GiB free disk before a
launch. Training is capped at 48 GiB of GPU allocations. Active jobs reserve
additional disk capacity. It never stops other processes or deletes old runs.
It records process start times, command lines and working directories to
avoid duplicate workers or mistaking a reused PID for an existing job.

Each worker downloads/verifies the shared data under a cache lock, performs
two actual student updates as a GPU preflight, saves/reloads the complete
checkpoint, and resumes to 36,000. Those first two updates count toward the
budget. A preflight failure stops that method rather than dropping a loss or
shrinking the fake scorer silently. Logs show tqdm progress in student updates;
W&B uses the project's existing auto-online/offline behavior.

Every 1,000 updates, the full resumable `last.pt` is atomically replaced,
the student adapter is archived, and eight fixed validation samples are
evaluated. **Checkpoint evaluation is qualitative; it is not checkpoint FID.**
After the final update, the worker evaluates FID, IS and CLIP on all 5,000
held-out images, checks provenance, then rebuilds the Assignment 2 PDF/ZIP.
Only completed full evaluations enter its additional-baselines table.

Run folders: `runs/sd15/recent36k/{dmd2,f_distill}_seed42/`.
Results: `results/sd15/recent36k/`.
Report: `group_11_report_project2.pdf` beside the README.

A failed job is not repeatedly relaunched. Inspect its log, fix the cause,
then use `launch --retry-failed` to request one retry, either from the live
supervisor or a restarted one. Code/config changes after scheduling are
rejected by the pinned queue plan; an intentional revised experiment needs a
new plan/run location. A server reboot requires rerunning the launch command.

Validate the implementation before launching:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m pytest -q tests/test_recent_baselines.py tests/test_recent_queue.py
```

## Faster execution with a helper GPU

The optional performance path runs generator, teacher, scorer and discriminator
on GPU 3, and overlaps frozen FP32 VAE encoding on GPU 2. Each helper process is
capped at 2.5 GiB. Previews decode one image at a time on the helper to respect
that cap. No other GPU processes are stopped or moved.

Eight logical microbatches of two can be grouped into two execution microbatches
of eight. This preserves the caption ordering and **effective batch 16**, all
nine auxiliary updates, guidance, losses, learning rates and 36,000-update
budget. Fused AdamW retains optimizer moments/counters; disabling activation
checkpointing spends additional memory to avoid recomputation. Larger kernels
and different RNG call shapes change the numerical trajectory, so old/new
weights are not claimed to be identical. Execution settings are saved with
checkpoints, and resume refuses mismatched settings.

On the shared H200, two-update probes measured the following second-update
times while the original two production runs continued:

| Execution | Seconds / student update | Peak training memory |
|---|---:|---:|
| Original | 39.9 | 21.1 GiB |
| Fused optimizer, no activation checkpointing | 34.2 | 22.1 GiB |
| Above plus GPU 2 VAE helper | 30.4 | 21.8 GiB |
| Above with microbatch 4 × accumulation 4 | 20.7 | 24.1 GiB |
| Above with microbatch 8 × accumulation 2 | 16.4 | 28.7 GiB |

The f-distill probe also completed at 16.4 seconds/update with its full
batch-global JS weighting. These are short measurements under shared load;
production throughput must be measured after rollout. Measurements are in
`artifacts/performance/`.

```bash
.venv/bin/python scripts/accelerate_recent_baselines.py launch --train-gpu 3 --helper-gpu 2 --micro-factor 4
.venv/bin/python scripts/accelerate_recent_baselines.py status
```

The detached rollout worker waits for each native checkpoint-evaluation child.
It recognizes relative and absolute checkpoint arguments and checks that
releasing the old process leaves at least 34 GiB for the measured 32 GiB
training cap. It also checks helper-GPU headroom and waits when capacity is
insufficient. Existing training keeps running during that wait. Transitions
are serialized until the first resumed update completes.
At that boundary the training update and resumable save have finished. It
pauses the CPU parents, verifies all saved counters/configuration, lets the
preview finish, and replaces only that run's worker. It reloads the complete
checkpoint, retaining student, scorer, discriminator, optimizers, EMA, ratio
history and RNG state. It never rolls back to an older checkpoint or moves
live files. `speedup_resume.json` records the checkpoint hash and process
identities; `execution.json` records the selected runtime settings.
