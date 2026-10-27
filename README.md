# Few-step Stable Diffusion: experiments and CVPR report

The active study uses **Stable Diffusion 1.5 at 512×512**, rank-32 LoRA students
and **36,000 total training steps**. The proposal combines DMD, perceptual
regression and teacher-calibrated same-prompt diversity matching. Comparisons
include progressive distillation (PD), SwiftBrush,
consistency distillation, DMD and InstaFlow, plus three factorial ablations.
Historical edge-loss experiments, CIFAR-10 and CelebA-HQ remain
optional profiles with separate results.

The supplied handout defines CE7454 Project 2: a four-page CVPR report and
presentation video. See [the rubric evidence plan](docs/PROJECT_CRITERIA.md).
The proposed improvement is a hypothesis; setup cannot guarantee a grade.

## Current paper and selected demo

Run `make report` or `.venv/bin/python scripts/build_active_report.py` to rebuild
**[main_report.pdf](main_report.pdf)** beside this README and the optional
**`report/build/qualitative.pdf`** gallery. The LaTeX build copy remains at
`report/build/main.pdf`. Successful automatic rebuilds refresh the root-level PDF.
The primary paper uses the active proposal and non-edge baselines.
All numerical rows require completed **36k-step, one-step / one-NFE** results
evaluated on **all 5,000 COCO-2017 validation images**. Unfinished experiments
are marked pending during training. All nine current experiments have finished
and passed full validation. All baselines appear in one table; the caption
records SwiftBrush's CFG-4.5/FP32 settings. The full proposal achieves FID 30.40;
the perceptual-only ablation achieves 29.94, so the final analysis reports the
diversity term's tradeoff rather than claiming it improves every metric.

`configs/active_study.json` points to the result folders; native teammate JSONs
can be placed at the expected method/seed filenames there and imported by
the same command. The builder checks result schemas, config/checkpoint identity,
caption/reference hashes, total update budget and the official validation IDs.
`report/generated/active_study/summary.csv` and `manifest.json` record the values
and provenance behind the paper.

The main comparison uses the user-selected dog-in-flowers, teddy-bear and
lighthouse subjects. Dog/lighthouse use the original preset prompts and noise;
the teddy-bear row reuses the saved validation outputs. Every method uses the
same caption and noise within each row, with checkpoint identity checks.
`configs/report_selection.json` points to this portable native comparison.
The generator `scripts/generate_main_comparison.py --gpu 0 --watch` also fills
in newly evaluated baselines; it uses CPU text/VAE offload and a 4 GiB GPU limit.
All eight saved validation outputs and selected failure cases remain in the
gallery; all 5,000 validation outputs contribute to FID/IS/CLIP.
The 3-by-4 animal/action grid uses the evaluated 36k proposal and one complete
grid selected from four candidate seeds, with shared noise seed 12546.
The compact dog-to-cat figure uses **real CLIP hidden-state interpolation** at
weights 0, .25, .5, .75 and 1, with fixed noise within each row. Two rows are
selected from three generated seeds. No generated image is edited or blended.
Change the selection file and rebuild to choose other recorded samples.

To regenerate native qualitative panels with CPU text/VAE and a memory-limited
GPU U-Net (using existing completed model weights):

```bash
.venv/bin/python scripts/generate_active_qualitative.py --gpu 1 --resume
.venv/bin/python scripts/generate_prompt_interpolation.py --gpu 1
make report
```

The active training workers automatically evaluate their final checkpoints on
all 5,000 validation images. Ablations also run the four-seed diversity check.
Start or restore the detached report watcher with:

```bash
make report-watch
# Equivalent: .venv/bin/python scripts/build_active_report.py --background
cat report/build/active_report_status.json
tail -f report/build/active_report_watch.log
```

The watcher checks every minute and updates tables, comparison figures and both
PDFs as each final result arrives, without waiting for the whole queue. It also
imports teammate JSONs placed in the configured result folders. It retries
temporary file/build errors and reuses an existing watcher if launched twice.
The watcher uses CPU only and stays running after VS Code closes while the host
remains on. After a server restart, run `make report-watch` again. Use `--watch`
instead of `--background` to run it in the foreground.
The earlier paper is preserved in `report/legacy_edge.tex`; explicit legacy
`scripts/build_report.py --results-dir ... --profile ...` builds now write
`report/build/legacy_edge.pdf`, leaving the active paper intact.

## Install and download

Use Python 3.12, pip, Git, and an NVIDIA driver compatible with CUDA 12.8.
Python 3.10–3.12 are supported; 3.12 uses the tested dependency lock.

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Setup installs pinned dependencies and caches the pinned SD teacher, COCO
captions, 5,000 training images, 1,000 validation reference images, and Inception
and CLIP metric weights. It initializes Git if needed. Downloads are cached;
COCO captions are fetched with verified HTTP ranges, followed by individual
images, avoiding the full training archive. No account or API key is required.
Sources: the public [SD 1.5 mirror](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
and official [COCO](https://cocodataset.org/#download) S3 bucket. Revisions,
caption manifests, and annotation hashes are recorded locally. Validation is
held out from student training; overlap with teacher pretraining is unknown.

The default microbatch is 1, accumulating 4 microbatches. Frozen U-Net weights
use bf16, adapters/diffusion math use float32, and the VAE uses float32.
Activation checkpointing reduces memory. A GPU with roughly 24 GB or more is a
starting point; check actual peak memory in a pilot. Verification on H200 does
not guarantee a particular memory requirement on another GPU.

For the legacy CPU smoke setup:

```bash
FSD_PROFILE=cifar10 FSD_TORCH_INDEX=https://download.pytorch.org/whl/cpu bash scripts/setup.sh
make smoke
make check
```

Smoke uses a tiny unconditional model and fake data; it cannot populate research
tables or figures. Tests use small local models without SD downloads. For PDF
compilation install `latexmk`, `texlive-latex-extra`, `texlive-fonts-recommended`,
and `poppler-utils`. These are available on this host. The official CVPR 2026
style is vendored; see `report/THIRD_PARTY.md`.

## Launch experiments

Check `nvidia-smi`, select an available GPU, inspect the plan, then run a pilot:

```bash
bash scripts/run_experiments.sh --preset pilot --gpu 3 --dry-run
bash scripts/run_experiments.sh --preset pilot --gpu 3 --resume
# After fixing settings:
bash scripts/run_experiments.sh --preset full --gpu 3 --dry-run
bash scripts/run_experiments.sh --preset full --gpu 3 --resume
```

The SD pilot trains baseline/proposed through 32, 16, 8, and 4 steps, with 100
updates per stage. VSD, consistency, and DMD each receive 400 generator updates;
VSD/DMD each add 400 scorer updates. DMD precomputes 64 teacher regression pairs.
Evaluation
uses 128 fixed validation pairs. It checks stability and resource costs; disclose
validation reuse when selecting hyperparameters.

### Current training scope (21 October)

The active study now contains only the direct one-step DMD perceptual/diversity
proposal, its three factorial controls, and five non-edge baselines: plain PD,
DMD, consistency, InstaFlow and SwiftBrush. **No further edge experiments are
scheduled.** Historical edge checkpoints/results remain available as prior work.
`configs/active_study.json` records the selected methods and excludes the former
edge proposal, constant-edge control and both edge-weight ablations.

```bash
cat runs/sd15/active_one_step/launch_status.json
cat runs/sd15/active_one_step/summary.csv
```

GPU 0 resumes plain PD from its saved continuation checkpoint. GPU 1 runs the
perceptual-only DMD ablation and GPU 2 the diversity-only ablation, in parallel.
The grouped DMD control and InstaFlow remain on GPU 3. The full new proposal,
continued DMD, consistency and SwiftBrush have already finished their 36,000
cumulative updates. Per-run `selected_status.json` and `selected_launch.log`
track the three new parallel workers. The original DMD controller verifies and
reuses externally completed ablation results under the same experiment locks.

Every retained method still targets **36,000 cumulative generator updates** and
**one-step / one-NFE inference**; microbatches, data and objectives remain fixed.
The parallel ablations perform their 5,000-image FID/IS/CLIP and four-seed
diversity evaluations automatically. The old mixed-study supervisor is retained
only to finish its already-running InstaFlow worker; use the consolidated active
status above rather than its historical failed edge-worker entries.

The old matched pilot has been stopped at the user's request. Its checkpoints
and results remain under `fair_pilot`. The `scale50k` study is training, with a
fresh-data one-step continuation queued behind each GPU's existing work.

### Continue on the remaining 68,287 training images

The user-approved continuation uses COCO training images at sorted positions
50,000–118,286: **68,287 fresh images**, disjoint from the original 50,000 and
the 5,000 held-out validation images. Together the two training pools contain
all 118,287 COCO-2017 training images. This is the available unique-image count;
it is not described as exactly 70,000 images.

```bash
.venv/bin/python scripts/launch_continuation.py --dry-run
# Launch once; refuses a duplicate active queue.
.venv/bin/python scripts/launch_continuation.py
cat runs/sd15/fresh68287/launch_status.json
tail -f runs/sd15/fresh68287/prepare.log
tail -f runs/sd15/fresh68287/gpu0.log
```

The detached supervisor downloads the fresh photos now and waits for each
existing GPU queue to finish before continuing its assigned methods. It uses
the same GPU assignments as below, including all three edge ablations, with
`swiftbrush_lora` replacing the retired `swiftbrush_vsd` run. It preserves the
original checkpoints and creates new runs under `runs/sd15/fresh68287`.

Every continuation adds **18,000 generator updates at one step / one NFE**,
effective batch **16**, seed **42**, and student **LoRA rank 32**. These are
updates, not an image count: 288,000 training draws traverse shuffled complete
epochs of the fresh pool. Student, EMA, G/F optimizer moments, scorer weights,
and RNG state carry over from the completed parent. AdamW methods start a new
warmup/cosine schedule; SwiftBrush retains its constant source-recipe rates and
EMA age. The progressive methods continue only the final 2-to-1 objective with
their frozen two-step teachers. Consistency retains its teacher grid; DMD uses
new 5,000-pair regression data from the fresh pool. InstaFlow continues only
endpoint distillation, using fresh couplings for all 68,287 prompts from its
parent's frozen reflow teacher; it does not repeat reflow optimization or the
raw-SD teacher-pair pass.

Each finished continuation evaluates FID, IS and CLIP on the same 5,000 held-out
pairs and noise seeds. New metrics go to `results/sd15/fresh68287`; SwiftBrush's
CFG-4.5 / FP32 recipe stays separate in
`results/sd15/fresh68287_swiftbrush_cfg4p5`. Parent hashes, zero training-image
overlap, phase updates, and cumulative **36,000** generator updates are recorded
in checkpoint metadata and result JSONs. Old and continued cohorts cannot be
silently combined by the report importer. All 5,000 generated images are used
for metrics; after the result is saved, only the first eight PNGs and the sample
grid are retained for this new cohort to limit disk use. They can be regenerated
from the saved checkpoint and evaluation seed.

The queue refreshes the qualitative figures and compiles the report after all
continuations finish. To rebuild after collecting teammate result JSONs:

```bash
.venv/bin/python scripts/build_report.py --profile sd15 \
  --results-dir results/sd15/fresh68287 --require-results
```

Closing VS Code leaves the detached queue running while the server stays on.
To recover a stopped supervisor, use `--supervise` with the directory recorded
in `artifacts/launches/latest_sd15_continuation.txt`; completed artifacts are
verified and unfinished runs resume from their new `last.pt` checkpoints.

### Direct one-step DMD proposal and controlled ablations

`scripts/launch_dmd_diversity.py` implements a separate four-variant experiment:
grouped latent-MSE DMD, LPIPS-VGG regression only, diversity matching only, and
the full perceptual/diversity proposal. The additional term matches the
LPIPS-VGG distance between two student images for the same prompt to the
distance between the corresponding teacher images. Teacher, VAE and LPIPS
weights remain frozen; gradients reach both student images. This is a
DMD/RDD/CoDMD-inspired project extension, not a claim of a new divergence.

```bash
.venv/bin/python scripts/launch_dmd_diversity.py --dry-run
.venv/bin/python scripts/launch_dmd_diversity.py --gpu 3
cat runs/sd15/dmd_diversity_study/launch_status.json
tail -f runs/sd15/dmd_diversity_study/prepare.log
tail -f runs/sd15/dmd_diversity_study/dmd_perceptual_diversity_seed42/launch.log
```

The detached queue first creates one shared bank of **5,000 teacher images for
2,500 fresh training prompts, with two noise seeds per prompt**. All variants
wait for the original 18,000-update DMD parent to finish and pin its checkpoint
SHA-256 before training. Each adds **18,000 G/F updates**, rank-32 students,
CFG 7.5 teacher, effective batch 16 (microbatch 2 × accumulation 8), the same
68,287 fresh-caption pool, and **one-step / one-NFE inference**. The regression
stream also has 16 images per update, grouped into same-prompt pairs. The full
proposal runs first, followed by the three controls. The queue waits for at
least 26 GiB free GPU memory before each training job. Existing runs are kept.

Every variant receives the usual 5,000-image FID/IS/CLIP evaluation and a separate
diversity evaluation using **128 held-out prompts × four fixed noise seeds**.
Diversity uses LPIPS-Alex, which is not used in the training loss; the teacher
gets the same diversity evaluation as a reference. Larger perceptual distance
alone is not better generation, so quality and prompt alignment remain part
of the comparison. Shared-load GPU timings are labelled unsuitable for speed
comparisons. Teacher-pair generation and perceptual backpropagation costs are
recorded separately from matched generator updates.

Portable results go to `results/sd15/dmd_diversity_study`. After all four finish,
the launcher fills the comparison CSV, LaTeX table and fixed-order sample
figure, then compiles **`report/build/dmd_diversity.pdf`** from
`report/dmd_diversity.tex`. This dedicated experiment report uses the new
objective; the original edge-study report is kept as its own document.

```bash
.venv/bin/python scripts/build_dmd_report.py \
  --results-dir results/sd15/dmd_diversity_study
```

### Active 50,000-caption study

```bash
# Already running in this workspace; this command starts or resumes it on a new launch.
.venv/bin/python scripts/launch_scale50k.py
cat runs/sd15/scale50k/launch_status.json
tail -f runs/sd15/scale50k/download.log
tail -f runs/sd15/scale50k/gpu2.log
tail -f runs/sd15/scale50k/gpu3.log
```

`50k` means **50,000 COCO training image–caption pairs**, separate from all 5,000
validation pairs. Every trained variant receives **18,000 generator updates**,
effective batch **16** (microbatch 1 × accumulation 16), and final **1-step / 1-NFE**
generation. PD has 3,000 updates at each of six stages; VSD/CD/DMD each have
18,000 updates; InstaFlow splits 9,000 reflow + 9,000 endpoint-distillation updates.
Scorer updates and teacher-pair generation remain additional disclosed costs.
The initial study uses seed 42; it is a larger development run, not three-seed
final evidence. `--seeds 42 43 44` launches a three-seed study when required.

| GPU | Assigned methods |
| --- | --- |
| 0 | PD, constant edge, edge weight 0.03, teacher evaluation |
| 1 | Proposed, edge weight 0.30 |
| 2 | SwiftBrush-inspired VSD, consistency distillation |
| 3 | DMD-inspired, InstaFlow-inspired |

The active queue has been updated at the user's request: GPU 2 now runs
`configs/sd15_swiftbrush_lora.yaml`, then consistency distillation. The student
stays **rank-32 LoRA**; Adam G/F rates are **1e-6/1e-3**, the scorer uses rank-64
q/v adapters with alpha 108, and sample reuse, constant rates, and dynamic EMA
follow the SwiftBrush source recipe. Its official-example CFG **4.5** and FP32
setting differ from the shared study, so its outputs are kept separately:

```bash
cat runs/sd15/swiftbrush_lora_seed42/launch_status.json
tail -f runs/sd15/swiftbrush_lora_seed42/train.jsonl
tail -f runs/sd15/scale50k/gpu2.log
```

Final JSONs go to `results/sd15/swiftbrush_lora`. The superseded LoRA VSD run
remains in `runs/sd15/scale50k/swiftbrush_vsd_seed42`, with an interruption
record and its last checkpoint. GPU 0/1/3 training processes were preserved
under a replacement queue controller; the shared `launch_status.json` follows
the new controller. The optional full-generator recipe is not running.

InstaFlow has also been launched **alongside DMD on GPU 3**, rather than waiting
for DMD to finish. This makes five active experiment processes across four GPUs.
The extra worker first generates 50,000 teacher noise/latent pairs, then runs
the same 18,000-update, rank-32 LoRA, two-phase flow recipe and 5,000-image final
evaluation. GPU 3's jobs share compute, so their completion estimates can change.

```bash
# Already active; a second launcher is rejected while this worker is running:
.venv/bin/python scripts/launch_instaflow.py --gpu 3
cat runs/sd15/scale50k/instaflow_launch_status.json
tail -f runs/sd15/scale50k/instaflow_seed42/launch.log
```

`parallel_workers.json` lists independently launched workers in the shared run
folder. The existing queue retains its InstaFlow entry, which waits on a
per-experiment lock and reuses the finished training/evaluation rather than
duplicating it. The regular queue produces its qualitative panels and report.

Saved 8-step PD and proposed checkpoints can be compared before one-step
training finishes:

```bash
.venv/bin/python scripts/evaluate_checkpoint_pair.py --steps 8 --gpu 3
cat runs/sd15/scale50k/step8_eval_status.json
tail -f runs/sd15/scale50k/step8_baseline_eval.log
tail -f runs/sd15/scale50k/step8_proposed_eval.log
```

This evaluates the immutable `stage_8` exports, each trained for 9,000 G updates,
on the same 5,000 COCO validation captions/references, evaluation seed 12345,
8 steps/8 NFE, and rank-32 adapters. The matching checks reject unequal training
budgets, grids, teachers, or prompt pools. Existing training keeps running.
The detached evaluator saves portable result JSONs and `comparison.csv` in
`results/sd15/scale50k_8steps`; shared-load GPU timings are labelled accordingly.

The report builder automatically imports this sibling checkpoint folder as a
separate intermediate-results paragraph when both evaluations are complete.
It validates shared data/teacher settings against the primary results and
keeps the one-step tables under their original protocol. Use
`--checkpoint-results-dir` to select a different intermediate cohort explicitly.

Image-free VSD and DMD can start before image downloading finishes. Image-based
jobs wait for the verified cache. DMD uses 5,000 cached teacher pairs; InstaFlow
uses one pair for each of all 50,000 training captions in each phase. Its larger
cache and two generation passes are part of training cost. The 50k InstaFlow
variant converts pretrained epsilon predictions to velocity with normalized
straight-path inputs and nearest VP SNR. The legacy pilot's direct-velocity
adapter remains separately reproducible. Both are adaptations of InstaFlow.
A real small-set learning diagnostic gates the larger InstaFlow job; diagnostic
results never enter paper metric tables.

Each worker trains/evaluates distinct methods and writes to
`results/sd15/scale50k/`. The detached supervisor waits for all workers, generates
their qualitative JSONs, and compiles the report. Closing VS Code leaves it running.
The full experiment will take substantially longer than the short pilot.
Completed runs are reused; interrupted runs resume their own checkpoints.
The launcher refuses to duplicate an active study.

```bash
# Run one method yourself with identical study settings:
bash scripts/run_experiments.sh --preset scale50k --methods proposed --gpu 1 --resume
# Rebuild only after adding results; --study-config keeps 50k budgets in the prose:
bash scripts/build_report.sh --results-dir results/sd15/scale50k \
  --study-config configs/sd15_50k.yaml --require-results
```

The older **matched one-step pilot** remains available for quick checks:

```bash
bash scripts/run_experiments.sh --preset fair_pilot --ablations --gpu 3 --resume
tail -f runs/sd15/fair_pilot/launch.log
```

Every trained method generates with **1 step / 1 NFE** and receives **600 generator
updates**, effective batch 4, training seed 42, shared SD 1.5 weights, rank-32
LoRA, identical evaluation prompts/noise, and 128 generated/reference images.
PD uses 100 updates at each of its six stages down to one step; VSD, consistency,
and DMD use 600 updates in their one-step stage. InstaFlow uses 300 reflow + 300
endpoint-distillation updates. DMD/VSD scorer work, DMD regression, and cached
pair generation remain extra costs, recorded separately. Teacher runs at 64 and
1 steps are explicitly separate references (128 and 2 NFE with CFG).

The original `pilot` stops PD at four steps and is preliminary; do not use it
to claim an equal-step comparison with one-step methods. Its results remain
separate. Matched results go to **`results/sd15/fair_pilot/`**. After training,
the background queue evaluates all methods and rebuilds the report automatically.
This is an exploratory one-seed, small-sample study; the full sweep supplies
three seeds and larger evaluations.

### Controlled ablations

`--ablations` adds `edge_weight_low` (0.03) and `edge_weight_high` (0.30) using
the same progressive schedule, data, rank, seed, and updates as the proposed loss.
Together with existing controls, the study contains:

| Variant | Edge weight | Gate |
| --- | --- | --- |
| Baseline PD | 0 | Off |
| Constant edge | 0.1 | Constant |
| Noise-aware edge | 0.03, 0.1, 0.3 | Signal level, alpha² |

The automatic ablation table uses the lowest shared inference step count and
the intersection of completed training seeds across all five loss settings.
Unmatched outcomes remain in the manifest/CSV, without entering the controlled
table. Weight sweeps stay separate from the main method-comparison table.

```bash
# Add the same ablations to the three-seed full study:
bash scripts/run_experiments.sh --preset full --ablations --gpu 3 --resume
# Build from matched pilot JSONs, including files returned by friends:
bash scripts/build_report.sh --results-dir results/sd15/fair_pilot --require-results
```

| Setting | Default full SD setup |
| --- | --- |
| Teacher | Pinned SD 1.5, ~860M U-Net parameters |
| Output / diffusion state | 512×512 RGB / 4×64×64 VAE latents |
| Student / auxiliary scorer | Rank-32 attention LoRA adapters |
| Curriculum | 64 → 32 → 16 → 8 → 4 → 2 → 1 |
| Methods | `baseline`, `proposed`, `edge_constant`, `swiftbrush_vsd`, `consistency`, `dmd`, `instaflow` |
| Training seeds | 42, 43, 44 |
| Generator updates | PD: 500/stage; VSD/CD/DMD: 3,000 each; InstaFlow: 1,500 reflow + 1,500 distillation |
| Extra scorer updates | VSD and DMD: 3,000 per run |
| DMD regression pairs | 256 per seed, 64-step guided teacher; 32,768 extra teacher NFE |
| InstaFlow pairs | 256 per seed; 32,768 guided teacher NFE + 16,384 reflow teacher NFE |
| Effective batch | 4 (microbatch 1 × accumulation 4) |
| Training data | First 5,000 COCO train IDs; lowest-ID caption per image |
| Quality evaluation | First 1,000 validation image–caption pairs |
| Metrics | FID, IS, CLIP alignment, latency, peak memory |
| Guidance | Teacher CFG 7.5; distilled students CFG 1 |
| Evaluation noise seed | 12345, shared across methods and captions |

The full sweep has 21 training jobs, **63,000 generator updates**, **18,000 extra
scorer updates**, and 73 quality evaluations, plus DMD/InstaFlow pair generation.
Adding `--ablations` gives 27 training jobs, 81,000 generator updates and 109 evaluations.
Estimate time and disk from a pilot;
this is a substantial experiment. Jobs run sequentially and do not reserve GPUs.
The 1,000-pair FID is a fixed-subset metric, not the COCO-2014 benchmark used in
other papers. CLIP alignment is mean `100 * max(cosine(image, caption), 0)` using
pinned CLIP ViT-B/32. A guided teacher step evaluates two denoiser branches even
when batched: 64 teacher steps cost 128 NFE; a one-step student costs 1 NFE.
Latency includes prompt encoding, denoising, and VAE decoding, excluding weight
loading, initial noise allocation, disk I/O, and metrics.

Run individual methods or direct configs:

```bash
bash scripts/run_experiments.sh --preset full --methods proposed --seeds 42 --gpu 3 --resume
CUDA_VISIBLE_DEVICES=3 fsd train --config configs/sd15_proposed.yaml
CUDA_VISIBLE_DEVICES=3 fsd train --config configs/sd15_proposed.yaml --resume
CUDA_VISIBLE_DEVICES=3 fsd train --config configs/sd15_vsd.yaml
CUDA_VISIBLE_DEVICES=3 fsd train --config configs/consistency.yaml
CUDA_VISIBLE_DEVICES=3 fsd train --config configs/dmd.yaml
CUDA_VISIBLE_DEVICES=2 fsd train --config configs/instaflow.yaml
CUDA_VISIBLE_DEVICES=3 fsd evaluate --config configs/sd15_proposed.yaml \
  --checkpoint runs/sd15/proposed_seed42/stage_4 --result-dir results/my_sd_run
```

Configs: `sd15.yaml` (baseline), `sd15_proposed.yaml`, `sd15_edge_constant.yaml`,
`sd15_vsd.yaml`, `consistency.yaml`, `dmd.yaml`, and `instaflow.yaml`.
Inspect with `fsd config --config configs/sd15_proposed.yaml`.
Strict overrides use `--set data.batch_size=1 --set train.grad_accumulation=4`.
Keep rank, effective batch, generator budgets, guidance, and evaluation matched.
`--num-samples` changes generated count; change reference count in YAML if needed.

Checkpoints include adapters, EMA, optimizers, RNG state, and reproducible data
order. VSD and DMD also save their scorers and optimizers. DMD pairs use a
checksum-validated safetensors cache. Interrupted jobs resume from
`last.pt`; completed jobs are reused. Experiment-setting changes are rejected
on resume, while display and W&B settings may change. Stage exports include
`adapter.safetensors`, standard Diffusers `pytorch_lora_weights.safetensors`, and
the exact inference grid and prediction type. Consistency also needs its saved
boundary wrapper; InstaFlow needs its Euler velocity sampler. Use `fsd sample`
to restore these prediction types. Sample with the saved sampler and grid.

The old pilot's queue and logs are retained as execution evidence. That queue is
stopped; the active 50k study uses the separate paths documented above:

```bash
tail -f runs/sd15/fair_pilot/launch.log
cat runs/sd15/fair_pilot/launch_status.json
tail -f runs/sd15/fair_pilot/gpu0.log
tail -f runs/sd15/fair_pilot/gpu1.log
tail -f runs/sd15/fair_pilot/instaflow_gpu2.log
```

Its queue is recorded in `runs/sd15/fair_pilot/launch_queue.json`. The job is detached
from the terminal when launched. Do not launch a second copy into active output folders.

### Progress and automatic W&B

Training uses tqdm per stage with loss, learning rate, rate, and ETA. Downloads,
generation, and CLIP evaluation also show progress. Disable with `--no-progress`
in the launcher or `--set progress_bar=false` with `fsd`.

W&B defaults to `auto`: an existing login or `WANDB_API_KEY` enables online
tracking; otherwise it logs offline without prompting. Auto initialization
failures fall back to offline. Configs, losses, gradients, update counts, time,
previews, and completion status are logged. Evaluation metrics go into JSON.

```bash
.venv/bin/wandb login  # optional one-time login
bash scripts/run_experiments.sh --preset full --gpu 3 --resume \
  --wandb-project few-step-diffusion --wandb-entity YOUR_TEAM
bash scripts/run_experiments.sh --preset pilot --gpu 3 --resume --wandb-mode offline
# --wandb-mode disabled switches tracking off.
```

`WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_GROUP`, and `WANDB_NAME`
override config. Keep keys in the environment or credential store. Upload offline
runs with `wandb sync PATH_TO_OFFLINE_RUN`. Online resume reuses a run ID; offline
resume creates a linked session because the SDK cannot resume offline history.

### Methods and attribution

PD uses the inverse of two teacher DDIM transitions. Initial teacher guidance is
baked into the conditional student; later teachers are EMA students. The base
loss is clipped-SNR latent reconstruction. The proposed edge term compares
horizontal/vertical **decoded RGB** gradients with gate `alpha²`; the constant
ablation removes that gate. Frozen VAE weights still pass gradients to adapters.
This adapts [progressive distillation](https://arxiv.org/abs/2202.00512)'s
parameterization and schedule, rather than reproducing it exactly.

VSD uses captions and generated latents, without reading training images. Real
and fake scorers use the same CFG. The generator receives the stopped
`sigma² * (epsilon_real - epsilon_fake)` direction; the scorer learns
`alpha² * MSE(epsilon_fake, noise)` with 10% text dropout on detached generations.
This follows the [official SwiftBrush objective](https://github.com/VinAIResearch/SwiftBrush),
but trains the generator with LoRA instead of full U-Net weights. Call it
**SwiftBrush-inspired VSD**, not a reproduction or its published FID. Matching
G updates does not equalize total compute. The surrogate is not a calibrated KL
value. Identical initial scorers yield zero first G gradient until the fake
scorer changes.

An [implementation audit](docs/SWIFTBRUSH_SOURCE_AUDIT.md) lists the additional
recipe differences: optimizer and learning rates, scorer adapters, sample
reuse, EMA, and terminal parameterization. A separate full-generator port is
available in `configs/sd15_swiftbrush_source.yaml`:

```bash
.venv/bin/python scripts/audit_swiftbrush.py
.venv/bin/python scripts/run_swiftbrush_source.py --gpu 3 --dry-run
.venv/bin/python scripts/run_swiftbrush_source.py --gpu 3 --background
```

This source-aligned recipe still uses SD1.5/COCO instead of the original
teacher and prompt corpus. It writes to a separate run/result folder and is
not part of the current rank-matched LoRA study. Existing jobs continue with
their saved configurations. The audit and short functional tests do not
establish image quality or reproduce the published score.

For the user's selected **rank-32 student with source training settings**, use:

```bash
.venv/bin/python scripts/run_swiftbrush_source.py \
  --config configs/sd15_swiftbrush_lora.yaml --gpu 2 --background --resume
```

The active queue already supervises this run; do not launch a duplicate. This
variant keeps the project's q/k/v/out rank-32 student and independently sets
the official rank-64 q/v scorer. Its CFG 4.5/FP32 results stay separate from the
CFG 7.5/bf16 controlled comparison. Rank-matching the student is an explicit
adaptation; original full-student hyperparameters are not evidence of equivalent
quality with LoRA.

Consistency distillation matches a boundary-conditioned endpoint prediction to
an EMA student's endpoint after one guided teacher DDIM step. It trains on
adjacent levels of the shared grid and uses latent L2. The clean boundary is an
exact identity; one-step inference uses this mapping rather than an ordinary
DDIM step. This is a VP/LoRA adaptation of
[consistency distillation](https://arxiv.org/abs/2303.01469), not the original
EDM/LPIPS recipe or a pretrained LCM checkpoint.

[DMD](https://arxiv.org/abs/2311.18828) uses a per-sample normalized real/fake
denoising direction plus a separate regression stream of fixed noise–teacher
pairs. Our version uses LoRA, latent L2 with weight 0.25, CFG 7.5 for the real
score and CFG 1 for the fake score, and unweighted fake epsilon denoising.
The paired teacher runs once during preprocessing and is cached. Generator
updates match the other one-step methods, but DMD has extra regression forwards,
scorer updates, and pair-generation cost. These are reported, including pair
teacher NFE and preprocessing within training time. Label this **DMD-inspired**;
it is not a reproduction of DMD's full-model/LPIPS setup or DMD2's adversarial recipe.

The [DMD settings audit](docs/DMD_SETTINGS_AUDIT.md) compares the running
configuration with the paper's SD1.5 Appendix B.3 recipe. Learning rate,
guidance, regression loss, teacher pairs, prompt corpus, batch size, and
gradient clipping differ; weight 0.25 alone does not match the recipe. This
audit does not change the ongoing DMD run.

[InstaFlow](https://github.com/gnobitab/InstaFlow) first straightens a flow using
noise–teacher-output couplings, then distills a one-step model. Our
**InstaFlow-inspired** baseline uses the same rank-32 LoRA and total G update
budget as the other methods. Half the updates fit the uniform straight path
`x(u)=(1-u)*noise+u*teacher_latent`, with velocity target `teacher_latent-noise`.
Its frozen EMA generates a new coupling using 64 Euler steps. The remaining
updates fit that coupling's endpoint velocity; inference is `noise + velocity`
in exactly one U-Net call. Time labels use `round(999*(1-u))`, and each phase
uses 256 cached pairs (64 in pilots). This reduced-budget LoRA recipe differs
from the original full-model implementation. Frozen flow adapters, pair caches,
optimizers, and RNG state support exact resume across the phase boundary.
Pair NFE and training time include both coupling-generation passes.

Recent related work is cited in the paper: [WaDi (CVPR 2026)](https://arxiv.org/abs/2603.08258)
uses weight-direction adapters, and [Pixel MeanFlow (ICML 2026)](https://proceedings.mlr.press/v306/lu26t.html)
uses pixel-space generation. They are related work, with no imported benchmark
scores or claims of matched training results under our SD1.5/LoRA setting.

## Generate scenes

```bash
CUDA_VISIBLE_DEVICES=3 fsd sample --prompt "A photograph of a lighthouse at sunset."
CUDA_VISIBLE_DEVICES=3 fsd sample --config configs/sd15_proposed.yaml \
  --checkpoint runs/sd15/proposed_seed42/stage_4 \
  --prompt "A photograph of a mountain lake." --output artifacts/lake.png
.venv/bin/python scripts/make_teacher_preview.py --gpu 3
```

Repeat `--prompt` for multiple images. Teacher defaults to 64 steps; students use
the saved grid and CFG 1. Without prompts, eight fixed scenes are generated.
PNG and prompt sidecar are saved under `artifacts/`.

### Paper-style comparison, ablation, and prompt-control figures

```bash
.venv/bin/python scripts/generate_qualitative.py --gpu 3 --resume
bash scripts/build_report.sh --results-dir results/sd15/fair_pilot --require-results
```

The generator restores real one-step checkpoints and saves portable
`results/sd15/fair_pilot/qualitative/*.qualitative.json` files. Fixed prompts are
defined in `src/few_step_diffusion/qualitative.py` before generation:

- Comparison: methods across columns, prompts down rows, identical noise per row.
  All six prompts appear in a separate gallery; the main paper shows the first three.
- Visual ablation: five edge-loss settings, train-painting/tiger-city prompts,
  two shared noise seeds per prompt, and matched 1-step / 1-NFE inference.
- Prompt control: owl/raccoon/cat/panda across columns and book/latte/pizza down
  rows, all using the same noise. This measures compositional prompt edits;
  it is not continuous prompt-embedding interpolation.

Copy teammates' qualitative JSONs into the results folder alongside metric JSONs;
renaming and nested folders work. The builder checks prompt/noise protocols,
prediction budgets, and checkpoint hashes against matching quantitative records.
Qualitative files do not enter metric tables. Missing variants stay pending;
no synthetic, reference-paper, or cherry-picked images fill missing cells.
The automatic queue generates these images and rebuilds both
`report/build/main.pdf` (four-page limit) and `report/build/qualitative.pdf`
(optional separate gallery). Native PDF figure assets are in `report/generated/`.

## Share results and compile the paper

Use the same commit and final settings. Split the fixed job list:

```bash
# Friend A; friends B/C set shard-index to 1/2 on their own machines.
bash scripts/run_experiments.sh --preset full --shard-index 0 --shard-count 3 --gpu 0 --resume
```

Shard 0 evaluates the teacher. Keep shard count unchanged on resume. Explicit
`--methods` / `--seeds`, `--train-only`, and `--eval-only` also work.
Each friend returns JSONs from **`results/sd15/full/`**. Copy into that folder;
renamed files and nested friend folders are supported. JSONs include metrics,
protocol hashes, provenance, and eight PNG samples, so no model checkpoint or
separate image file is needed for report generation. Keep training logs as evidence.

```bash
mkdir -p results/sd15/full
bash scripts/build_report.sh --results-dir results/sd15/full --require-results
# Build the explicitly unfinished draft before results exist:
bash scripts/build_report.sh
```

Output: **`report/build/main.pdf`**. The builder validates JSONs, aggregates seeds,
fills quality (FID/IS/CLIP), latency, and cost tables, creates plots, a sample figure,
`summary.csv`, and a manifest, then compiles LaTeX/BibTeX and checks four pages.
SD main tables show one-step students and teacher references; all intermediate-step
measurements remain in the CSV, manifest, and exported plots.
Re-run after adding files; `--no-compile` generates inputs only. Table standard
deviations describe training-seed variation, not IS split variation.

The figure shows the first four consecutive 512px outputs with matched prompts
and noise. Default: seed 42, one step. Use `--sample-steps 4 --sample-seed 42` for
another stage. Before completed comparisons exist, genuine SD teacher outputs
from predefined prompts are explicitly labelled as a preview. No smoke samples
are inserted. Independent implementations can follow `schemas/result.schema.json`
and `examples/sd15.result.template.json`; replace all placeholders with measurements.
Null metrics, NaN, conflicting duplicates, mixed prompts/CFG/rank/sample counts,
grids, or generator budgets are rejected. Consistency's validated endpoint
sampler can differ from the other methods while sharing the same data protocol.
Keep pilot and final results separate.
Runtime tables group hardware and batch size separately.

Edit `report/main.tex` for authors, novelty review, interpretations, failures,
and conclusions. Automation fills measured evidence, without inventing results
or finishing scientific analysis. The paper remains an unfinished draft.

## Legacy profiles and Git

```bash
FSD_PROFILE=celebahq bash scripts/setup.sh
bash scripts/run_experiments.sh --profile cifar10 --preset pilot --gpu 3 --resume
bash scripts/run_experiments.sh --profile celebahq --preset pilot --gpu 3 --resume
bash scripts/build_report.sh --profile cifar10 --results-dir results/full
```

CIFAR keeps `results/full/`; CelebA-HQ uses `results/celebahq/full/`. Their original
RGB U-Nets and unconditional VSD remain supported. Do not merge with the SD paper.

Git ignores environments, data, weights, runs, imported results, W&B sessions,
secrets, and generated PDFs. Source, configs, tests, report source, supplied docs,
and preview assets can be committed normally. No remote has been created.

```bash
git add .
git commit -m "Set up Stable Diffusion experiments and CVPR report"
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Commit before final experiments so JSONs record the actual revision. Deliberately
version selected final results with `git add -f` after review; keep weights outside Git.
# OneStepDiffusion
