# Independent 36,000-update direction-alignment training

This experiment starts distillation from pretrained SD1.5 with **fresh rank-32
student and scorer adapters**, empty optimizer states, and a new EMA. It does
not load the 36k DMD checkpoint, the old proposal, or any previous RNG state.
“From scratch” here means independent distillation from the common pretrained
backbone, not retraining Stable Diffusion from random U-Net weights.

Two runs are scheduled, proposal first:

- `response_scratch`: DMD + LPIPS + angular response alignment from update 1.
- `dmd_lpips_scratch`: identical training without the response loss.

The control can also run concurrently on a selected GPU:

```bash
.venv/bin/python scripts/launch_independent_job.py launch --method dmd_lpips_scratch --gpu 1 --memory-cap-gib 13.5
.venv/bin/python scripts/launch_independent_job.py status --method dmd_lpips_scratch --gpu 1
```

This uses the same pinned training configuration and output folder. Its
two-update preflight must pass before full training resumes. The 13.5 GiB
allocation cap requires at least 14.5 GiB free at startup; it changes no loss,
batch size, precision, or update budget. `parallel.json` records worker and
child identities, and `make independent-status` includes the parallel job.
The shared run lock prevents the serial supervisor from duplicating this run.

Both use exactly 36,000 student updates and 36,000 scorer updates. The first
18,000 updates use the first 50,000 COCO training captions; the remaining
18,000 use the disjoint 68,287 captions. Validation images are held out.
Effective batch is 16 (two samples × eight accumulation steps), teacher CFG
7.5, student CFG 1, and generation uses one step / one NFE. Student learning
rate is 1e-4 with 100 warmup updates and cosine decay over the complete run;
the scorer uses 1e-4. EMA is .995. Optimizers are not reset at the phase boundary.
This differs from DMD2/f-distill's method-specific CFG and auxiliary schedule;
equal student-update budgets do not imply equal training compute.

The LPIPS weight is .25. The response coefficient is fixed at
`0.2984940099461313`, the earlier pilot's selected value. It is **not** claimed
to retain a 50% gradient ratio at fresh initialization; that calibration was
performed on the earlier trained checkpoint. No new validation-based tuning
or warm-start training is hidden in this run.

Each phase uses 5,000 frozen teacher outputs: two independent noises for each
of 2,500 training captions. The second phase can reuse the verified existing
teacher-only bank; the first phase needs a new grouped bank. Reusing frozen
teacher targets does not inherit any distilled student/scorer weights.
Preparing these targets performs zero student updates. The first bank takes
time because each target uses 64 guided teacher steps.

`initialization.json` records the actual fresh adapter identities, zero
optimizer-state entries and zero prior student updates. Resume checkpoints
are bound to this record. The control checks that its initial student and
scorer identities match the proposal. Old trained checkpoints cannot be
passed as valid independent resumes.

```bash
make independent-response
make independent-status
tail -f runs/sd15/independent36k/response_scratch_seed42/prepare.log
tail -f runs/sd15/independent36k/response_scratch_seed42/train.log
```

The detached supervisor uses GPUs 0, 1 or 2 as memory becomes available. Target
preparation requires 9 GiB free GPU memory and 2 GiB disk; training requires
16 GiB free GPU memory and 6 GiB disk. It does not terminate other jobs or
remove old runs. It survives closing VS Code/SSH while the server stays on;
after a server reboot, rerun the launch command to resume.

Two real student updates form a GPU preflight, after which the complete state
is saved and resumed to 36,000. Those updates count toward the budget. Every
1,000 updates, it saves resumable state, an EMA adapter and eight fixed
validation samples. These are **qualitative checkpoint evaluations**, not
full-set checkpoint FID. Progress uses tqdm and the existing automatic W&B
online/offline configuration.

After training, FID, IS and CLIP are evaluated on all 5,000 held-out captions
and images. Completed independent results enter an appendix table when the
report rebuilds. The old 36,500-update proposal is retained under its actual
identity until a subsequent explicit report selection; its scores are not
relabelled as fresh training results.

Run folders: `runs/sd15/independent36k/`.
Final results: `results/sd15/independent36k/`.
The new teacher-pair cache is shared by the two independent runs.

The implementation is pinned when scheduled. Changes to training code after
launch require a reviewed new plan; the supervisor fails rather than silently
changing the experiment between preparation and training.
