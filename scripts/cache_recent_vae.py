#!/usr/bin/env python3
"""Cache deterministic frozen-VAE statistics, then resume the two saved runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def build_cache(args):
    import numpy as np
    import torch
    from diffusers import AutoencoderKL
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    from few_step_diffusion.config import load_config
    from few_step_diffusion.runtime import atomic_json, experiment_lock, sha256
    from few_step_diffusion.sd_data import pixels
    from few_step_diffusion.vae_cache import cache_protocol

    config = load_config(ROOT/"configs/sd15_dmd2_36k.yaml")
    protocol, records = cache_protocol(config)
    folder = args.cache_dir
    folder.mkdir(parents=True, exist_ok=True)
    status = ROOT/"runs/sd15/recent36k/posterior_cache_status.json"
    with experiment_lock(folder):
        metadata_path = folder/"metadata.json"
        existing = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        if existing and existing.get("protocol") != protocol:
            raise ValueError("Existing cache uses another frozen model or preprocessing protocol")
        if existing and existing.get("status") == "complete":
            if sha256(folder/"posterior.npy") != existing["posterior_sha256"]:
                raise ValueError("Completed cache checksum mismatch")
            atomic_json(status, {"status": "complete", "count": len(records), "cache_dir": str(folder)})
            return
        mode = "r+" if (folder/"posterior.npy").exists() else "w+"
        stats = np.lib.format.open_memmap(folder/"posterior.npy", mode=mode,
                                         dtype=np.float32, shape=tuple(protocol["shape"]))
        valid = np.lib.format.open_memmap(folder/"valid.npy", mode=mode,
                                         dtype=np.bool_, shape=(len(records),))
        if mode == "w+":
            valid[:] = False
            np.save(folder/"image_ids.npy", np.array([r["image_id"] for r in records], dtype=np.int64))
        atomic_json(metadata_path, {"status": "building", "protocol": protocol})
        todo = np.flatnonzero(~valid).tolist()

        class Images(Dataset):
            def __len__(self):
                return len(todo)

            def __getitem__(self, index):
                actual = todo[index]
                return actual, pixels(config, records[actual])

        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda:0")
        free, total = torch.cuda.mem_get_info(device)
        if free < (args.memory_gib+1)*2**30:
            raise RuntimeError("Not enough GPU memory for the bounded posterior-cache builder")
        torch.cuda.set_per_process_memory_fraction(args.memory_gib*2**30/total, 0)
        vae = AutoencoderKL.from_pretrained(config["model"]["source"], subfolder="vae",
            revision=config["model"]["revision"], use_safetensors=True,
            torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
        loader = DataLoader(Images(), batch_size=2, num_workers=4, pin_memory=True,
                            shuffle=False, persistent_workers=bool(todo))
        started = time.monotonic()
        parity = None
        count = int(valid.sum())
        with torch.inference_mode(), tqdm(total=len(records), initial=count,
                desc="Cache frozen VAE posteriors", unit="image", dynamic_ncols=True) as bar:
            for indices, images in loader:
                pixels_gpu = images.to(device, non_blocking=True).float()/127.5-1
                posterior = vae.encode(pixels_gpu).latent_dist
                values = torch.cat((posterior.mean, posterior.std), dim=1).cpu().numpy()
                if not np.isfinite(values).all() or (values[:, 4:] < 0).any():
                    raise FloatingPointError("Invalid frozen posterior statistics")
                stats[indices.numpy()] = values
                valid[indices.numpy()] = True
                if parity is None:
                    restored = torch.from_numpy(np.array(stats[indices.numpy()], copy=True)).to(device)
                    torch.testing.assert_close(restored[:, :4], posterior.mean, rtol=0, atol=0)
                    torch.testing.assert_close(restored[:, 4:], posterior.std, rtol=0, atol=0)
                    state = torch.cuda.get_rng_state(device)
                    expected = posterior.sample()*vae.config.scaling_factor
                    expected_rng = torch.cuda.get_rng_state(device)
                    torch.cuda.set_rng_state(state, device)
                    actual = (restored[:, :4]+restored[:, 4:]*torch.randn_like(restored[:, :4]))*vae.config.scaling_factor
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    assert torch.equal(torch.cuda.get_rng_state(device), expected_rng)
                    parity = {"posterior_mean_std": "exact FP32 round trip", "sample_and_rng": "exact",
                              "posterior_noise_cached": False}
                bar.update(len(indices))
                count += len(indices)
                if count % 256 == 0 or count == len(records):
                    stats.flush()
                    valid.flush()
                    atomic_json(status, {"status": "building", "count": count, "total": len(records),
                        "cache_dir": str(folder), "elapsed_seconds": time.monotonic()-started,
                        "parity": parity})
        if not valid.all():
            raise ValueError("Posterior cache is incomplete")
        stats.flush()
        valid.flush()
        atomic_json(metadata_path, {"status": "complete", "protocol": protocol,
            "posterior_sha256": sha256(folder/"posterior.npy"), "parity": parity,
            "posterior_noise_cached": False, "vae_scaling_factor": vae.config.scaling_factor})
        atomic_json(status, {"status": "complete", "count": len(records), "total": len(records),
                            "cache_dir": str(folder), "parity": parity})


def resume_jobs(args):
    from few_step_diffusion.runtime import atomic_json
    from launch_scale50k import process_matches
    from launch_dmd_conflict import started_process_record
    import launch_recent_baselines as queue
    plan = queue.read(queue.PLAN)
    queue.verify_plan(plan)
    for job in plan["jobs"]:
        output = ROOT/job["run_dir"]
        execution = queue.read(output/"execution.json")
        execution.update(posterior_cache=str(args.cache_dir), helper_gpu=0, training_gpu=3, resume_step=9000)
        atomic_json(output/"execution.json", execution)
        while True:
            state = queue.read(output/"queue.json")
            if queue.live_worker(state):
                break
            free = queue.gpu_free()
            if free.get(3, 0) >= execution["training_memory_gib"]+2 and free.get(0, 0) >= 3:
                argv = [sys.executable, "scripts/launch_recent_baselines.py", "worker",
                        "--method", job["method"], "--gpu", "3"]
                with (output/"launch.log").open("ab", buffering=0) as log:
                    child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        env=dict(os.environ, PYTHONUNBUFFERED="1"))
                record = started_process_record(child, argv)
                if record is None:
                    raise RuntimeError("Cached training worker failed at startup")
                atomic_json(output/"cache_resume.json", {"worker": record, "execution": execution,
                    "resumed_from_step": 9000, "requested_at": queue.now()})
                print(f"Resumed {job['method']} from saved checkpoint: PID {child.pid}", flush=True)
                # Give this process time to initialize its actual allocation before
                # admitting the second full scorer. No unrelated process is touched.
                deadline = time.monotonic()+300
                while time.monotonic() < deadline:
                    current = queue.read(output/"queue.json")
                    if current.get("worker") == record and current.get("phase") == "train" and current.get("child"):
                        if process_matches(current["child"]):
                            time.sleep(45)
                            break
                    if not process_matches(record) or current.get("status") == "failed":
                        raise RuntimeError("Cached training failed; saved checkpoint is preserved")
                    time.sleep(1)
                break
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--memory-gib", type=float, default=4)
    parser.add_argument("--resume-recent", action="store_true")
    args = parser.parse_args()
    args.cache_dir = args.cache_dir.resolve()
    os.chdir(ROOT)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    build_cache(args)
    if args.resume_recent:
        resume_jobs(args)


if __name__ == "__main__":
    main()
