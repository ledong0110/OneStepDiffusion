#!/usr/bin/env python3
"""Generate matched comparison, ablation, and compositional prompt-control images."""

import argparse
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", default="runs/sd15/fair_pilot")
    p.add_argument("--output-dir", default="results/sd15/fair_pilot/qualitative")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=12345)
    p.add_argument("--gpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-teacher", action="store_true")
    args = p.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from few_step_diffusion.qualitative import CORE_METHODS, EDGE_METHODS, generate_qualitative
    generate_qualitative(args.run_root, args.output_dir, args.methods or CORE_METHODS+EDGE_METHODS,
                         args.seed, args.eval_seed, args.resume, args.skip_teacher)


if __name__ == "__main__":
    main()
