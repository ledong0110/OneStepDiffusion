import argparse
import json

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Diffusion training, distillation, and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "evaluate", "sample", "config"):
        p = sub.add_parser(command)
        p.add_argument("--config", default="configs/sd15.yaml")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        if command == "prepare":
            p.add_argument("--metrics", action="store_true", help="Also download Inception weights")
        elif command == "train":
            p.add_argument("--resume", action="store_true")
        elif command == "evaluate":
            p.add_argument("--checkpoint", default="teacher")
            p.add_argument("--steps", type=int)
            p.add_argument("--result-dir", default="results")
            p.add_argument("--skip-metrics", action="store_true", help="Smoke only; excluded from paper")
        elif command == "sample":
            p.add_argument("--checkpoint", default="teacher")
            p.add_argument("--steps", type=int)
            p.add_argument("--prompt", action="append", help="Repeat for multiple prompts")
            p.add_argument("--output", default="artifacts/sd15_samples.png")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    if args.command == "config":
        print(json.dumps(config, indent=2))
    elif args.command == "prepare":
        from .data import prepare
        prepare(config, args.metrics)
    elif args.command == "train":
        from .training import train
        train(config, args.resume)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(config, args.checkpoint, args.steps, args.result_dir, args.skip_metrics)
    elif args.command == "sample":
        from .sd_models import sample_sd
        sample_sd(config, args.checkpoint, args.steps, args.prompt, args.output)


if __name__ == "__main__":
    main()
