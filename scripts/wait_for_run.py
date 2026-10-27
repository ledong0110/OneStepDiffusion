#!/usr/bin/env python3
"""Join an independently detached worker before building its shared report."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", required=True, type=Path)
    args = parser.parse_args()
    print(f"Waiting for worker: {args.status}", flush=True)
    while True:
        if args.status.is_file():
            status = json.loads(args.status.read_text())
            if status["status"] == "complete":
                print("Worker completed successfully", flush=True)
                return
            if status["status"] == "failed":
                raise RuntimeError(f"Worker failed: {status.get('error')}")
        time.sleep(10)


if __name__ == "__main__":
    main()
