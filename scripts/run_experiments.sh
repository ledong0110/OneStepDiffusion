#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/python ]]; then
  bash scripts/setup.sh
fi
exec .venv/bin/python scripts/run_experiments.py "$@"
