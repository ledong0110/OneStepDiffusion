#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
FSD_PYTHON="${FSD_PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
  "$FSD_PYTHON" -m venv --without-pip .venv
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  "$FSD_PYTHON" -m pip --python .venv/bin/python install --upgrade pip
fi
# Set FSD_TORCH_INDEX=https://download.pytorch.org/whl/cpu on CPU-only hosts.
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url "${FSD_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
FSD_LOCK_DEFAULT=0
if [[ "$(.venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')" == 3.12 ]]; then
  FSD_LOCK_DEFAULT=1
fi
if [[ -f requirements-lock.txt && "${FSD_USE_LOCK:-$FSD_LOCK_DEFAULT}" == 1 ]]; then
  .venv/bin/python -m pip install -r requirements-lock.txt
fi
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip check
mkdir -p results/sd15/full results/sd15/pilot results/full results/pilot
# Downloads are cached and safe to repeat. No GPU allocation is needed here.
case "${FSD_PROFILE:-sd15}" in
  sd15) FSD_CONFIG=configs/sd15.yaml ;;
  celebahq) FSD_CONFIG=configs/celebahq.yaml ;;
  cifar10) FSD_CONFIG=configs/baseline.yaml ;;
  *) printf 'FSD_PROFILE must be sd15, celebahq, or cifar10\n' >&2; exit 1 ;;
esac
.venv/bin/fsd prepare --config "$FSD_CONFIG" --metrics
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b main
fi
printf '\nSetup complete. Activate with: source .venv/bin/activate\n'
printf 'Preview the sweep: python scripts/run_experiments.py --preset pilot --dry-run\n'
