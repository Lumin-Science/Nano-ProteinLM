#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
output_root="${OUTPUT_ROOT:-outputs/qualify-300m}"
mkdir -p "$output_root"

uv sync --frozen --all-groups
uv run --frozen python scripts/check_environment.py \
  --require-gpus "${NPROC:-4}" \
  --output "$output_root/ENVIRONMENT.json"
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${NPROC:-4}" \
  -m nano_protein.train \
  --config "${CONFIG:-configs/esmc_300m.yaml}" \
  --data-root "${DATA_ROOT:?set DATA_ROOT to a prepared corpus}" \
  --output-root "$output_root" \
  --walltime-seconds "${WALLTIME_SECONDS:-60}"
