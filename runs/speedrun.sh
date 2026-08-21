#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cluster_root="${CLUSTER_ROOT:-/home/muchenli/workspace/esmc-open-step9-clusters-v1/clusters}"
data_root="${DATA_ROOT:-$repo_root/data/processed/pilot-v1}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs/speedrun-300m}"
config="${CONFIG:-$repo_root/configs/esmc_300m.yaml}"
walltime_seconds="${WALLTIME_SECONDS:-1800}"

mkdir -p "$output_root"
uv sync --frozen
uv run --frozen python scripts/check_environment.py \
  --require-gpus 4 \
  --output "$output_root/ENVIRONMENT.json"

if [[ ! -f "$data_root/manifest.json" ]]; then
  uv run --frozen python scripts/prepare_data.py \
    --cluster-root "$cluster_root" \
    --output-root "$data_root" \
    --pcore-index "${PCORE_ROOT:-/home/muchenli/datasets/pcore/v0.1}/index.jsonl" \
    --contact-manifest "${CONTACT_ROOT:-/home/muchenli/datasets/esmc-paper-contact-v1}/CONTACT_MANIFEST.jsonl"
fi

uv run --frozen python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  -m nano_protein.train \
  --config "$config" \
  --data-root "$data_root" \
  --output-root "$output_root" \
  --walltime-seconds "$walltime_seconds"

DATA_ROOT="$data_root" OUTPUT_ROOT="$output_root" bash runs/evaluate.sh
