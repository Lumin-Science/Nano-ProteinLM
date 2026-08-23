#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

data_root="${DATA_ROOT:-$repo_root/data/processed/stage1-300m-production-v1}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs/stage1-300m-4xa100-4h}"
config="${CONFIG:-$repo_root/configs/esmc_300m_stage1_4xa100_4h.yaml}"
uv_bin="${UV_BIN:-uv}"
uv_cache_dir="${UV_CACHE_DIR:-$repo_root/.uv-cache}"

if [[ ! -f "$data_root/manifest.json" ]]; then
  DATA_ROOT="$data_root" bash runs/prepare_stage1_300m_corpus.sh
fi

mkdir -p "$output_root"
UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" sync --frozen
UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" run --frozen python scripts/check_environment.py \
  --require-gpus 4 \
  --output "$output_root/ENVIRONMENT.json"

UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" run --frozen python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  -m nano_protein.train \
  --config "$config" \
  --data-root "$data_root" \
  --output-root "$output_root"

DATA_ROOT="$data_root" \
OUTPUT_ROOT="$output_root" \
EVAL_PROFILE=full \
EVAL_CHECKPOINTS=checkpoint-final.pt \
CONTACT_CHAINS=20775 \
CONTACT_BOOTSTRAP=5000 \
CONTACT_SHARDS=3 \
PCORE_BOOTSTRAP=10000 \
PCORE_TASK_PARALLEL=6 \
PCORE_PROBE_THREADS=4 \
PCORE_BATCH_RESIDUES=32768 \
UV_BIN="$uv_bin" \
UV_CACHE_DIR="$uv_cache_dir" \
bash runs/evaluate_full_parallel.sh
