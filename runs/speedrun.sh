#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

env_file="${SPEEDRUN_ENV_FILE:-$repo_root/.env}"
if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi

uv_bin="${UV_BIN:-uv}"
if ! command -v "$uv_bin" >/dev/null 2>&1; then
  echo "runs/speedrun.sh requires uv >=0.11.31,<0.12 on PATH (or set UV_BIN)." >&2
  exit 1
fi

artifact_root="${ARTIFACT_ROOT:-$repo_root/.exps}"
data_repo_id="${DATA_REPO_ID:-LuminScience/LuminBench-Nano-ESMC}"
data_revision="${DATA_REVISION:-bd38448d50d8f426d7b9bd4410b53159ea001259}"
training_samples="${TRAINING_SAMPLES:-5376000}"
download_workers="${DOWNLOAD_WORKERS:-8}"
num_gpus="${NUM_GPUS:-4}"
walltime_seconds="${WALLTIME_SECONDS:-14400}"
run_name="${RUN_NAME:-esmc-300m-original}"
config="${CONFIG:-$repo_root/configs/esmc-300m-original.yaml}"
data_cache_root="${DATA_CACHE_ROOT:-$artifact_root/cache/huggingface-dataset}"
data_root="${DATA_ROOT:-$artifact_root/data/training-samples-$training_samples}"
output_root="${OUTPUT_ROOT:-$artifact_root/runs/$run_name}"

for value_name in training_samples download_workers num_gpus walltime_seconds; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$value_name must be a positive integer; found '$value'." >&2
    exit 1
  fi
done
if [[ ! -f "$config" ]]; then
  echo "Training config does not exist: $config" >&2
  exit 1
fi
if [[ -e "$output_root" ]]; then
  echo "Refusing to overwrite an existing run: $output_root" >&2
  echo "Set RUN_NAME or OUTPUT_ROOT to a new location." >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$artifact_root/cache/huggingface-client}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$artifact_root/cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$artifact_root/venv}"

mkdir -p "$artifact_root" "$data_cache_root"

echo "[1/4] Creating the locked Python/CUDA environment with uv"
"$uv_bin" sync --frozen --no-dev

if [[ -f "$data_root/manifest.json" && -f "$data_root/CORPUS_VERIFICATION.json" ]]; then
  echo "[2/4] Reusing verified prepared data: $data_root"
elif [[ -e "$data_root" ]]; then
  echo "Prepared data directory is incomplete; refusing to overwrite it: $data_root" >&2
  echo "Move it aside or set DATA_ROOT to a new location." >&2
  exit 1
else
  echo "[2/4] Downloading, hashing, and materializing the verified Hub shard prefix"
  "$uv_bin" run --frozen --no-dev python scripts/download_data.py \
    --repo-id "$data_repo_id" \
    --revision "$data_revision" \
    --training-samples "$training_samples" \
    --download-workers "$download_workers" \
    --cache-root "$data_cache_root" \
    --output-root "$data_root"
fi

mkdir -p "$output_root"
echo "[3/4] Qualifying CUDA, BF16, FlashAttention, and a forward/backward pass"
"$uv_bin" run --frozen --no-dev python scripts/check_environment.py \
  --require-gpus "$num_gpus" \
  --output "$output_root/ENVIRONMENT.json"

echo "[4/4] Training ESMC-300M on $num_gpus GPUs for up to $walltime_seconds seconds"
"$uv_bin" run --frozen --no-dev python -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$num_gpus" \
  -m nano_protein.train \
  --config "$config" \
  --data-root "$data_root" \
  --output-root "$output_root" \
  --walltime-seconds "$walltime_seconds"

for receipt in \
  "$data_root/CORPUS_VERIFICATION.json" \
  "$output_root/ENVIRONMENT.json" \
  "$output_root/run_contract.json" \
  "$output_root/TRAINING_COMPLETE.json" \
  "$output_root/checkpoint-final.pt"; do
  if [[ ! -s "$receipt" ]]; then
    echo "Speedrun did not produce the required artifact: $receipt" >&2
    exit 1
  fi
done

echo "Speedrun complete."
echo "  data: $data_root"
echo "  run:  $output_root"
