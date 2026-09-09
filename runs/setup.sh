#!/usr/bin/env bash
# Usage: bash runs/setup.sh [--training-shards N | --training-samples N]
# Run alone to prepare data, or source from speedrun.sh to share the roots.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
set -a
if [[ -f "$repo_root/.env" ]]; then
  source "$repo_root/.env"
fi
source "$repo_root/.env.example"
set +a
: "${DATA_ROOT:?set DATA_ROOT in .env}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT in .env}"
training_shards=""
training_samples=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --training-shards)
      if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "--training-shards needs an integer from 3 to 565." >&2
        exit 1
      fi
      training_shards="$2"
      shift 2 ;;
    --training-samples)
      if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "--training-samples needs a positive integer (include sampling headroom)." >&2
        exit 1
      fi
      training_samples="$2"
      shift 2 ;;
    --) shift; break ;;
    -h|--help)
      echo "Usage: bash runs/setup.sh [--training-shards N | --training-samples N]"
      echo "Default: 7 of 565 training Parquet shards; all MLM validation and contact P@L data."
      echo "Choose a fresh DATA_ROOT for a different training shard count."
      return 0 2>/dev/null || exit 0 ;;
    *) echo "Unknown setup argument: $1" >&2; exit 1 ;;
  esac
done
if [[ -n "$training_shards" && -n "$training_samples" ]]; then
  echo "Choose either --training-shards or --training-samples." >&2
  exit 1
fi
uv_bin="${UV_BIN:-uv}"
if ! command -v "$uv_bin" >/dev/null 2>&1; then
  echo "Install uv >=0.11.31,<0.12, then rerun this script." >&2
  exit 1
fi

echo "Preparing the locked Python environment"
"$uv_bin" sync --frozen --no-dev

if [[ -z "$training_shards" && -z "$training_samples" ]]; then
  if [[ -f "$DATA_ROOT/training/download-plan.json" ]]; then
    training_shards="$("$uv_bin" run --frozen --no-dev python -c \
      'import json, sys; p=json.load(open(sys.argv[1])); print(sum(len(s["train"]) for s in p["sources"].values()))' \
      "$DATA_ROOT/training/download-plan.json")"
  else
    training_shards=7
  fi
fi

budget_args=(--training-shards "$training_shards")
if [[ -n "$training_samples" ]]; then
  budget_args=(--training-samples "$training_samples")
fi
echo "Preparing verified training shards for ${budget_args[*]} and all MLM validation shards"
"$uv_bin" run --frozen --no-dev python -m nanoprotein.sharded_data \
  --repo-id LuminScience/LuminBench-Nano-ESMC \
  --revision bd38448d50d8f426d7b9bd4410b53159ea001259 \
  "${budget_args[@]}" --reuse \
  --cache-root "$DATA_ROOT/cache" --output-root "$DATA_ROOT/training"

"$uv_bin" run --frozen --no-dev python -m nanoprotein.setup_evaluation \
  --data-root "$DATA_ROOT"
mkdir -p "$OUTPUT_ROOT"
echo "Setup complete: training + MLM validation + contact P@L; outputs: $OUTPUT_ROOT"
