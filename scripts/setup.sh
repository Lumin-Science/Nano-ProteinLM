#!/usr/bin/env bash
# Usage: bash scripts/setup.sh [--training-shards N | --training-samples N] [--recovered-contact-pool PATH | --contact-v3-archive PATH]
# Run alone to prepare data, or source from speedrun.sh to share the roots.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
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
recovered_contact_pool=""
contact_v3_archive=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --contact-v3-archive)
      [[ $# -ge 2 ]] || { echo "--contact-v3-archive needs a file" >&2; exit 2; }
      contact_v3_archive="$2"
      shift 2 ;;
    --recovered-contact-pool)
      [[ $# -ge 2 ]] || { echo "--recovered-contact-pool needs a directory" >&2; exit 2; }
      recovered_contact_pool="$2"
      shift 2 ;;
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
      echo "Usage: bash scripts/setup.sh [--training-shards N | --training-samples N] [--recovered-contact-pool PATH | --contact-v3-archive PATH]"
      echo "Default: 30 of 565 training Parquet shards; all MLM validation; evaluation v3 requires the verified local recovery or portable v3 archive."
      echo "Choose a fresh DATA_ROOT for a different training shard count."
      return 0 2>/dev/null || exit 0 ;;
    *) echo "Unknown setup argument: $1" >&2; exit 1 ;;
  esac
done
if [[ -n "$training_shards" && -n "$training_samples" ]]; then
  echo "Choose either --training-shards or --training-samples." >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/evaluation/contact-v3" && -z "$recovered_contact_pool" && -z "$contact_v3_archive" ]]; then
  echo "Fresh evaluation v3 setup requires --recovered-contact-pool PATH or --contact-v3-archive PATH." >&2
  exit 2
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
    training_shards=30
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
  --data-root "$DATA_ROOT" --historical-only
profile_args=()
if [[ -n "$recovered_contact_pool" ]]; then
  profile_args+=(--recovered-contact-pool "$recovered_contact_pool")
fi
if [[ -n "$contact_v3_archive" ]]; then
  profile_args+=(--contact-v3-archive "$contact_v3_archive")
fi
"$uv_bin" run --frozen --no-dev python -m nanoprotein.prepare_evaluation_profiles \
  --data-root "$DATA_ROOT" "${profile_args[@]}"
mkdir -p "$OUTPUT_ROOT"
echo "Setup complete: training + frozen search/scale-up evaluation profiles; outputs: $OUTPUT_ROOT"
