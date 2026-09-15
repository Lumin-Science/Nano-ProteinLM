#!/usr/bin/env bash
# Compatibility entry point; the standard evaluator owns parallel P@L execution.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output_root="${OUTPUT_ROOT:-$repo_root/outputs/stage1-300m-4xa100-4h}"
eval_root="${EVAL_OUTPUT_ROOT:-$output_root/eval-p-at-l}"
if [[ -e "$eval_root" ]]; then
  echo "refusing to reuse evaluation output: $eval_root" >&2
  exit 2
fi
contact_args=(--run-contact --skip-validation-mlm)
if [[ -n "${CONTACT_SCORING_CACHE_ROOT:-}" ]]; then
  contact_args+=(--contact-scoring-cache-root "$CONTACT_SCORING_CACHE_ROOT")
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/.uv-cache}"
exec "${UV_BIN:-uv}" run --frozen python -m nanoprotein.evaluate \
  --checkpoint "${CHECKPOINT:-$output_root/checkpoint-final.pt}" \
  --data-root "${DATA_ROOT:-$repo_root/data/processed/full-open-v2-4h}" \
  --output-root "$eval_root" \
  --contact-root "${CONTACT_ROOT:?set CONTACT_ROOT to the frozen contact dataset}" \
  --external-src "${EXTERNAL_SRC:?set EXTERNAL_SRC to the ESMC contact evaluator source}" \
  --contact-gpus "${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}" \
  --contact-workers "${CONTACT_SHARDS:-32}" \
  --contact-chains "${CONTACT_CHAINS:-20775}" \
  "${contact_args[@]}"
