#!/usr/bin/env bash
# One-probe contact diagnostic. Use speedrun.sh --evaluate for the final three metrics.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
set -a
if [[ -f .env ]]; then source .env; fi
source .env.example
set +a
run_name="default-100k"
if [[ $# -gt 0 && "$1" != --* ]]; then run_name="$1"; shift; fi
if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Supply a simple run name." >&2
  exit 2
fi
exec "${UV_BIN:-uv}" run --frozen python -m nanoprotein.evaluate \
  --profile component --run-contact --skip-validation-mlm \
  --checkpoint "$OUTPUT_ROOT/$run_name/checkpoint-final.pt" \
  --data-root "$DATA_ROOT/training" --output-root "$OUTPUT_ROOT/$run_name/contact-diagnostic" \
  --contact-root "$DATA_ROOT/evaluation/contact-v3" --external-src "$DATA_ROOT/evaluation/source" \
  --contact-chains 26062 --contact-bootstrap 0 "$@"
