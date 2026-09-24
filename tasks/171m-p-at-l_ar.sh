#!/usr/bin/env bash
# Identical single-run measurement; the task selects P@L as the score.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash tasks/171m-validation-loss_ar.sh "$@"
