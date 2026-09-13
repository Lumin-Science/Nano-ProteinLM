#!/usr/bin/env bash
# Identical measurements; tasks/171m-p-at-l.md selects P@L as the reward.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash tasks/171m-validation-loss_ar.sh \
  "${1:-configs/default.yaml}" "${2:-experiment-p-at-l-001}"
