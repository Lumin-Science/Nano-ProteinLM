#!/usr/bin/env bash
# Compatibility alias: paired 8192-chain metrics with the default MLM selection gate.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec bash tasks/171m-validation-loss_ar.sh "$@"
