#!/usr/bin/env bash
# Same training measurement as the validation-loss task, plus full contact P@L as the score.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NANOPROTEIN_TASK_CONTACT=1 exec bash tasks/171m-validation-loss_ar.sh "$@"
