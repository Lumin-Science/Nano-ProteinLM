#!/usr/bin/env bash
# Default final evaluation: P@L plus two MLM populations, each with five repeats.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo_root/scripts/speedrun.sh" --evaluate "$@"
