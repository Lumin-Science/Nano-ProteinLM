#!/usr/bin/env bash
# Create a fresh agent workspace from the released autoresearch branch.
set -euo pipefail
script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON_BIN:-python3}" "$script_root/.dev/scripts/prepare_autoresearch.py" "$@"
