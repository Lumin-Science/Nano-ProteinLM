#!/usr/bin/env bash
# Run on its own to prepare training, or source from speedrun.sh to share the roots.
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
uv_bin="${UV_BIN:-uv}"
if ! command -v "$uv_bin" >/dev/null 2>&1; then
  echo "Install uv >=0.11.31,<0.12, then rerun this script." >&2
  exit 1
fi

echo "Preparing the locked Python environment"
"$uv_bin" sync --frozen --no-dev

if [[ -e "$DATA_ROOT/training" ]]; then
  if [[ ! -f "$DATA_ROOT/training/manifest.json" || ! -f "$DATA_ROOT/training/CORPUS_VERIFICATION.json" ]]; then
    echo "Incomplete data at $DATA_ROOT/training; choose a fresh DATA_ROOT or move it aside." >&2
    exit 1
  fi
  echo "Reusing prepared training data: $DATA_ROOT/training"
else
  echo "Downloading and verifying the benchmark training subset"
  "$uv_bin" run --frozen --no-dev python -m nanoprotein.sharded_data \
    --repo-id LuminScience/LuminBench-Nano-ESMC \
    --revision bd38448d50d8f426d7b9bd4410b53159ea001259 \
    --training-samples 5376000 \
    --cache-root "$DATA_ROOT/cache" --output-root "$DATA_ROOT/training"
fi

# Validate existing receipts too, so another corpus is never silently reused.
"$uv_bin" run --frozen --no-dev python - "$DATA_ROOT/training" <<'PYTHON'
from pathlib import Path
import sys
from nanoprotein.train import validate_data_manifest

manifest = validate_data_manifest(Path(sys.argv[1]))
counts = {name: manifest['sources'][name]['train']['records'] for name in ('uniref90', 'mgnify', 'omg_img')}
if manifest['release_manifest_sha256'] != 'fe1ac0657085ab19fe6f56786006e9eb004ca66bc6c5b81dfd8e6bc3dcfda6ff' or counts != {'uniref90': 2430914, 'mgnify': 1437829, 'omg_img': 3240726}:
    raise ValueError('prepared corpus differs from the benchmark subset; use a fresh DATA_ROOT')
PYTHON
mkdir -p "$OUTPUT_ROOT"
echo "Setup complete. Training data: $DATA_ROOT/training; outputs: $OUTPUT_ROOT"
