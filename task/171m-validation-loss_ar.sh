#!/usr/bin/env bash
# One recipe, two seeds: direct training and evaluation, with no search/acceptance policy.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a
if [[ -f .env ]]; then
  source .env
fi
source .env.example
set +a
: "${DATA_ROOT:?set DATA_ROOT in .env}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT in .env}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
recipe="${1:-configs/default.yaml}"
run_name="${2:-171m-validation-loss-ar-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
run_root="$OUTPUT_ROOT/$run_name"
mkdir -p "$OUTPUT_ROOT"
mkdir "$run_root" # Refuse an existing experiment; every score uses fresh runs.
cp "$recipe" "$run_root/recipe.yaml"
recipe="$run_root/recipe.yaml"
uv sync --frozen

# Input locations may vary; the benchmark corpus, contact population and size bound do not.
uv run --frozen python - "$recipe" "$DATA_ROOT" <<'PY'
import json, pathlib, sys, yaml
config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
if not 162137610 <= config['expected_parameter_count'] <= 179204726:
    raise ValueError('recipe must stay within ±5% of the 171M reference')
stages = config['stages']
if len(stages) != 1 or stages[0]['name'] != 'stage1' or stages[0]['context_length'] != 512:
    raise ValueError('the task requires Stage 1 at context 512')
if stages[0]['mixture'] != {'uniref90':0.36, 'mgnify':0.11, 'omg_img':0.54}:
    raise ValueError('the training mixture is frozen')
if config.get('stage1_cooldown_fraction', 0) or config.get('expected_world_size', 4) != 4:
    raise ValueError('the task requires four GPUs and constant LR after warmup')
manifest = json.loads((root/'training/manifest.json').read_text())
if not (root/'training/CORPUS_VERIFICATION.json').is_file():
    raise ValueError('prepare and verify the training corpus first')
counts = {name:manifest['sources'][name]['train']['records'] for name in ('uniref90','mgnify','omg_img')}
if manifest['release_manifest_sha256'] != 'fe1ac0657085ab19fe6f56786006e9eb004ca66bc6c5b81dfd8e6bc3dcfda6ff' or counts != {'uniref90':2430914,'mgnify':1437829,'omg_img':3240726}:
    raise ValueError('prepared data differs from the benchmark release')
sys.path.insert(0, str(root/'evaluation/source'))
from autoresearch_esm.paper_contact_runtime import ContactDataset
if ContactDataset(root/'evaluation/contact').manifest_receipt.manifest_sha256 != 'c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3':
    raise ValueError('contact data differs from the benchmark population')
PY
uv run --frozen python scripts/check_environment.py \
  --require-gpus 4 --gpu-name L40S --attention-backend flash \
  --output "$run_root/environment.json"

for seed in 42 43; do
  run_dir="$run_root/seed-$seed"
  uv run --frozen python -m torch.distributed.run --standalone --nproc-per-node=4 \
    -m nano_protein.train --config "$recipe" --seed "$seed" \
    --data-root "$DATA_ROOT/training" --output-root "$run_dir" \
    --walltime-seconds 3600 --max-steps none --max-model-tokens none --schedule-steps none \
    --attention-backend flash --warmup-steps 554 \
    --checkpoint-interval 0 --periodic-evaluation-interval 0 \
    --peak-bf16-tflops-per-gpu 312

  uv run --frozen python - "$run_dir/TRAINING_COMPLETE.json" <<'PY'
import json, pathlib, sys
r = json.loads(pathlib.Path(sys.argv[1]).read_text())
if r['stop_reason'] != 'walltime' or r['training_seconds'] < 3600:
    raise ValueError('incomplete research training budget')
if not 162137610 <= r['parameter_count'] <= 179204726:
    raise ValueError('actual model size is outside the task bound')
PY
  # MLM supplies the score. Full P@L remains a diagnostic and does not select recipes.
  uv run --frozen python -m nano_protein.evaluate \
    --checkpoint "$run_dir/checkpoint-final.pt" --data-root "$DATA_ROOT/training" \
    --output-root "$run_dir/evaluation" \
    --validation-batches 8 --validation-batch-size 4 --validation-context 512 \
    --run-contact --contact-chains 20775 --contact-bootstrap 5000 \
    --contact-root "$DATA_ROOT/evaluation/contact" --external-src "$DATA_ROOT/evaluation/source"
done

uv run --frozen python scripts/summarize_training_runs.py \
  "$run_root/seed-42" "$run_root/seed-43" --validation-sequences 32 \
  --output "$run_root/summary.json"
