#!/usr/bin/env bash
# Usage: bash runs/speedrun.sh [recipe.yaml] [run-name] [nanoprotein.train options...]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
recipe="${1:-configs/default.yaml}"
run_name="${2:-setting3-100k}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
if [[ ! -f "$recipe" || ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Supply an existing recipe and a simple run name (letters, numbers, '.', '_', '-')." >&2
  exit 1
fi

source "$repo_root/runs/setup_env_and_data.sh" --
run_dir="$OUTPUT_ROOT/$run_name"
mkdir "$run_dir" # Refuse to overwrite a previous run.
cp "$recipe" "$run_dir/recipe.yaml"
train_args=(
  --config "$run_dir/recipe.yaml"
  --max-steps 100000 --walltime-seconds 57600
  "$@"
)
# Qualify the effective backend, including any normal training-CLI override.
attention_backend="$("$uv_bin" run --frozen --no-dev python -m nanoprotein.train \
  "${train_args[@]}" --print-config | "$uv_bin" run --frozen --no-dev python -c \
  'import sys, yaml; print(yaml.safe_load(sys.stdin)["attention_backend"])')"
"$uv_bin" run --frozen --no-dev python -m nanoprotein.check_environment \
  --require-gpus 4 --attention-backend "$attention_backend" \
  --output "$run_dir/ENVIRONMENT.json"

echo "Training $recipe on four GPUs; output: $run_dir"
"$uv_bin" run --frozen --no-dev python -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m nanoprotein.train \
  "${train_args[@]}" --data-root "$DATA_ROOT/training" --output-root "$run_dir"

for receipt in ENVIRONMENT.json run_contract.json TRAINING_COMPLETE.json checkpoint-final.pt; do
  if [[ ! -s "$run_dir/$receipt" ]]; then
    echo "Training did not produce the required artifact: $run_dir/$receipt" >&2
    exit 1
  fi
done
echo "Training finished: $run_dir/TRAINING_COMPLETE.json"
