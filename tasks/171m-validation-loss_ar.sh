#!/usr/bin/env bash
# Measure one budgeted training run; search policy belongs to the caller.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
  echo "Usage: bash tasks/171m-validation-loss_ar.sh RECIPE RUN_NAME SEED"
  echo "One run: 20 minutes on 4 H100s or 1 hour on 4 L40S GPUs, then MLM validation on all 12,288 proteins."
}
if [[ "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ $# -ne 3 ]]; then usage >&2; exit 2; fi
recipe="$1"
run_name="$2"
seed="$3"
if [[ ! -f "$recipe" || ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "Supply an existing recipe, a simple fresh run name and a nonnegative integer seed." >&2
  exit 2
fi

set -a
if [[ -f .env ]]; then source .env; fi
source .env.example
set +a
uv_bin="${UV_BIN:-uv}"
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
[[ ${#gpu_ids[@]} -eq 4 ]] || { echo "Expose exactly four matching GPUs" >&2; exit 2; }
evaluation_gpus="${gpu_ids[0]},${gpu_ids[1]},${gpu_ids[2]},${gpu_ids[3]}"

run_root="$OUTPUT_ROOT/$run_name"
mkdir -p "$OUTPUT_ROOT"
mkdir "$run_root" # Refuse to overwrite a previous measurement.
cp "$recipe" "$run_root/recipe.yaml"
"$uv_bin" run --frozen --no-dev python -m nanoprotein.check_environment \
  --require-gpus 4 --autoresearch --attention-backend auto \
  --output "$run_root/ENVIRONMENT.json"

read -r attention_backend peak_tflops training_seconds < <(
  "$uv_bin" run --frozen --no-dev python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["attention_backend"], r["peak_bf16_tflops_per_gpu"], r["training_walltime_seconds"])' "$run_root/ENVIRONMENT.json"
)
"$uv_bin" run --frozen --no-dev python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m nanoprotein.train --config "$run_root/recipe.yaml" --seed "$seed" \
  --data-root "$DATA_ROOT/training" --output-root "$run_root" \
  --walltime-seconds "$training_seconds" --max-steps none --max-model-tokens none --schedule-steps none \
  --attention-backend "$attention_backend" --warmup-steps 500 \
  --checkpoint-interval 0 --periodic-evaluation-interval 0 --warm-data-cache \
  --peak-bf16-tflops-per-gpu "$peak_tflops"

# Validation loss needs only MLM evaluation; the P@L task's wrapper also requests contact P@L.
evaluation_args=(--validation-context 512)
if [[ "${NANOPROTEIN_TASK_CONTACT:-0}" == 1 ]]; then
  evaluation_args+=(
    --run-contact --contact-mode parallel --contact-chains 20775 --contact-bootstrap 5000
    --contact-gpus "$evaluation_gpus" --contact-workers 32
    --contact-root "$DATA_ROOT/evaluation/contact" --external-src "$DATA_ROOT/evaluation/source"
  )
fi
"$uv_bin" run --frozen --no-dev python -m nanoprotein.evaluate \
  --checkpoint "$run_root/checkpoint-final.pt" --data-root "$DATA_ROOT/training" \
  --output-root "$run_root/evaluation" "${evaluation_args[@]}"
