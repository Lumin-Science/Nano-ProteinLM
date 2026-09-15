#!/usr/bin/env bash
# One measurement recipe; the agent reviews task boundaries and completion.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
set -a
if [[ -f .env ]]; then source .env; fi
source .env.example
set +a

IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
[[ ${#gpu_ids[@]} -ge 4 ]] || { echo "This task requires four visible GPUs" >&2; exit 2; }
evaluation_gpus="${gpu_ids[0]},${gpu_ids[1]},${gpu_ids[2]},${gpu_ids[3]}"

recipe="${1:-configs/default.yaml}"
run_root="$OUTPUT_ROOT/${2:-experiment-001}"
mkdir -p "$(dirname "$run_root")"
mkdir "$run_root" # Keep previous measurements intact.
cp "$recipe" "$run_root/recipe.yaml"
recipe="$run_root/recipe.yaml"

for seed in 42 43; do
  run_dir="$run_root/seed-$seed"
  uv run --frozen python -m torch.distributed.run --standalone --nproc-per-node=4 \
    -m nanoprotein.train --config "$recipe" --seed "$seed" \
    --data-root "$DATA_ROOT/training" --output-root "$run_dir" \
    --walltime-seconds 3600 --max-steps none --max-model-tokens none --schedule-steps none \
    --attention-backend flash --warmup-steps 554 \
    --checkpoint-interval 0 --periodic-evaluation-interval 0 \
    --peak-bf16-tflops-per-gpu 312

  # The standard evaluator shares one probe across parallel P@L workers.
  uv run --frozen python -m nanoprotein.evaluate \
    --checkpoint "$run_dir/checkpoint-final.pt" --data-root "$DATA_ROOT/training" \
    --output-root "$run_dir/evaluation" \
    --validation-batches 8 --validation-batch-size 4 --validation-context 512 \
    --run-contact --contact-mode parallel --contact-chains 20775 --contact-bootstrap 5000 \
    --contact-gpus "$evaluation_gpus" --contact-workers 32 \
    --contact-root "$DATA_ROOT/evaluation/contact" --external-src "$DATA_ROOT/evaluation/source"
done

uv run --frozen python -m nanoprotein.summarize_training_runs \
  "$run_root/seed-42" "$run_root/seed-43" --validation-sequences 32 \
  --output "$run_root/summary.json"
