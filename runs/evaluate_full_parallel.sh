#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

pcore_root="${PCORE_ROOT:-/home/muchenli/datasets/pcore/v0.1}"
contact_root="${CONTACT_ROOT:-/home/muchenli/datasets/esmc-paper-contact-v1}"
external_src="${EXTERNAL_SRC:-/home/muchenli/projects/AutoResearch_ESMC/src}"
data_root="${DATA_ROOT:-$repo_root/data/processed/full-open-v2-4h}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs/stage1-300m-4xa100-4h}"
checkpoint_name="${EVAL_CHECKPOINT:-checkpoint-final.pt}"
visible_gpus="${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
contact_chains="${CONTACT_CHAINS:-20775}"
contact_bootstrap="${CONTACT_BOOTSTRAP:-5000}"
contact_shards="${CONTACT_SHARDS:-3}"
pcore_bootstrap="${PCORE_BOOTSTRAP:-10000}"
pcore_task_parallel="${PCORE_TASK_PARALLEL:-6}"
pcore_probe_threads="${PCORE_PROBE_THREADS:-4}"
pcore_batch_residues="${PCORE_BATCH_RESIDUES:-32768}"
uv_bin="${UV_BIN:-uv}"
uv_cache_dir="${UV_CACHE_DIR:-$repo_root/.uv-cache}"

checkpoint="$output_root/$checkpoint_name"
test -f "$checkpoint"
if [[ "$contact_shards" -lt 1 ]]; then
  echo "CONTACT_SHARDS must be positive" >&2
  exit 2
fi
IFS=',' read -r -a gpu_list <<< "$visible_gpus"
required_gpus=$((contact_shards + 1))
if [[ "${#gpu_list[@]}" -lt "$required_gpus" ]]; then
  echo "full evaluation needs $required_gpus visible GPU identifiers" >&2
  exit 2
fi

eval_root="$output_root/eval-${checkpoint_name%.pt}"
component_root="$eval_root/components"
mkdir -p "$component_root"
UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" sync --frozen

pids=()
for ((shard = 0; shard < contact_shards; shard++)); do
  shard_root="$component_root/contact-shard-$shard"
  mkdir -p "$shard_root"
  CUDA_VISIBLE_DEVICES="${gpu_list[$shard]}" UV_CACHE_DIR="$uv_cache_dir" \
    "$uv_bin" run --frozen python -m nano_protein.evaluate \
    --checkpoint "$checkpoint" \
    --data-root "$data_root" \
    --output-root "$shard_root" \
    --external-src "$external_src" \
    --contact-root "$contact_root" \
    --contact-chains "$contact_chains" \
    --contact-bootstrap 0 \
    --contact-shard-index "$shard" \
    --contact-shard-count "$contact_shards" \
    --run-contact \
    --skip-validation-mlm \
    --resume-components \
    > "$shard_root/evaluate.stdout" 2> "$shard_root/evaluate.stderr" &
  pids+=("$!")
done

pcore_root_output="$component_root/pcore"
mkdir -p "$pcore_root_output"
CUDA_VISIBLE_DEVICES="${gpu_list[$contact_shards]}" UV_CACHE_DIR="$uv_cache_dir" \
  "$uv_bin" run --frozen python -m nano_protein.evaluate \
  --checkpoint "$checkpoint" \
  --data-root "$data_root" \
  --output-root "$pcore_root_output" \
  --external-src "$external_src" \
  --pcore-root "$pcore_root" \
  --pcore-bootstrap "$pcore_bootstrap" \
  --pcore-task-parallel "$pcore_task_parallel" \
  --pcore-probe-threads "$pcore_probe_threads" \
  --pcore-batch-residues "$pcore_batch_residues" \
  --run-pcore \
  --resume-components \
  > "$pcore_root_output/evaluate.stdout" 2> "$pcore_root_output/evaluate.stderr" &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more full-evaluation components failed; inspect $component_root" >&2
  exit "$status"
fi

merge_args=()
for ((shard = 0; shard < contact_shards; shard++)); do
  merge_args+=(--contact-report "$component_root/contact-shard-$shard/EVALUATION.json")
done
UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" run --frozen python \
  scripts/merge_full_evaluation.py \
  "${merge_args[@]}" \
  --pcore-report "$pcore_root_output/EVALUATION.json" \
  --expected-contact-chains "$contact_chains" \
  --contact-bootstrap "$contact_bootstrap" \
  --output "$eval_root/EVALUATION.json" \
  > "$eval_root/merge.stdout" 2> "$eval_root/merge.stderr"
