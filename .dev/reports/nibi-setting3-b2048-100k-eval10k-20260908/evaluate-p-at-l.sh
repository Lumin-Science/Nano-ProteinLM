#!/usr/bin/env bash
set -euo pipefail

repo_root=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908-run
cd "$repo_root"

contact_root="${CONTACT_ROOT:?set CONTACT_ROOT to the frozen contact dataset}"
external_src="${EXTERNAL_SRC:?set EXTERNAL_SRC to the ESMC contact evaluator source}"
data_root="${DATA_ROOT:-$repo_root/data/processed/full-open-v2-4h}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs/stage1-300m-4xa100-4h}"
checkpoint="${CHECKPOINT:-$output_root/checkpoint-final.pt}"
eval_root="${EVAL_OUTPUT_ROOT:-$output_root/eval-p-at-l}"
visible_gpus="${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
contact_chains="${CONTACT_CHAINS:-20775}"
contact_shards="${CONTACT_SHARDS:-32}"
uv_bin="${UV_BIN:-uv}"
uv_cache_dir="${UV_CACHE_DIR:-$repo_root/.uv-cache}"
contact_scoring_cache_root="${CONTACT_SCORING_CACHE_ROOT:-}"

test -f "$checkpoint"
if [[ "$contact_chains" -le 0 || "$contact_shards" -le 0 ]]; then
  echo "CONTACT_CHAINS and CONTACT_SHARDS must be positive" >&2
  exit 2
fi
IFS=',' read -r -a gpu_list <<< "$visible_gpus"
if [[ "${#gpu_list[@]}" -ne 4 ]]; then
  echo "Fast P@L evaluation requires exactly four GPU identifiers" >&2
  exit 2
fi
if [[ -e "$eval_root" ]]; then
  echo "refusing to reuse evaluation output: $eval_root" >&2
  exit 2
fi

component_root="$eval_root/components"
mkdir -p "$component_root"
started_epoch="$(date +%s)"
printf 'started_at_utc=%s\ncheckpoint=%s\nvisible_gpus=%s\ncontact_chains=%s\ncontact_shards=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$checkpoint" "$visible_gpus" \
  "$contact_chains" "$contact_shards" > "$eval_root/EXECUTION.txt"

run_python=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/runtime/.venv/bin/python
cache_args=()
if [[ -n "$contact_scoring_cache_root" ]]; then
  cache_preflight="$eval_root/CONTACT_SCORING_CACHE_PREFLIGHT.json"
  UV_CACHE_DIR="$uv_cache_dir" "$run_python" \
    scripts/verify_contact_scoring_cache.py \
    --cache-root "$contact_scoring_cache_root" \
    --output "$cache_preflight" \
    > "$eval_root/cache-preflight.stdout" 2> "$eval_root/cache-preflight.stderr"
  cache_args=(
    --contact-scoring-cache-root "$contact_scoring_cache_root"
    --contact-scoring-cache-preflight "$cache_preflight"
  )
fi
probe_receipt="$eval_root/CONTACT_PROBE.json"
CUDA_VISIBLE_DEVICES="${gpu_list[0]}" UV_CACHE_DIR="$uv_cache_dir" \
  "$run_python" scripts/fit_contact_probe.py \
  --checkpoint "$checkpoint" \
  --external-src "$external_src" \
  --contact-root "$contact_root" \
  --output "$probe_receipt" \
  > "$eval_root/probe.stdout" 2> "$eval_root/probe.stderr"
pids=()
for ((shard = 0; shard < contact_shards; shard++)); do
  shard_root="$component_root/contact-shard-$shard"
  mkdir -p "$shard_root"
  gpu_slot=$((shard % 4))
  CUDA_VISIBLE_DEVICES="${gpu_list[$gpu_slot]}" UV_CACHE_DIR="$uv_cache_dir" \
    "$run_python" -m nano_protein.evaluate \
    --checkpoint "$checkpoint" \
    --data-root "$data_root" \
    --output-root "$shard_root" \
    --external-src "$external_src" \
    --contact-root "$contact_root" \
    --contact-chains "$contact_chains" \
    --contact-bootstrap 0 \
    --contact-shard-index "$shard" \
    --contact-shard-count "$contact_shards" \
    --contact-probe-receipt "$probe_receipt" \
    "${cache_args[@]}" \
    --run-contact \
    --skip-validation-mlm \
    > "$shard_root/evaluate.stdout" 2> "$shard_root/evaluate.stderr" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "one or more P@L shards failed; inspect $component_root" >&2
  exit "$status"
fi

merge_args=()
for ((shard = 0; shard < contact_shards; shard++)); do
  merge_args+=(--contact-report "$component_root/contact-shard-$shard/EVALUATION.json")
done
UV_CACHE_DIR="$uv_cache_dir" "$run_python" \
  scripts/merge_contact_evaluation.py \
  "${merge_args[@]}" \
  --expected-contact-chains "$contact_chains" \
  --output "$eval_root/P_AT_L.json" \
  > "$eval_root/merge.stdout" 2> "$eval_root/merge.stderr"

finished_epoch="$(date +%s)"
printf 'finished_at_utc=%s\nevaluation_wall_seconds=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$((finished_epoch - started_epoch))" \
  >> "$eval_root/EXECUTION.txt"
cat "$eval_root/EXECUTION.txt"
