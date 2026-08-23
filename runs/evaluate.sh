#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

pcore_root="${PCORE_ROOT:-/home/muchenli/datasets/pcore/v0.1}"
contact_root="${CONTACT_ROOT:-/home/muchenli/datasets/esmc-paper-contact-v1}"
external_src="${EXTERNAL_SRC:-/home/muchenli/projects/AutoResearch_ESMC/src}"
data_root="${DATA_ROOT:-$repo_root/data/processed/pilot-v1}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs/speedrun-300m}"
eval_profile="${EVAL_PROFILE:-speedrun}"
visible_gpus="${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
eval_checkpoints="${EVAL_CHECKPOINTS:-checkpoint-stage1.pt,checkpoint-final.pt}"
pcore_task_timeout="${PCORE_TASK_TIMEOUT_SECONDS:-600}"
pcore_probe_threads="${PCORE_PROBE_THREADS:-4}"
pcore_bootstrap="${PCORE_BOOTSTRAP:-10000}"
pcore_task_parallel="${PCORE_TASK_PARALLEL:-2}"
contact_chains="${CONTACT_CHAINS:-32}"
contact_bootstrap="${CONTACT_BOOTSTRAP:-5000}"
eval_parallel="${EVAL_PARALLEL:-1}"
uv_bin="${UV_BIN:-uv}"
uv_cache_dir="${UV_CACHE_DIR:-$repo_root/.uv-cache}"

case "$eval_profile" in
  minimal|standard|speedrun|full) ;;
  *) echo "EVAL_PROFILE must be minimal, standard, speedrun, or full" >&2; exit 2 ;;
esac

UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" sync --frozen

evaluate_one() {
  local checkpoint_name="$1"
  local eval_gpu="$2"
  checkpoint="$output_root/$checkpoint_name"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  eval_root="$output_root/eval-${checkpoint_name%.pt}"
  mkdir -p "$eval_root"
  eval_flags=()
  if [[ "$eval_profile" != "minimal" ]]; then
    eval_flags+=(--run-contact)
  fi
  if [[ "$eval_profile" == "speedrun" ]]; then
    eval_flags+=(--run-pcore-diagnostic)
  fi
  if [[ "$eval_profile" == "full" ]]; then
    eval_flags+=(--run-pcore)
  fi
  CUDA_VISIBLE_DEVICES="$eval_gpu" UV_CACHE_DIR="$uv_cache_dir" \
    "$uv_bin" run --frozen python -m nano_protein.evaluate \
    --checkpoint "$checkpoint" \
    --data-root "$data_root" \
    --output-root "$eval_root" \
    --external-src "$external_src" \
    --pcore-root "$pcore_root" \
    --contact-root "$contact_root" \
    --contact-chains "$contact_chains" \
    --contact-bootstrap "$contact_bootstrap" \
    --pcore-bootstrap "$pcore_bootstrap" \
    --pcore-task-parallel "$pcore_task_parallel" \
    --pcore-diagnostic-timeout "$pcore_task_timeout" \
    --pcore-probe-threads "$pcore_probe_threads" \
    --resume-components \
    "${eval_flags[@]}" 2>&1 | tee "$eval_root/evaluate.log"
}

IFS=',' read -r -a gpu_list <<< "$visible_gpus"
IFS=',' read -r -a checkpoint_list <<< "$eval_checkpoints"
if [[ "$eval_parallel" == "1" && ${#gpu_list[@]} -ge 2 && ${#checkpoint_list[@]} -eq 2 ]]; then
  evaluate_one "${checkpoint_list[0]}" "${gpu_list[0]}" &
  stage_pid=$!
  evaluate_one "${checkpoint_list[1]}" "${gpu_list[1]}" &
  final_pid=$!
  status=0
  wait "$stage_pid" || status=$?
  wait "$final_pid" || status=$?
  exit "$status"
fi

eval_gpu="${EVAL_GPU:-${gpu_list[0]}}"
for checkpoint_name in "${checkpoint_list[@]}"; do
  evaluate_one "$checkpoint_name" "$eval_gpu"
done
