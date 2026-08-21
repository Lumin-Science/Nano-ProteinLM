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
pcore_task_timeout="${PCORE_TASK_TIMEOUT_SECONDS:-600}"
pcore_probe_threads="${PCORE_PROBE_THREADS:-4}"
eval_parallel="${EVAL_PARALLEL:-1}"

case "$eval_profile" in
  minimal|standard|speedrun|full) ;;
  *) echo "EVAL_PROFILE must be minimal, standard, speedrun, or full" >&2; exit 2 ;;
esac

uv sync --frozen

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
  CUDA_VISIBLE_DEVICES="$eval_gpu" uv run --frozen python -m nano_protein.evaluate \
    --checkpoint "$checkpoint" \
    --data-root "$data_root" \
    --output-root "$eval_root" \
    --external-src "$external_src" \
    --pcore-root "$pcore_root" \
    --contact-root "$contact_root" \
    --pcore-diagnostic-timeout "$pcore_task_timeout" \
    --pcore-probe-threads "$pcore_probe_threads" \
    --resume-components \
    "${eval_flags[@]}" 2>&1 | tee "$eval_root/evaluate.log"
}

IFS=',' read -r -a gpu_list <<< "$visible_gpus"
if [[ "$eval_parallel" == "1" && ${#gpu_list[@]} -ge 2 ]]; then
  evaluate_one checkpoint-stage1.pt "${gpu_list[0]}" &
  stage_pid=$!
  evaluate_one checkpoint-final.pt "${gpu_list[1]}" &
  final_pid=$!
  status=0
  wait "$stage_pid" || status=$?
  wait "$final_pid" || status=$?
  exit "$status"
fi

eval_gpu="${EVAL_GPU:-${gpu_list[0]}}"
evaluate_one checkpoint-stage1.pt "$eval_gpu"
evaluate_one checkpoint-final.pt "$eval_gpu"
