#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cluster_root="${CLUSTER_ROOT:-/home/muchenli/workspace/esmc-open-step9-clusters-v1/clusters}"
pcore_root="${PCORE_ROOT:-/home/muchenli/datasets/pcore/v0.1}"
contact_root="${CONTACT_ROOT:-/home/muchenli/datasets/esmc-paper-contact-v1}"
external_src="${EXTERNAL_SRC:-/home/muchenli/projects/AutoResearch_ESMC/src}"
data_root="${DATA_ROOT:-$repo_root/data/processed/pilot-v1}"
output_root="${OUTPUT_ROOT:-$repo_root/outputs/speedrun-300m}"
config="${CONFIG:-$repo_root/configs/esmc_300m.yaml}"
walltime_seconds="${WALLTIME_SECONDS:-1800}"
eval_profile="${EVAL_PROFILE:-full}"
visible_gpus="${CUDA_VISIBLE_DEVICES:-0}"
eval_gpu="${EVAL_GPU:-${visible_gpus%%,*}}"

mkdir -p "$output_root"
uv sync --frozen
uv run --frozen python scripts/check_environment.py \
  --require-gpus 4 \
  --output "$output_root/ENVIRONMENT.json"

if [[ ! -f "$data_root/manifest.json" ]]; then
  uv run --frozen python scripts/prepare_data.py \
    --cluster-root "$cluster_root" \
    --output-root "$data_root" \
    --pcore-index "$pcore_root/index.jsonl" \
    --contact-manifest "$contact_root/CONTACT_MANIFEST.jsonl"
fi

uv run --frozen python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  -m nano_protein.train \
  --config "$config" \
  --data-root "$data_root" \
  --output-root "$output_root" \
  --walltime-seconds "$walltime_seconds"

for checkpoint_name in checkpoint-stage1.pt checkpoint-final.pt; do
  checkpoint="$output_root/$checkpoint_name"
  eval_root="$output_root/eval-${checkpoint_name%.pt}"
  eval_flags=()
  if [[ "$eval_profile" == "standard" || "$eval_profile" == "full" ]]; then
    eval_flags+=(--run-contact)
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
    "${eval_flags[@]}"
done
