#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Original ESMC-171M AdamW baseline: four L40S GPUs for one hour.
# Frozen training and evaluation settings for the default program.md.
data_repo_id="LuminScience/LuminBench-Nano-ESMC"
data_revision="bd38448d50d8f426d7b9bd4410b53159ea001259"
release_manifest_sha256="fe1ac0657085ab19fe6f56786006e9eb004ca66bc6c5b81dfd8e6bc3dcfda6ff"
training_samples=5376000
baseline_config="$repo_root/configs/esmc-171m-original.yaml"
parameter_ceiling=171000000
num_gpus=4
walltime_seconds=3600
contact_chains=20775
contact_manifest_sha256="c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3"

artifact_root="${ARTIFACT_ROOT:-$repo_root/.exps}"
data_cache_root="${DATA_CACHE_ROOT:-$artifact_root/cache/huggingface-dataset}"
data_root="${DATA_ROOT:-$artifact_root/data/training-samples-$training_samples}"
output_root="${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh round directory}"
config="${CONFIG:-$baseline_config}"
contact_root="${CONTACT_ROOT:?set CONTACT_ROOT to the frozen contact dataset}"
external_src="${EXTERNAL_SRC:?set EXTERNAL_SRC to the ESMC contact evaluator source}"
visible_gpus="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
contact_shards="${CONTACT_SHARDS:-32}"
contact_scoring_cache_root="${CONTACT_SCORING_CACHE_ROOT:-}"
download_workers="${DOWNLOAD_WORKERS:-8}"
uv_bin="${UV_BIN:-uv}"
uv_cache_dir="${UV_CACHE_DIR:-$repo_root/.uv-cache}"

if [[ -e "$output_root" ]]; then
  echo "refusing to reuse round output: $output_root" >&2
  exit 2
fi
if [[ ! -f "$config" ]]; then
  echo "missing experiment config: $config" >&2
  exit 2
fi
IFS=',' read -r -a gpu_list <<< "$visible_gpus"
if [[ "${#gpu_list[@]}" -ne "$num_gpus" ]]; then
  echo "AutoResearch rounds require exactly $num_gpus visible GPU identifiers" >&2
  exit 2
fi

mkdir -p "$output_root"
state_path="$output_root/ROUND_STATE"
round_state="running"

write_state() {
  local temporary="$state_path.partial"
  printf 'state=%s\nupdated_at_utc=%s\npid=%s\noutput_root=%s\n' \
    "$round_state" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$output_root" \
    > "$temporary"
  mv "$temporary" "$state_path"
}

record_exit() {
  local status=$?
  if [[ "$status" -ne 0 ]]; then
    round_state="failed"
    write_state
  fi
}
trap record_exit EXIT
write_state

mkdir -p "$data_cache_root"
UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" sync --frozen
UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" run --frozen python -c \
  'import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
observed = int(config["expected_parameter_count"])
ceiling = int(sys.argv[2])
if observed > ceiling:
    raise SystemExit(f"candidate exceeds {ceiling:,}-parameter ceiling: {observed:,}")' \
  "$config" "$parameter_ceiling"
if [[ -f "$data_root/manifest.json" && -f "$data_root/CORPUS_VERIFICATION.json" ]]; then
  :
elif [[ -e "$data_root" ]]; then
  echo "prepared data directory is incomplete: $data_root" >&2
  exit 2
else
  UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" run --frozen python scripts/download_data.py \
    --repo-id "$data_repo_id" \
    --revision "$data_revision" \
    --training-samples "$training_samples" \
    --download-workers "$download_workers" \
    --cache-root "$data_cache_root" \
    --output-root "$data_root"
fi
UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" run --frozen python -c \
  'import json, sys
manifest = json.load(open(sys.argv[1]))
expected = {"uniref90": 2430914, "mgnify": 1437829, "omg_img": 3240726}
observed = {name: int(manifest["sources"][name]["train"]["records"]) for name in expected}
if manifest.get("release_manifest_sha256") != sys.argv[2] or observed != expected:
    raise SystemExit("prepared corpus differs from the frozen AutoResearch reference")' \
  "$data_root/manifest.json" "$release_manifest_sha256"
UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" run --frozen python -c \
  'from pathlib import Path
import sys
sys.path.insert(0, sys.argv[2])
from autoresearch_esm.paper_contact_runtime import ContactDataset
observed = ContactDataset(Path(sys.argv[1])).manifest_receipt.manifest_sha256
if observed != sys.argv[3]:
    raise SystemExit("contact population differs from the frozen AutoResearch reference")' \
  "$contact_root" "$external_src" "$contact_manifest_sha256"
UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" run --frozen python scripts/check_environment.py \
  --require-gpus "$num_gpus" \
  --output "$output_root/ENVIRONMENT.json"

UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" run --frozen python -m torch.distributed.run \
  --standalone \
  --nproc-per-node="$num_gpus" \
  -m nano_protein.train \
  --config "$config" \
  --data-root "$data_root" \
  --output-root "$output_root" \
  --walltime-seconds "$walltime_seconds" \
  > "$output_root/train.stdout" 2> "$output_root/train.stderr"

test -s "$output_root/TRAINING_COMPLETE.json"
test -s "$output_root/checkpoint-final.pt"
UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" run --frozen python -c \
  'import json, sys; report = json.load(open(sys.argv[1])); assert report["stop_reason"] == "walltime", report["stop_reason"]' \
  "$output_root/TRAINING_COMPLETE.json"

validation_root="$output_root/eval-validation"
CUDA_VISIBLE_DEVICES="${gpu_list[0]}" UV_CACHE_DIR="$uv_cache_dir" \
  "$uv_bin" run --frozen python -m nano_protein.evaluate \
  --checkpoint "$output_root/checkpoint-final.pt" \
  --data-root "$data_root" \
  --output-root "$validation_root" \
  --validation-batches 8 \
  --validation-batch-size 4 \
  --validation-context 512 \
  > "$output_root/validation.stdout" 2> "$output_root/validation.stderr"

DATA_ROOT="$data_root" \
OUTPUT_ROOT="$output_root" \
CHECKPOINT="$output_root/checkpoint-final.pt" \
EVAL_OUTPUT_ROOT="$output_root/eval-p-at-l" \
EVAL_GPUS="$visible_gpus" \
CONTACT_ROOT="$contact_root" \
EXTERNAL_SRC="$external_src" \
CONTACT_CHAINS="$contact_chains" \
CONTACT_SHARDS="$contact_shards" \
CONTACT_SCORING_CACHE_ROOT="$contact_scoring_cache_root" \
UV_BIN="$uv_bin" \
UV_CACHE_DIR="$uv_cache_dir" \
bash runs/evaluate_p_at_l_parallel.sh \
  > "$output_root/p-at-l.stdout" 2> "$output_root/p-at-l.stderr"

UV_CACHE_DIR="$uv_cache_dir" "$uv_bin" run --frozen python \
  scripts/summarize_autoresearch_round.py \
  --output-root "$output_root"

round_state="complete"
write_state
trap - EXIT
echo "AutoResearch round complete: $output_root/ROUND_SUMMARY.json"
