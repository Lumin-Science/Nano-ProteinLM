#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cluster_root="${CLUSTER_ROOT:-/home/muchenli/workspace/esmc-open-step9-clusters-v1/clusters}"
pcore_root="${PCORE_ROOT:-/home/muchenli/datasets/pcore/v0.1}"
contact_root="${CONTACT_ROOT:-/home/muchenli/datasets/esmc-paper-contact-v1}"
homology_root="${HOMOLOGY_ROOT:-/home/muchenli/datasets/evaluation-decontamination/stage1-candidates-v1}"
data_root="${DATA_ROOT:-$repo_root/data/processed/stage1-300m-production-v1}"
train_per_source="${TRAIN_PER_SOURCE:-3000000}"
validation_per_source="${VALIDATION_PER_SOURCE:-4096}"
uv_bin="${UV_BIN:-uv}"

receipt="$homology_root/HOMOLOGY_EXCLUSION_VERIFIED.json"
digests="$homology_root/homology_excluded_all_splits.txt"
test -f "$receipt"
test -f "$digests"
mkdir -p "$(dirname "$data_root")"
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" "$uv_bin" sync --frozen
if [[ -e "$data_root/manifest.json" ]]; then
  echo "verifying existing prepared corpus: $data_root/manifest.json"
else
  UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" "$uv_bin" run --frozen python \
    scripts/prepare_data.py \
    --cluster-root "$cluster_root" \
    --output-root "$data_root" \
    --pcore-index "$pcore_root/index.jsonl" \
    --contact-manifest "$contact_root/CONTACT_MANIFEST.jsonl" \
    --homology-exclusion-digests "$digests" \
    --homology-exclusion-receipt "$receipt" \
    --train-per-source "$train_per_source" \
    --validation-per-source "$validation_per_source" \
    > "$data_root.prepare.stdout" 2> "$data_root.prepare.stderr"
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" "$uv_bin" run --frozen python \
  scripts/verify_prepared_corpus.py \
  --data-root "$data_root" \
  > "$data_root.verify.stdout" 2> "$data_root.verify.stderr"

jq -e \
  --argjson train "$train_per_source" \
  --argjson validation "$validation_per_source" \
  '.decontamination.homology_exclusion == true and
   ([.sources[].train.records] | all(. == $train)) and
   ([.sources[].validation.records] | all(. == $validation))' \
  "$data_root/manifest.json" > /dev/null
jq -e \
  '.status == "verified" and
   .protocol == "prepared-corpus-decontamination-verification-v1" and
   ([.sources[].train_excluded_intersection] | all(. == 0)) and
   ([.sources[].validation_excluded_intersection] | all(. == 0)) and
   ([.sources[].train_validation_intersection] | all(. == 0))' \
  "$data_root/CORPUS_VERIFICATION.json" > /dev/null
