#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cluster_root="${CLUSTER_ROOT:-/home/muchenli/workspace/esmc-open-step9-clusters-v1/clusters}"
evaluation_root="${EVALUATION_ROOT:-/home/muchenli/datasets/evaluation-decontamination/v1}"
output_root="${OUTPUT_ROOT:-/home/muchenli/datasets/evaluation-decontamination/mmseqs-v1}"
mmseqs="${MMSEQS_BIN:-/home/muchenli/.local/mmseqs/17-b804f/bin/mmseqs}"
uv_bin="${UV_BIN:-/home/muchenli/.local/bin/uv}"
threads="${MMSEQS_THREADS:-64}"
maximum_sequences="${MMSEQS_MAX_SEQS:-1000000}"

query_fasta="$evaluation_root/evaluation_all_splits.fasta"
ledger="$evaluation_root/EVALUATION_SPLIT_LEDGER.json"
memberships="$evaluation_root/sequence_memberships.jsonl"

test -x "$mmseqs"
test -f "$query_fasta"
test -f "$ledger"
test -f "$memberships"
mkdir -p "$output_root/db" "$output_root/results" "$output_root/tmp" "$output_root/logs"

commands="$output_root/MMSEQS_COMMANDS.txt"
: > "$commands"
printf '%q ' "$mmseqs" createdb "$query_fasta" "$output_root/db/query" >> "$commands"
printf '\n' >> "$commands"
"$mmseqs" createdb "$query_fasta" "$output_root/db/query" \
  > "$output_root/logs/query-createdb.log" 2>&1

for source in uniref90 mgnify omg_img; do
  fasta="$cluster_root/$source/representatives.fasta"
  target="$output_root/db/$source"
  result="$output_root/results/$source"
  temporary="$output_root/tmp/$source"
  hits="$output_root/results/$source.tsv"
  test -f "$fasta"

  printf '%q ' "$mmseqs" createdb "$fasta" "$target" >> "$commands"
  printf '\n' >> "$commands"
  "$mmseqs" createdb "$fasta" "$target" \
    > "$output_root/logs/$source-createdb.log" 2>&1

  printf '%q ' "$mmseqs" search "$output_root/db/query" "$target" "$result" \
    "$temporary" --min-seq-id 0.30 -c 0.80 --cov-mode 0 --max-seqs \
    "$maximum_sequences" -s 7.5 --threads "$threads" >> "$commands"
  printf '\n' >> "$commands"
  "$mmseqs" search "$output_root/db/query" "$target" "$result" "$temporary" \
    --min-seq-id 0.30 \
    -c 0.80 \
    --cov-mode 0 \
    --max-seqs "$maximum_sequences" \
    -s 7.5 \
    --threads "$threads" \
    > "$output_root/logs/$source-search.log" 2>&1

  printf '%q ' "$mmseqs" convertalis "$output_root/db/query" "$target" "$result" \
    "$hits" --format-output query,target,pident,alnlen,qcov,tcov,evalue,bits >> "$commands"
  printf '\n' >> "$commands"
  "$mmseqs" convertalis "$output_root/db/query" "$target" "$result" "$hits" \
    --format-output query,target,pident,alnlen,qcov,tcov,evalue,bits \
    > "$output_root/logs/$source-convertalis.log" 2>&1
  test -s "$hits"
done

mmseqs_version="$($mmseqs version)"
jq -n \
  --arg status complete \
  --arg protocol mmseqs2-evaluation-homology-search-v1 \
  --arg mmseqs_version "$mmseqs_version" \
  --arg query_fasta_sha256 "$(sha256sum "$query_fasta" | awk '{print $1}')" \
  --arg ledger_sha256 "$(sha256sum "$ledger" | awk '{print $1}')" \
  --arg commands_sha256 "$(sha256sum "$commands" | awk '{print $1}')" \
  --argjson threads "$threads" \
  --argjson maximum_sequences "$maximum_sequences" \
  '{
    status: $status,
    protocol: $protocol,
    mmseqs_version: $mmseqs_version,
    query_fasta_sha256: $query_fasta_sha256,
    evaluation_split_ledger_sha256: $ledger_sha256,
    commands_sha256: $commands_sha256,
    threads: $threads,
    maximum_sequences_per_query: $maximum_sequences,
    minimum_sequence_identity: 0.30,
    minimum_query_coverage: 0.80,
    minimum_target_coverage: 0.80,
    coverage_mode: 0
  }' > "$output_root/MMSEQS_SEARCH_COMPLETE.json.partial"
mv "$output_root/MMSEQS_SEARCH_COMPLETE.json.partial" \
  "$output_root/MMSEQS_SEARCH_COMPLETE.json"

cd "$repo_root"
UV_CACHE_DIR="${UV_CACHE_DIR:-/home/muchenli/.cache/uv}" \
  "$uv_bin" run --frozen python scripts/finalize_homology_exclusion.py \
  --ledger "$ledger" \
  --memberships "$memberships" \
  --hits "uniref90=$output_root/results/uniref90.tsv" \
  --hits "mgnify=$output_root/results/mgnify.tsv" \
  --hits "omg_img=$output_root/results/omg_img.tsv" \
  --source-receipt "uniref90=$cluster_root/uniref90/verification.json" \
  --source-receipt "mgnify=$cluster_root/mgnify/verification.json" \
  --source-receipt "omg_img=$cluster_root/omg_img/verification.json" \
  --command-receipt "$output_root/MMSEQS_SEARCH_COMPLETE.json" \
  --output-root "$output_root" \
  --maximum-sequences "$maximum_sequences" \
  > "$output_root/finalize.stdout" 2> "$output_root/finalize.stderr"
