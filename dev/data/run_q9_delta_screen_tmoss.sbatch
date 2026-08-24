#!/usr/bin/env bash
#SBATCH --job-name=pcore-q9-screen
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/muchenli/projects/nano-protein-data-build-20260823/q9-screen-%j.out
#SBATCH --error=/home/muchenli/projects/nano-protein-data-build-20260823/q9-screen-%j.err

set -euo pipefail

uv_bin=/home/muchenli/.local/bin/uv
pipeline=/home/muchenli/projects/nano-protein-data-build-20260823/process_full_corpus.py
evaluation_root=/home/muchenli/datasets/evaluation-decontamination/v2-20260823
target_db_root=/home/muchenli/datasets/evaluation-decontamination/mmseqs-v1/db
output=/home/muchenli/datasets/evaluation-decontamination/q9-delta-mmseqs-v2-20260823
mmseqs=/home/muchenli/.local/mmseqs/17-b804f/bin/mmseqs

test -x "$uv_bin"
test -f "$pipeline"
test -s "$evaluation_root/evaluation_q9_delta.fasta"
test -x "$mmseqs"

# Source databases are serialized to bound peak target-index RAM on this 1 TB
# host. This does not drop queries, targets, or alignments.
"$uv_bin" run --with pyarrow==21.0.0 python "$pipeline" delta-screen \
  --query-fasta "$evaluation_root/evaluation_q9_delta.fasta" \
  --target-db-root "$target_db_root" \
  --output "$output" \
  --mmseqs "$mmseqs" \
  --threads 64
