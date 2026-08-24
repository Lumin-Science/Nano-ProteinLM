#!/usr/bin/env bash
#SBATCH --job-name=pcore-q9-recover
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/muchenli/projects/nano-protein-data-build-20260823/q9-recovery-%j.out
#SBATCH --error=/home/muchenli/projects/nano-protein-data-build-20260823/q9-recovery-%j.err

set -euo pipefail

uv_bin=/home/muchenli/.local/bin/uv
pipeline=/home/muchenli/projects/nano-protein-data-build-20260823/process_full_corpus.py
evaluation_root=/home/muchenli/datasets/evaluation-decontamination/v2-20260823
target_db_root=/home/muchenli/datasets/evaluation-decontamination/mmseqs-v1/db
output=/home/muchenli/datasets/evaluation-decontamination/q9-delta-mmseqs-v2-20260823
mmseqs=/home/muchenli/.local/mmseqs/17-b804f/bin/mmseqs

if [[ -s "$output/MMSEQS_DELTA_SEARCH_COMPLETE.json" ]]; then
  echo "delta screen already complete; recovery is a verified no-op"
  exit 0
fi

# Completed source TSVs from the interrupted concurrent attempt are immutable
# inputs. Only missing sources are searched, serially, at the unchanged contract.
"$uv_bin" run --with pyarrow==21.0.0 python "$pipeline" delta-screen \
  --query-fasta "$evaluation_root/evaluation_q9_delta.fasta" \
  --target-db-root "$target_db_root" \
  --output "$output" \
  --mmseqs "$mmseqs" \
  --threads 64 \
  --resume
