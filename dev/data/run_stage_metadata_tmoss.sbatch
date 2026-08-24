#!/usr/bin/env bash
#SBATCH --job-name=protein-metadata-v2
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/muchenli/projects/nano-protein-data-build-20260823/metadata-%j.out
#SBATCH --error=/home/muchenli/projects/nano-protein-data-build-20260823/metadata-%j.err

set -euo pipefail

build_root=/home/muchenli/projects/nano-protein-data-build-20260823
"/home/muchenli/.local/bin/uv" run --with pyarrow==21.0.0 python \
  "$build_root/process_full_corpus.py" stage-metadata \
  --release-root /home/muchenli/datasets/luminbench-nano-esmc-full-open-v2-20260823 \
  --template-root "$build_root/release-template" \
  --omg-manifest "$build_root/omg_upstream_shards.tsv" \
  --screen-root /home/muchenli/datasets/evaluation-decontamination/mmseqs-v2-20260823 \
  --evaluation-root /home/muchenli/datasets/evaluation-decontamination/v2-20260823
