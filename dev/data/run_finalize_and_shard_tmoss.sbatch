#!/usr/bin/env bash
#SBATCH --job-name=protein-shards-v2
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/muchenli/projects/nano-protein-data-build-20260823/shard-%j.out
#SBATCH --error=/home/muchenli/projects/nano-protein-data-build-20260823/shard-%j.err

set -euo pipefail

export MALLOC_ARENA_MAX=4
uv_bin=/home/muchenli/.local/bin/uv
pipeline=/home/muchenli/projects/nano-protein-data-build-20260823/process_full_corpus.py
legacy_screen=/home/muchenli/datasets/evaluation-decontamination/mmseqs-v1
evaluation_root=/home/muchenli/datasets/evaluation-decontamination/v2-20260823
delta_root=/home/muchenli/datasets/evaluation-decontamination/q9-delta-mmseqs-v2-20260823
screen_root=/home/muchenli/datasets/evaluation-decontamination/mmseqs-v2-20260823
cluster_root=/data/muchenli/esmc-open-step9-clusters-v1/clusters
release_root=/home/muchenli/datasets/luminbench-nano-esmc-full-open-v2-20260823

test -f "$delta_root/MMSEQS_DELTA_SEARCH_COMPLETE.json"
test ! -e "$screen_root"
test ! -e "$release_root"

"$uv_bin" run --with pyarrow==21.0.0 python "$pipeline" finalize-screen \
  --evaluation-root "$evaluation_root" \
  --parent-receipt "$legacy_screen/HOMOLOGY_EXCLUSION_VERIFIED.json" \
  --parent-exclusions "$legacy_screen/homology_excluded_all_splits.txt" \
  --delta-root "$delta_root" \
  --cluster-root "$cluster_root" \
  --output "$screen_root"

"$uv_bin" run --with pyarrow==21.0.0 python "$pipeline" shard \
  --cluster-root "$cluster_root" \
  --screen-root "$screen_root" \
  --evaluation-root "$evaluation_root" \
  --output "$release_root" \
  --validation-per-source 4096 \
  --shard-residues 268435456

"$uv_bin" run --with pyarrow==21.0.0 python "$pipeline" verify-release \
  --root "$release_root" \
  --screen-root "$screen_root" \
  --evaluation-root "$evaluation_root"
