#!/bin/bash
set -euo pipefail
root=/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909
repo="$root-run"
old=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908-run
printf 'NODE=%s\nSLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\n' "$(hostname -s)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$root/prepare-identity.txt"
test ! -e "$repo"
git clone --no-hardlinks "$old" "$repo"
git -C "$repo" fetch "$root/source.bundle" HEAD
git -C "$repo" checkout --detach FETCH_HEAD
[[ "$(git -C "$repo" rev-parse --short=7 HEAD)" == bc54124 ]]
git -C "$repo" rev-parse HEAD > "$root/SOURCE_COMMIT.txt"
export PYTHONPATH="$repo/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/runtime/.venv/bin/python "$root/prepare.py"
