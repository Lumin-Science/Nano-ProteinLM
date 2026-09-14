#!/usr/bin/env bash
set -euo pipefail
root=/scratch/muchenli/Nano-Protein-LM-nibi-atlas-setting3-b1024-20260914
[[ "$(hostname -s):${SLURM_JOB_ID:?}" == g27:12162637 ]]
mode=${1:?pilot or full}
case "$mode" in pilot) records=2000000;; full) records=120000000;; *) exit 2;; esac
py=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/runtime/.venv/bin/python
export PYTHONPATH="$root/source/src:$root/prep-libs" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out="$SLURM_TMPDIR/nano-atlas-20260914/$mode"
printf '%s\n' "$out" > "$root/$mode-preparation-root.txt"
exec "$py" -m nanoprotein.atlas_data --output-root "$out" \
  --validation-root "$SLURM_TMPDIR/nano-nibi-setting3-stage2-20260911" \
  --records "$records" --workers 8
