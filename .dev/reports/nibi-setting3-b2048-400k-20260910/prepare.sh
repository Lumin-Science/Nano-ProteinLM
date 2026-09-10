#!/bin/bash
set -euo pipefail
root=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-400k-20260910
printf 'NODE=%s\nSLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\n' "$(hostname -s)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$root/prepare-identity.txt"
export PYTHONPATH=/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909-run/src
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset HF_HUB_OFFLINE
exec /scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/runtime/.venv/bin/python "$root/prepare.py"
