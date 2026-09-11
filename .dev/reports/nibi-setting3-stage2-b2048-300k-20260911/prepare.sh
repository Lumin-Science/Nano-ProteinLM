#!/usr/bin/env bash
set -euo pipefail
root=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-stage2-b2048-300k-20260911
[[ "$(hostname -s):$SLURM_JOB_ID" == g27:12162637 ]]
printf 'NODE=%s\nSLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\n' "$(hostname -s)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$root/prepare-identity.txt"
export PYTHONPATH="$root/preparation-source/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset HF_HUB_OFFLINE
exec /scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/runtime/.venv/bin/python "$root/prepare.py"
