#!/bin/bash
set -euo pipefail
root=/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909
printf 'NODE=%s\nSLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\n' "$(hostname -s)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$root/download-identity.txt"
export PYTHONPATH="$root/source-preview/src" PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE
/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/runtime/.venv/bin/python "$root/download.py"
