#!/bin/bash
set -euo pipefail
mode=${1:?trial or full}
export PAIR_ROOT=/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909
export PAIR_REPO="$PAIR_ROOT-run"
export TRAIN_PYTHON=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/runtime/.venv/bin/python
export EXTERNAL_SRC=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908-run/reports/fir-r02-rope10k-100k-20260906/launch/contact-evaluator-src
[[ "$(hostname -s):$SLURM_JOB_ID" == g27:12162637 ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
report="$PAIR_REPO/.dev/reports/nibi-paired-unique-b2048-100k-20260909"
mkdir -p "$PAIR_ROOT/$mode"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv > "$PAIR_ROOT/$mode/gpu-map.csv"
printf 'NODE=%s\nSLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\n' "$(hostname -s)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$PAIR_ROOT/$mode/pair-identity.txt"
bash "$report/launch-one.sh" baseline "$mode" > "$PAIR_ROOT/$mode/baseline-driver.log" 2>&1 &
baseline_pid=$!
bash "$report/launch-one.sh" setting3 "$mode" > "$PAIR_ROOT/$mode/setting3-driver.log" 2>&1 &
setting3_pid=$!
status=0
wait "$baseline_pid" || status=1
wait "$setting3_pid" || status=1
if ((status)); then exit "$status"; fi
if [[ "$mode" == trial ]]; then
  "$TRAIN_PYTHON" - "$PAIR_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);results={}
for name in ('baseline','setting3'):
 out=root/'trial'/name
 r=json.loads((out/'TRAINING_VERIFIED.json').read_text())
 e=json.loads((out/'evaluations/step-000010/RESULT_VERIFIED.json').read_text())
 assert r['status']=='passed' and r['optimizer_steps']==20 and r['no_resampling_verified']
 assert e['optimizer_steps']==10 and e['uncertainty']['unit_count']==20775
 results[name]=r
(root/'QUALIFICATION_PASSED.json').write_text(json.dumps({'status':'passed','source_commit':(root/'SOURCE_COMMIT.txt').read_text().strip(),'trials':results,'both_continued_after_full_step10_evaluation':True},indent=2)+'\n')
PY
fi
