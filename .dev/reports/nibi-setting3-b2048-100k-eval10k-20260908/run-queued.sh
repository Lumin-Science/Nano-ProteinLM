#!/usr/bin/env bash
set -euo pipefail
root=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908
repo="$root-run"
report="$repo/reports/nibi-setting3-b2048-100k-eval10k-20260908"
py="$root/runtime/.venv/bin/python"
[[ "$(hostname -s):$SLURM_JOB_ID" == g27:12162637 ]]
cd "$repo"
state() { "$py" "$report/queue_state.py" "$root" "$1" "$2"; }
printf 'NODE=%s\nSLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\n' "$(hostname -s)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" > "$root/compute-identity.txt"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
state verifying_predecessor "Checking the baseline's final persistent checkpoint and all ten completed evaluation artifacts."
"$py" "$report/verify_predecessor.py" --deep > "$root/BASELINE_READY.json"
state qualifying_trial "Testing Setting 3 on eight GPUs: 20 steps, full evaluation at step 10, then continued training."
bash "$report/launch.sh" trial
state qualifying_resume "Testing full Muon/AdamW checkpoint continuation from eight to four GPUs at unchanged global batch 2048."
bash "$report/launch.sh" resume4
test -e "$root/QUALIFICATION_PASSED.json"
state training "Qualification passed; starting Setting 3 from scratch for 100k steps with evaluation every 10k."
bash "$report/launch.sh" full
