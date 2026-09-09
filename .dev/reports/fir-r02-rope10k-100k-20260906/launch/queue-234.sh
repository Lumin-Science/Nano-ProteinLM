#!/usr/bin/env bash
set -euo pipefail
root=/scratch/muchenli/Nano-Protein-LM-r02-rope10k-100k-20260906
[[ "$(hostname -s):$SLURM_JOB_ID" == fc10212:58303724 ]]
test -f "$root/ALL_TRIALS_PASSED.json"
mkdir "$root/queue-234-started"
trap 'rc=$?; if ((rc!=0)); then printf "exit_code=%s\n" "$rc" > "$root/QUEUE_234_FAILED.txt"; fi' EXIT
for method in r04_batchbalance r10_sqrtloss r29_tied; do
  printf '%s\n' "$method" > "$root/QUEUE_CURRENT.txt"
  bash "$root/launch.sh" full "$method"
done
date -u +%Y-%m-%dT%H:%M:%SZ > "$root/QUEUE_234_COMPLETE.txt"
