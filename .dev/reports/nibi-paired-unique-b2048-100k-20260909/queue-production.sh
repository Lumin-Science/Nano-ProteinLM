#!/bin/bash
set -euo pipefail
export PATH="/opt/software/slurm/25.11.7p/bin:/usr/bin:/bin:$PATH"
root=/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909
exec 9>"$root/production.lock"
flock -n 9 || exit 0
trap 'rc=$?; if ((rc!=0)); then printf "exit_code=%s\n" "$rc" > "$root/PRODUCTION_QUEUE_FAILED.txt"; fi' EXIT
while [[ ! -e "$root/DATA_READY.json" || ! -e "$root/QUALIFICATION_PASSED.json" ]]; do
 test ! -e "$root/trial/baseline/FAILED.txt"
 test ! -e "$root/trial/setting3/FAILED.txt"
 [[ "$(squeue -h -j 12162637 -o '%T')" == RUNNING ]]
 sleep 30
done
/usr/bin/python3 - "$root" <<'CHECK'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);commit=(r/'SOURCE_COMMIT.txt').read_text().strip()
assert commit=='bc54124abe193623abf64b44e65b881cded65f48'
for name in ('DATA_READY.json','QUALIFICATION_PASSED.json'):
 p=json.loads((r/name).read_text());assert p['status']=='passed' and p['source_commit']==commit
assert not (r/'full').exists()
CHECK
date -u +%Y-%m-%dT%H:%M:%SZ > "$root/PRODUCTION_DISPATCHED_UTC.txt"
srun --jobid=12162637 --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 --gres=gpu:8 --nodelist=g27 /bin/bash --login "$root-run/.dev/reports/nibi-paired-unique-b2048-100k-20260909/run-pair.sh" full > "$root/production-driver.log" 2>&1
test -e "$root/full/baseline/COMPLETE.txt"
test -e "$root/full/setting3/COMPLETE.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$root/PAIR_COMPLETE.txt"
