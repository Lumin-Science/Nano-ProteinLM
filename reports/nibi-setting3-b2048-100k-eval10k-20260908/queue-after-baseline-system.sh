#!/bin/bash
# Recovery controller: metadata polling must survive a login-node CVMFS outage.
# Lightweight metadata polling only; all hashing/training/evaluation runs in Slurm.
set -euo pipefail
export PATH="/usr/bin:/bin:/opt/software/slurm/25.11.7p/bin:$PATH"
py=/usr/bin/python3
root=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908
repo="$root-run"
report="$repo/reports/nibi-setting3-b2048-100k-eval10k-20260908"
baseline=/scratch/muchenli/Nano-Protein-LM-nibi-b2048-100k-eval10k-20260908/full
exec 9>"$root/queue.lock"
flock -n 9 || exit 0
test ! -e "$root/QUEUE_DRIVER_PID"
printf '%s\n' "$$" > "$root/QUEUE_DRIVER_PID"
date -u +%Y-%m-%dT%H:%M:%SZ > "$root/QUEUE_STARTED_UTC.txt"
state() { "$py" "$report/queue_state.py" "$root" "$1" "$2"; }
trap 'rc=$?; if ((rc!=0)); then printf "exit_code=%s\n" "$rc" > "$root/QUEUE_FAILED.txt"; state failed "Queue exited with code $rc; inspect queue.log. No automatic retry or cancellation."; fi' EXIT
cd "$repo"
# Check the small frozen launcher manifest here. Full clean Git/source checks
# remain mandatory in launch.sh on the compute node before any training.
"$py" - "$root" "$report" <<'PYTHON'
import hashlib, sys
from pathlib import Path
root, report = map(Path, sys.argv[1:])
assert (root / "SOURCE_COMMIT.txt").read_text().strip() == "c76a07998987a4746d4bffc89d878a758cd735cb"
for line in (report / "ARTIFACTS.sha256").read_text().splitlines():
    digest, relative = line.split("  ", 1)
    assert hashlib.sha256((report / relative).read_bytes()).hexdigest() == digest
config = report.parents[1] / "configs/esmc-171m-setting3-nibi-fa3-b2048-stage1-100k.yaml"
assert hashlib.sha256(config.read_bytes()).hexdigest() == "76a6f26e7ef6d381c8777a0b772c79741f5d25d5f6c84e888839521dfea882ac"
PYTHON
while true; do
  [[ "$(squeue -h -j 12162637 -o '%T')" == RUNNING ]]
  test ! -e "$baseline/FAILED.txt"
  if [[ ! -e "$baseline/COMPLETE.txt" || ! -e "$baseline/CHECKPOINT_PRESERVED.txt" ]]; then
    state waiting_for_baseline "Waiting for baseline step 12162637.11: 100k training, all ten evaluations and persistent checkpoint."
    sleep 60
    continue
  fi
  "$py" "$report/verify_predecessor.py" > "$root/BASELINE_READY_METADATA.json"
  # A retained batch/extern step owns the user's allocation. Any other step
  # could still be using the GPUs; wait for it without altering that workload.
  active=$(squeue --steps -h -j 12162637 -o '%i' | awk '$0 !~ /\.(batch|extern)$/ {print}')
  if [[ -n "$active" ]]; then
    state waiting_for_steps "Baseline results are complete; waiting for existing compute steps to exit: $active"
    sleep 60
    continue
  fi
  break
done
state dispatching "Baseline completed and no other compute steps remain; requesting the existing eight-GPU allocation."
srun --jobid=12162637 --overlap --nodes=1 --ntasks=1 --cpus-per-task=64 \
  --gres=gpu:8 --nodelist=g27 /bin/bash --login "$report/run-queued.sh" > "$root/compute-driver.log" 2>&1
test -e "$root/full/COMPLETE.txt"
state complete "Setting 3 completed 100k steps, all ten evaluations and persistent checkpoint preservation."
date -u +%Y-%m-%dT%H:%M:%SZ > "$root/QUEUE_COMPLETE.txt"

