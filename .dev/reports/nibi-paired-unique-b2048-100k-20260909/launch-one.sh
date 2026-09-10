#!/bin/bash
set -euo pipefail
recipe=${1:?baseline or setting3}
mode=${2:?trial or full}
case "$recipe" in baseline) export CUDA_VISIBLE_DEVICES=0,1,2,3;; setting3) export CUDA_VISIBLE_DEVICES=4,5,6,7;; *) exit 2;; esac
case "$mode" in trial|full) ;; *) exit 2;; esac
: "${PAIR_ROOT:?}" "${PAIR_REPO:?}" "${TRAIN_PYTHON:?}" "${EXTERNAL_SRC:?}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$PAIR_REPO/src:$EXTERNAL_SRC" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HOME=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/hf-cache HF_HUB_OFFLINE=1
export CONTACT_ROOT="$SLURM_TMPDIR/nano-contact-c135bc80"
report="$PAIR_REPO/.dev/reports/nibi-paired-unique-b2048-100k-20260909"
out="$PAIR_ROOT/$mode/$recipe"
test ! -e "$out"
mkdir -p "$out"
trap 'rc=$?; if ((rc!=0)); then printf "exit_code=%s\n" "$rc" > "$out/FAILED.txt"; fi' EXIT
cd "$PAIR_REPO"
commit=$(cat "$PAIR_ROOT/SOURCE_COMMIT.txt")
[[ "$(git rev-parse HEAD)" == "$commit" && -z "$(git status --porcelain)" ]]
if [[ "$mode" == trial ]]; then
  export DATA_ROOT="$SLURM_TMPDIR/nano-esmc-data-bd38448d"
else
  export DATA_ROOT=$(cat "$PAIR_ROOT/PRODUCTION_DATA_ROOT.txt")
  test -f "$PAIR_ROOT/QUALIFICATION_PASSED.json"
  test -f "$PAIR_ROOT/DATA_READY.json"
fi
printf 'NODE=%s\nSLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\nCUDA_VISIBLE_DEVICES=%s\n' "$(hostname -s)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$CUDA_VISIBLE_DEVICES" > "$out/launch-identity.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/launch-started-utc.txt"
"$TRAIN_PYTHON" - "$PAIR_REPO" "$PAIR_ROOT" "$out" "$recipe" "$mode" "$commit" <<'PY'
import json,os,sys,yaml
from pathlib import Path
from nanoprotein.data_budget import data_coverage
repo,root,out=map(Path,sys.argv[1:4]);recipe,mode,commit=sys.argv[4:]
c=yaml.safe_load((repo/f'configs/esmc-171m-{recipe}-nibi-unique-b2048-100k.yaml').read_text())
if mode=='trial':c.update(max_steps=20,schedule_steps=20,warmup_steps=5,log_interval=1,checkpoint_interval=10,periodic_evaluation_interval=10,walltime_seconds=1800)
else:
 gate=json.loads((root/'QUALIFICATION_PASSED.json').read_text());assert gate['status']=='passed' and gate['source_commit']==commit
 ready=json.loads((root/'DATA_READY.json').read_text());assert ready['status']=='passed'
data=Path(os.environ['DATA_ROOT']);manifest=json.loads((data/'manifest.json').read_text())
coverage=data_coverage(c,manifest,world_size=4);assert coverage['status']=='passed'
(out/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
PY
"$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 -m nanoprotein.train \
  --config "$out/config.yaml" --data-root "$DATA_ROOT" --output-root "$out" > "$out/train.log" 2>&1
"$TRAIN_PYTHON" "$report/verify_training.py" "$out" "$commit" > "$out/verify.log" 2>&1
if [[ "$mode" == full ]]; then
  durable="/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-paired-unique-b2048-100k-20260909/$recipe"
  mkdir -p "$durable"
  test ! -e "$durable/checkpoint-final.pt"
  cp "$out/checkpoint-final.pt" "$durable/checkpoint-final.pt.partial"
  "$TRAIN_PYTHON" - "$out" "$durable" <<'PY'
import hashlib,json,sys
from pathlib import Path
out,dst=map(Path,sys.argv[1:]);p=dst/'checkpoint-final.pt.partial'
with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest==json.loads((out/'TRAINING_VERIFIED.json').read_text())['checkpoint_sha256']
p.replace(dst/'checkpoint-final.pt');(dst/'CHECKPOINT_SHA256.txt').write_text(digest+'  checkpoint-final.pt\n')
PY
  cp "$out/config.yaml" "$out/TRAINING_VERIFIED.json" "$out/run_contract.json" "$out/DATA_COVERAGE.json" "$PAIR_ROOT/SOURCE_COMMIT.txt" "$durable/"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$out/CHECKPOINT_PRESERVED.txt"
  bash "$report/evaluate-checkpoint.sh" "$out/checkpoint-final.pt" "$out/evaluations/step-100000" > "$out/final-evaluation.log" 2>&1
  rsync -a --exclude='components/' --exclude='*.log' --exclude='*.stdout' --exclude='*.stderr' "$out/evaluations/" "$durable/evaluations/"
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/COMPLETE.txt"
