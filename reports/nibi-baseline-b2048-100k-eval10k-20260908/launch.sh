#!/usr/bin/env bash
set -euo pipefail
mode=${1:?trial or full}
case "$mode" in trial|full) ;; *) exit 2 ;; esac
root=/scratch/muchenli/Nano-Protein-LM-nibi-b2048-100k-eval10k-20260908
repo="$root-run"
py="$root/runtime/.venv/bin/python"
report="$repo/reports/nibi-baseline-b2048-100k-eval10k-20260908"
durable=/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-baseline-b2048-100k-eval10k-20260908
[[ "$(hostname -s):$SLURM_JOB_ID" == g27:12162637 ]]
expected_commit=$(cat "$root/SOURCE_COMMIT.txt")
cd "$repo"
[[ "$(git rev-parse HEAD)" == "$expected_commit" && -z "$(git status --porcelain)" ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
export PYTHONPATH="$repo:$repo/reports/fir-r02-rope10k-100k-20260906/launch/contact-evaluator-src"
export HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_ROOT="$SLURM_TMPDIR/nano-esmc-data-bd38448d" CONTACT_ROOT="$SLURM_TMPDIR/nano-contact-c135bc80"
test -f "$DATA_ROOT/manifest.json"
test -d "$CONTACT_ROOT"
out="$root/$mode"
test ! -e "$out"
mkdir -p "$out"
trap 'rc=$?; if ((rc!=0)); then printf "exit_code=%s\n" "$rc" > "$out/FAILED.txt"; fi' EXIT
printf 'NODE=%s\nSLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\nMODE=%s\n' "$(hostname -s)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$mode" > "$out/launch-identity.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/launch-started-utc.txt"
"$py" - "$repo" "$root" "$out" "$mode" "$expected_commit" <<'PY'
import hashlib,json,os,sys,yaml
from pathlib import Path
repo,root,out=map(Path,sys.argv[1:4]);mode,commit=sys.argv[4:]
config=yaml.safe_load((repo/'configs/esmc-171m-default-nibi-fa3-b2048-stage1-100k.yaml').read_text())
if mode=='trial': config.update(max_steps=20,schedule_steps=20,warmup_steps=5,log_interval=1,checkpoint_interval=10,periodic_evaluation_interval=10,walltime_seconds=1800)
else:
    gate=json.loads((root/'PERIODIC_EVALUATION_TRIAL_PASSED.json').read_text())
    assert gate['source_commit']==commit and gate['status']=='passed'
(out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
data=Path(os.environ['DATA_ROOT']);receipt=json.loads((data/'CORPUS_VERIFICATION.json').read_text())
for source,records in receipt['sources'].items():
    for split in ('train','validation'):
        for filename,key in [('tokens.bin','tokens_sha256'),('index.npy','index_sha256')]:
            with (data/source/split/filename).open('rb') as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==records[split][key]
assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()=='43675d51421066ce8c5f68427886d57980e808c53c5bb1641de90cb74dda39ab'
PY
if [[ "$mode" == full ]]; then
  "$py" - "$(squeue -h -j "$SLURM_JOB_ID" -o '%L')" <<'PY'
import sys
s=sys.argv[1].strip();days=0
if '-' in s:d,s=s.split('-');days=int(d)
p=list(map(int,s.split(':')))
if len(p)==2:p=[0]+p
assert days*86400+p[0]*3600+p[1]*60+p[2]>=100000
PY
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/training-started-utc.txt"
"$py" -m torch.distributed.run --standalone --nproc-per-node=8 -m nano_protein.train \
  --config "$out/config.yaml" --data-root "$DATA_ROOT" --output-root "$out" > "$out/train.log" 2>&1
"$py" "$report/verify_training.py" "$out" "$expected_commit" > "$out/verify.log" 2>&1
if [[ "$mode" == trial ]]; then
  "$py" - "$root" "$out" "$expected_commit" <<'PY'
import json,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:3]);complete=json.loads((out/'TRAINING_COMPLETE.json').read_text())
evaluation=json.loads((out/'evaluations/step-000010/RESULT_VERIFIED.json').read_text())
execution=json.loads((out/'evaluations/step-000010/EVALUATION_RUN.json').read_text())
assert complete['optimizer_steps']==20 and complete['periodic_evaluation_seconds']>=execution['wall_seconds']
assert evaluation['optimizer_steps']==10 and evaluation['validation']['sequences']==4096
assert evaluation['uncertainty']['unit_count']==20775 and execution['status']=='passed'
(root/'PERIODIC_EVALUATION_TRIAL_PASSED.json').write_text(json.dumps(dict(status='passed',source_commit=sys.argv[3],full_evaluation_at_step10=evaluation,execution=execution,training_continued_to_step20=True),indent=2)+'\n')
PY
else
  mkdir -p "$durable"
  test ! -e "$durable/checkpoint-final.pt"
  cp "$out/checkpoint-final.pt" "$durable/checkpoint-final.pt.partial"
  "$py" - "$out" "$durable" <<'PY'
import hashlib,json,sys
from pathlib import Path
out,dst=map(Path,sys.argv[1:]);p=dst/'checkpoint-final.pt.partial'
with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest==json.loads((out/'TRAINING_VERIFIED.json').read_text())['checkpoint_sha256']
p.replace(dst/'checkpoint-final.pt');(dst/'CHECKPOINT_SHA256.txt').write_text(digest+'  checkpoint-final.pt\n')
PY
  cp "$out/config.yaml" "$out/TRAINING_VERIFIED.json" "$out/run_contract.json" "$root/SOURCE_COMMIT.txt" "$root/runtime-requirements.txt" "$durable/"
  cp "$repo/docs/checkpoint-resume.md" "$durable/README-resume.md"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$out/CHECKPOINT_PRESERVED.txt"
  bash "$report/evaluate-checkpoint.sh" "$out/checkpoint-final.pt" "$out/evaluations/step-100000" > "$out/final-evaluation.log" 2>&1
  rsync -a --exclude='components/' --exclude='*.log' --exclude='*.stdout' --exclude='*.stderr' "$out/evaluations/" "$durable/evaluations/"
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/COMPLETE.txt"
