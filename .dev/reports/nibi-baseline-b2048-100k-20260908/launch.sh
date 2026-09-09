#!/usr/bin/env bash
set -euo pipefail
mode=${1:?qualification or full}
case "$mode" in qualification|full) ;; *) exit 2 ;; esac
root=/scratch/muchenli/Nano-Protein-LM-nibi-b2048-100k-20260908
repo="$root-run"
py="$root/runtime/.venv/bin/python"
report="$repo/reports/nibi-baseline-b2048-100k-20260908"
durable=/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-baseline-b2048-100k-20260908
[[ "$(hostname -s):$SLURM_JOB_ID" == g27:12162637 ]]
expected_commit=$(cat "$root/SOURCE_COMMIT.txt")
cd "$repo"
[[ "$(git rev-parse HEAD)" == "$expected_commit" ]]
[[ -z "$(git status --porcelain)" ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
export PYTHONPATH="$repo:$repo/reports/fir-r02-rope10k-100k-20260906/launch/contact-evaluator-src"
export HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_ROOT="$SLURM_TMPDIR/nano-esmc-data-bd38448d" CONTACT_ROOT="$SLURM_TMPDIR/nano-contact-c135bc80"
mkdir -p "$DATA_ROOT" "$CONTACT_ROOT"
rsync -a "$root/data/" "$DATA_ROOT/"
rsync -a "$root/contact-data/" "$CONTACT_ROOT/"
"$py" - "$DATA_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
data = Path(sys.argv[1]); receipt = json.loads((data/'CORPUS_VERIFICATION.json').read_text())
for source, records in receipt['sources'].items():
    for split in ('train', 'validation'):
        for filename, key in [('tokens.bin', 'tokens_sha256'), ('index.npy', 'index_sha256')]:
            with (data/source/split/filename).open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == records[split][key]
assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest() == '43675d51421066ce8c5f68427886d57980e808c53c5bb1641de90cb74dda39ab'
print('ALL_DATA_HASHES_VERIFIED', flush=True)
PY
"$py" scripts/check_environment.py --require-gpus 8 --attention-backend flash3 --output "$root/ENVIRONMENT-$mode.json"
run_train() {
  local name=$1 world=$2 endpoint=$3 checkpoint=${4:-}
  local out="$root/$name"
  test ! -e "$out"
  mkdir -p "$out"
  printf 'NODE=%s\nSLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\nMODE=%s\n' "$(hostname -s)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$mode" > "$out/launch-identity.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$out/launch-started-utc.txt"
  "$py" - "$repo" "$out" "$world" "$endpoint" "$mode" <<'PY'
import sys, yaml
from pathlib import Path
repo, out = map(Path, sys.argv[1:3]); world, endpoint = map(int, sys.argv[3:5])
config = yaml.safe_load((repo/'configs/esmc-171m-default-nibi-fa3-b2048-stage1-100k.yaml').read_text())
config.update(expected_world_size=world, max_steps=endpoint, schedule_steps=endpoint)
config['stages'][0]['gradient_accumulation'] = 2048 // (64*world)
if sys.argv[5] == 'qualification':
    config.update(warmup_steps=50, checkpoint_interval=100, walltime_seconds=1800)
(out/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
PY
  local resume_args=()
  if [[ -n "$checkpoint" ]]; then resume_args=(--resume "$checkpoint"); fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$out/training-started-utc.txt"
  "$py" -m torch.distributed.run --standalone --nproc-per-node="$world" -m nano_protein.train \
    --config "$out/config.yaml" --data-root "$DATA_ROOT" --output-root "$out" "${resume_args[@]}" > "$out/train.log" 2>&1
  "$py" "$report/verify_training.py" "$out" "$expected_commit" > "$out/verify.log" 2>&1
}
if [[ "$mode" == qualification ]]; then
  trap 'rc=$?; if ((rc!=0)); then printf "exit_code=%s\n" "$rc" > "$root/QUALIFICATION_FAILED.txt"; fi' EXIT
  run_train trial-8gpu 8 200
  run_train resume-8gpu 8 200 "$root/trial-8gpu/checkpoint-latest.pt"
  "$py" "$report/compare_resume.py" "$root/trial-8gpu/checkpoint-final.pt" "$root/resume-8gpu/checkpoint-final.pt" "$root/SAME_LAYOUT_RESUME.json"
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  run_train resume-4gpu 4 220 "$root/trial-8gpu/checkpoint-final.pt"
  "$py" -m nano_protein.evaluate --checkpoint "$root/resume-4gpu/checkpoint-final.pt" --data-root "$DATA_ROOT" \
    --output-root "$root/resume-4gpu/eval-validation" --validation-batches 8 --validation-batch-size 4 --validation-context 512 > "$root/resume-4gpu/validation.log" 2>&1
  "$py" - "$root" "$expected_commit" <<'PY'
import json, math, sys
from pathlib import Path
root=Path(sys.argv[1]); results={name:json.loads((root/name/'TRAINING_VERIFIED.json').read_text()) for name in ('trial-8gpu','resume-8gpu','resume-4gpu')}
assert results['resume-8gpu']['resume']['data_mode'] == 'restore_rank_rng_and_sampler'
assert results['resume-4gpu']['resume']['data_mode'] == 'new_deterministic_stream_for_changed_gpu_layout'
v=json.loads((root/'resume-4gpu/eval-validation/VALIDATION_MLM.json').read_text())
assert v['sequences']==32 and math.isfinite(v['sequence_mean_nll'])
(root/'QUALIFICATION_PASSED.json').write_text(json.dumps(dict(status='passed', source_commit=sys.argv[2], runs=results, validation=v),indent=2)+'\n')
PY
else
  test -f "$root/QUALIFICATION_PASSED.json"
  "$py" - "$root" "$expected_commit" "$(squeue -h -j "$SLURM_JOB_ID" -o '%L')" <<'PY'
import json, sys
from pathlib import Path
assert json.loads((Path(sys.argv[1])/'QUALIFICATION_PASSED.json').read_text())['source_commit']==sys.argv[2]
s=sys.argv[3].strip(); days=0
if '-' in s: d,s=s.split('-'); days=int(d)
p=list(map(int,s.split(':')))
if len(p)==2: p=[0]+p
assert days*86400+p[0]*3600+p[1]*60+p[2] >= 87300
PY
  trap 'rc=$?; if ((rc!=0)); then printf "exit_code=%s\n" "$rc" > "$root/FULL_FAILED.txt"; fi' EXIT
  run_train full 8 100000
  out="$root/full"
  mkdir -p "$durable"
  test ! -e "$durable/checkpoint-final.pt"
  cp "$out/checkpoint-final.pt" "$durable/checkpoint-final.pt.partial"
  "$py" - "$out" "$durable" <<'PY'
import hashlib,json,sys
from pathlib import Path
out,dst=map(Path,sys.argv[1:]); p=dst/'checkpoint-final.pt.partial'
with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest==json.loads((out/'TRAINING_VERIFIED.json').read_text())['checkpoint_sha256']
p.replace(dst/'checkpoint-final.pt')
(dst/'CHECKPOINT_SHA256.txt').write_text(digest+'  checkpoint-final.pt\n')
PY
  cp "$out/config.yaml" "$out/TRAINING_VERIFIED.json" "$out/run_contract.json" "$root/SOURCE_COMMIT.txt" "$durable/"
  cp "$repo/docs/checkpoint-resume.md" "$durable/README-resume.md"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$out/CHECKPOINT_PRESERVED.txt"
  "$py" -m nano_protein.evaluate --checkpoint "$out/checkpoint-final.pt" --data-root "$DATA_ROOT" \
    --output-root "$out/eval-validation" --validation-batches 256 --validation-batch-size 16 --validation-context 512 > "$out/validation.log" 2>&1
  export OUTPUT_ROOT="$out" CHECKPOINT="$out/checkpoint-final.pt" EVAL_OUTPUT_ROOT="$out/eval-p-at-l"
  export EXTERNAL_SRC="$repo/reports/fir-r02-rope10k-100k-20260906/launch/contact-evaluator-src"
  export EVAL_GPUS=0,1,2,3 CONTACT_CHAINS=20775 CONTACT_SHARDS=16 OMP_NUM_THREADS=1
  bash "$root/evaluate-p-at-l.sh" > "$out/contact.log" 2>&1
  "$py" "$root/verify_contact.py" "$out" > "$out/contact-verify.log" 2>&1
  date -u +%Y-%m-%dT%H:%M:%SZ > "$out/COMPLETE.txt"
fi
echo "$mode completed successfully"
