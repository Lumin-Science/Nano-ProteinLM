#!/usr/bin/env bash
set -euo pipefail
mode=${1:?trial or full}
method=${2:?method}
case "$method" in r02_rope10k|r04_batchbalance|r10_sqrtloss|r29_tied) ;; *) exit 2 ;; esac
case "$mode" in trial|full) ;; *) exit 2 ;; esac
root=/scratch/muchenli/Nano-Protein-LM-r02-rope10k-100k-20260906
repo="$root-run"
py=/scratch/muchenli/AutoResearch_ESMC/.venv/bin/python
case "$(hostname -s):$SLURM_JOB_ID" in fc10212:58303724|fc10111:58303658) ;; *) exit 2 ;; esac
if [[ "$mode" == trial ]]; then
  [[ "$(hostname -s)" == fc10212 ]]
  out="$root/trials/$method"
else
  test -f "$root/ALL_TRIALS_PASSED.json"
  if [[ "$method" == r02_rope10k ]]; then
    [[ "$(hostname -s):$SLURM_JOB_ID" == fc10111:58303658 ]]
    [[ "$(date -u +%s)" -ge 1788768000 ]]
  else
    [[ "$(hostname -s):$SLURM_JOB_ID" == fc10212:58303724 ]]
  fi
  out="$root/full/$method"
  remaining=$(squeue -h -j "$SLURM_JOB_ID" -o '%L')
  "$py" - "$remaining" <<'PY'
import sys
s=sys.argv[1].strip(); days=0
if '-' in s: d,s=s.split('-');days=int(d)
parts=[int(x) for x in s.split(':')]
if len(parts)==2: parts=[0]+parts
remaining=days*86400+parts[0]*3600+parts[1]*60+parts[2]
assert remaining>=58500, f'Allocation has only {remaining}s left; need 16h plus evaluation margin'
PY
fi
cd "$repo"
[[ "$(git rev-parse HEAD)" == 253c3ea442f2a1657eeb0f3ce2127ac0b1adfb25 ]]
[[ -z "$(git status --porcelain)" ]]
test ! -e "$out"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
mkdir -p "$out"
export PYTHONPATH="$repo:$root/contact-evaluator-src"
export HF_HOME=/scratch/muchenli/Nano-Protein-LM-fir-fa3-20260906/.exps/hf-cache
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_ROOT="$SLURM_TMPDIR/nano-esmc-data-bd38448d"
export CONTACT_ROOT="$SLURM_TMPDIR/nano-contact-c135bc80"
trap 'rc=$?; if ((rc!=0)); then printf "exit_code=%s\n" "$rc" > "$out/FAILED.txt"; fi' EXIT
printf 'NODE=%s\nSLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\nMODE=%s\nMETHOD=%s\n' "$(hostname -s)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$mode" "$method" > "$out/launch-identity.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/launch-started-utc.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv > "$out/HARDWARE.csv"
mkdir -p "$DATA_ROOT"
rsync -a /scratch/muchenli/Nano-Protein-LM-fir-fa3-20260906/.exps/data/training-samples-5376000/ "$DATA_ROOT/"
"$py" - "$repo" "$root" "$out" "$method" "$mode" <<'PY'
import hashlib,json,os,sys,torch,yaml
from pathlib import Path
repo,root,out=map(Path,sys.argv[1:4]);method,mode=sys.argv[4:]
manifest=json.loads((repo/'configs/program2_h100_100k/manifest.json').read_text())
entry=next(x for x in manifest['runs'] if x['method']==method)
src=repo/entry['config'];assert hashlib.sha256(src.read_bytes()).hexdigest()==entry['config_sha256']
config=yaml.safe_load(src.read_text())
if mode=='trial': config.update(max_steps=200,schedule_steps=200,warmup_steps=50,walltime_seconds=1200)
else:
    gate=json.loads((root/'ALL_TRIALS_PASSED.json').read_text())
    assert gate['status']=='passed' and len(gate['trials'])==4
    assert gate['production_config_sha256'][method]==entry['config_sha256']
(out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
assert torch.cuda.device_count()==4 and all(torch.cuda.mem_get_info(i)[0]>64*2**30 for i in range(4))
data=Path(os.environ['DATA_ROOT']);receipt=json.loads((data/'CORPUS_VERIFICATION.json').read_text())
for source,records in receipt['sources'].items():
    for split in ('train','validation'):
        for filename,key in [('tokens.bin','tokens_sha256'),('index.npy','index_sha256')]:
            with (data/source/split/filename).open('rb') as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==records[split][key]
assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()=='43675d51421066ce8c5f68427886d57980e808c53c5bb1641de90cb74dda39ab'
(out/'DATA_VERIFIED.json').write_text(json.dumps({'all_data_hashes_verified':True,'source_config_sha256':entry['config_sha256'],'data_root':str(data)},indent=2)+'\n')
print('CONFIG_AND_DATA_VERIFIED',method,mode,flush=True)
PY
"$py" scripts/check_environment.py --require-gpus 4 --attention-backend flash3 --output "$out/ENVIRONMENT.json" > "$out/environment.log" 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/training-started-utc.txt"
"$py" -m torch.distributed.run --standalone --nproc-per-node=4 -m nano_protein.train \
  --config "$out/config.yaml" --data-root "$DATA_ROOT" --output-root "$out" > "$out/train.log" 2>&1
"$py" "$root/verify_run.py" "$out" "$method" "$mode" > "$out/verify.log" 2>&1
if [[ "$mode" == trial ]]; then
  batches=8; batch_size=4
else
  batches=256; batch_size=16
fi
"$py" -m nano_protein.evaluate --checkpoint "$out/checkpoint-final.pt" --data-root "$DATA_ROOT" \
  --output-root "$out/eval-validation" --validation-batches "$batches" --validation-batch-size "$batch_size" \
  --validation-context 512 > "$out/validation.log" 2>&1
"$py" - "$out" "$mode" <<'PY'
import json,math,sys
from pathlib import Path
p=Path(sys.argv[1]);v=json.loads((p/'eval-validation/VALIDATION_MLM.json').read_text())
e=json.loads((p/'eval-validation/EVALUATION.json').read_text());c=json.loads((p/'TRAINING_VERIFIED.json').read_text())
assert e['checkpoint_sha256']==c['checkpoint_sha256']
assert v['sequences']==(32 if sys.argv[2]=='trial' else 4096)
assert math.isfinite(v['sequence_mean_nll']) and math.isfinite(v['perplexity'])
print('MLM_VALIDATION_PASSED',v['sequence_mean_nll'],flush=True)
PY
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/training-and-validation-complete-utc.txt"
if [[ "$mode" == full ]]; then
  export OUTPUT_ROOT="$out" CHECKPOINT="$out/checkpoint-final.pt" EVAL_OUTPUT_ROOT="$out/eval-p-at-l"
  export EXTERNAL_SRC="$root/contact-evaluator-src" EVAL_GPUS=0,1,2,3 CONTACT_CHAINS=20775 CONTACT_SHARDS=16
  export OMP_NUM_THREADS=1
  mkdir -p "$CONTACT_ROOT"
  rsync -a /scratch/muchenli/AutoResearch_ESMC/contact-benchmark/v1/compact-dataset-v1/dataset/ "$CONTACT_ROOT/"
  bash "$root/evaluate-p-at-l.sh" > "$out/contact.log" 2>&1
  "$py" "$root/verify_contact.py" "$out" > "$out/contact-verify.log" 2>&1
  date -u +%Y-%m-%dT%H:%M:%SZ > "$out/contact-complete-utc.txt"
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/COMPLETE.txt"
printf 'RUN_COMPLETE %s %s\n' "$mode" "$method"
