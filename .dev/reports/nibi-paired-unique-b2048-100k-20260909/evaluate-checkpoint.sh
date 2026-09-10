#!/usr/bin/env bash
set -euo pipefail
checkpoint=${1:?checkpoint}
out=${2:?evaluation output directory}
root=${PAIR_ROOT:?}
repo=${PAIR_REPO:?}
py=${TRAIN_PYTHON:?}
report="$repo/.dev/reports/nibi-paired-unique-b2048-100k-20260909"
cd "$repo"
export CONTACT_ROOT=${CONTACT_ROOT:-$root/contact-data}
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
[[ ${#gpu_ids[@]} -ge 4 ]]
mkdir -p "$out"
test ! -e "$out/RESULT_VERIFIED.json"
"$py" - "$checkpoint" "$out" <<'PY'
import hashlib,json,sys,torch,os
from pathlib import Path
checkpoint,out=map(Path,sys.argv[1:]); packet=torch.load(checkpoint,map_location='cpu',weights_only=False)
with checkpoint.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
assert packet['parameter_count'] in (170559856,170671168) and packet['world_size']==4
assert packet['data_manifest_sha256']==hashlib.sha256((Path(os.environ['DATA_ROOT'])/'manifest.json').read_bytes()).hexdigest()
assert packet['train_config']['data_resampling']=='error'
assert all(s['epoch']==0 for r in packet['runtime_states'] for b in r['batchers'].values() for s in b['samplers'].values())
assert all(torch.isfinite(value).all().item() for value in packet['model'].values())
receipt=dict(status='passed',scope='checkpoint_at_step',full_training_complete=packet['optimizer_step']>=packet['train_config']['max_steps'],
             checkpoint_sha256=digest,optimizer_steps=packet['optimizer_step'],sequences_seen=packet['sequences_seen'])
(out/'TRAINING_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "$py" -m nanoprotein.evaluate --checkpoint "$checkpoint" --data-root "$DATA_ROOT" \
  --output-root "$out/eval-validation" --validation-batches 256 --validation-batch-size 16 --validation-context 512 > "$out/validation.log" 2>&1
export OUTPUT_ROOT="$out" CHECKPOINT="$checkpoint" EVAL_OUTPUT_ROOT="$out/eval-p-at-l"
export EXTERNAL_SRC="${EXTERNAL_SRC:?}"
export EVAL_GPUS="${gpu_ids[0]},${gpu_ids[1]},${gpu_ids[2]},${gpu_ids[3]}" CONTACT_CHAINS=20775 CONTACT_SHARDS=16 OMP_NUM_THREADS=1
bash "$report/evaluate-p-at-l.sh" > "$out/contact.log" 2>&1
"$py" "$report/verify_contact.py" "$out" > "$out/contact-verify.log" 2>&1
"$py" - "$out" <<'PY'
import json,math,sys
from pathlib import Path
out=Path(sys.argv[1]); receipt=json.loads((out/'TRAINING_VERIFIED.json').read_text())
v=json.loads((out/'eval-validation/VALIDATION_MLM.json').read_text())
e=json.loads((out/'eval-validation/EVALUATION.json').read_text())
p=json.loads((out/'eval-p-at-l/P_AT_L_UNCERTAINTY.json').read_text())
assert e['checkpoint_sha256']==p['checkpoint_sha256']==receipt['checkpoint_sha256']
assert v['sequences']==4096 and v['masked_residues']==139963 and math.isfinite(v['sequence_mean_nll'])
assert p['uncertainty']['unit_count']==20775 and p['uncertainty']['replicates']==5000
receipt.update(validation=v,p_at_l=p['p_at_l'],uncertainty=p['uncertainty'])
(out/'RESULT_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PY
