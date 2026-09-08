#!/usr/bin/env bash
set -euo pipefail
checkpoint=${1:?checkpoint}
out=${2:?evaluation output directory}
root=/scratch/muchenli/Nano-Protein-LM-nibi-b2048-100k-eval10k-20260908
repo="$root-run"
py="$root/runtime/.venv/bin/python"
report="$repo/reports/nibi-baseline-b2048-100k-eval10k-20260908"
cd "$repo"
export CONTACT_ROOT=${CONTACT_ROOT:-$root/contact-data}
IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
[[ ${#gpu_ids[@]} -ge 4 ]]
mkdir -p "$out"
test ! -e "$out/RESULT_VERIFIED.json"
"$py" - "$checkpoint" "$out" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path
checkpoint,out=map(Path,sys.argv[1:]); packet=torch.load(checkpoint,map_location='cpu',weights_only=False)
with checkpoint.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
assert packet['parameter_count']==170671168 and packet['world_size'] in (4,8)
assert packet['data_manifest_sha256']=='43675d51421066ce8c5f68427886d57980e808c53c5bb1641de90cb74dda39ab'
assert all(torch.isfinite(value).all().item() for value in packet['model'].values())
receipt=dict(status='passed',scope='checkpoint_at_step',full_training_complete=packet['optimizer_step']>=packet['train_config']['max_steps'],
             checkpoint_sha256=digest,optimizer_steps=packet['optimizer_step'],sequences_seen=packet['sequences_seen'])
(out/'TRAINING_VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "$py" -m nano_protein.evaluate --checkpoint "$checkpoint" --data-root "$DATA_ROOT" \
  --output-root "$out/eval-validation" --validation-batches 256 --validation-batch-size 16 --validation-context 512 > "$out/validation.log" 2>&1
export OUTPUT_ROOT="$out" CHECKPOINT="$checkpoint" EVAL_OUTPUT_ROOT="$out/eval-p-at-l"
export EXTERNAL_SRC="$repo/reports/fir-r02-rope10k-100k-20260906/launch/contact-evaluator-src"
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
