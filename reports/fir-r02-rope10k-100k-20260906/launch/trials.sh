#!/usr/bin/env bash
set -euo pipefail
root=/scratch/muchenli/Nano-Protein-LM-r02-rope10k-100k-20260906
for method in r02_rope10k r04_batchbalance r10_sqrtloss r29_tied; do
  bash "$root/launch.sh" trial "$method"
done
/scratch/muchenli/AutoResearch_ESMC/.venv/bin/python - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
r=Path(sys.argv[1]);repo=Path(str(r)+'-run')
m=json.loads((repo/'configs/program2_h100_100k/manifest.json').read_text());trials=[]
for entry in m['runs']:
    p=r/'trials'/entry['method'];assert (p/'COMPLETE.txt').exists()
    v=json.loads((p/'TRAINING_VERIFIED.json').read_text());assert v['status']=='passed'
    v['trial_receipt_sha256']=hashlib.sha256((p/'TRAINING_VERIFIED.json').read_bytes()).hexdigest()
    v['diagnostic_validation']=json.loads((p/'eval-validation/VALIDATION_MLM.json').read_text())
    trials.append(v)
assert len(trials)==4 and len({t['model_tokens'] for t in trials})==1
packet={'status':'passed','observed_utc':datetime.now(timezone.utc).isoformat(),'source_commit':'253c3ea442f2a1657eeb0f3ce2127ac0b1adfb25','production_config_sha256':{e['method']:e['config_sha256'] for e in m['runs']},'trial_overrides':{'max_steps':200,'schedule_steps':200,'warmup_steps':50,'walltime_seconds':1200},'qualification_scope':'Technical stability at configured peak LR, full model/batch/FA3, checkpoint/optimizer/attention reload and 32-sequence MLM smoke evaluation. No quality selection from short trials. Production remains 100k steps and 1000 warmup.','trials':trials}
(r/'ALL_TRIALS_PASSED.json').write_text(json.dumps(packet,indent=2)+'\n')
print('ALL_FOUR_TRIALS_PASSED',flush=True)
PY
