import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

root=Path('/scratch/muchenli/Nano-Protein-LM-r02-rope10k-100k-20260906')
now=datetime.now(timezone.utc)
methods=['r02_rope10k','r04_batchbalance','r10_sqrtloss','r29_tied']
snapshot={'observed_utc':now.isoformat(),'trials':{},'full':{}}
for name in ['ALL_TRIALS_PASSED.json']:
    if (root/name).exists(): snapshot[name]=json.loads((root/name).read_text())
for name in ['QUEUE_CURRENT.txt','QUEUE_234_COMPLETE.txt','QUEUE_234_FAILED.txt']:
    if (root/name).exists(): snapshot[name]=(root/name).read_text().strip()
for mode in ('trials','full'):
    for method in methods:
        p=root/mode/method
        if not p.exists():
            snapshot[mode][method]={'state':'pending'}
            continue
        run={'state':'starting','output_root':str(p)}
        for filename,key in [('launch-identity.txt','identity'),('training-started-utc.txt','training_started_utc')]:
            if (p/filename).exists(): run[key]=(p/filename).read_text().strip()
        for filename,key in [('TRAINING_COMPLETE.json','training_completion'),('TRAINING_VERIFIED.json','verification'),('eval-validation/VALIDATION_MLM.json','validation'),('eval-p-at-l/P_AT_L.json','contact'),('eval-p-at-l/P_AT_L_UNCERTAINTY.json','contact_uncertainty')]:
            if (p/filename).exists():
                value=json.loads((p/filename).read_text())
                if key=='contact': value={k:v for k,v in value.items() if k!='components'}
                run[key]=value
        if (p/'metrics.jsonl').exists():
            rows=[]
            for line in (p/'metrics.jsonl').read_text().splitlines():
                try:
                    row=json.loads(line)
                    if row.get('event')=='train': rows.append(row)
                except json.JSONDecodeError: pass
            if rows:
                last=rows[-1];run['latest']=last;run['state']='training'
                run['finite_logged_losses_and_gradients']=all(math.isfinite(row[k]) for row in rows for k in ('loss','gradient_norm'))
                run['metrics_age_seconds']=now.timestamp()-(p/'metrics.jsonl').stat().st_mtime
                window=[row for row in rows if row['optimizer_step']>=max(50,last['optimizer_step']-200)]
                if len(window)>1:
                    seconds=(last['training_seconds']-window[0]['training_seconds'])/(last['optimizer_step']-window[0]['optimizer_step'])
                    target=200 if mode=='trials' else 100000
                    eta=datetime.fromtimestamp((p/'metrics.jsonl').stat().st_mtime,timezone.utc)+timedelta(seconds=seconds*(target-last['optimizer_step']))
                    run['timing']={'seconds_per_step':seconds,'training_eta_toronto':eta.astimezone(ZoneInfo('America/Toronto')).isoformat()}
        if 'training_completion' in run: run['state']='training_complete_verifying_or_evaluating'
        if 'validation' in run and mode=='full': run['state']='mlm_complete_contact_pending_or_running'
        if (p/'COMPLETE.txt').exists(): run['state']='complete'
        if (p/'FAILED.txt').exists(): run['state']='failed';run['failure']=(p/'FAILED.txt').read_text().strip()
        snapshot[mode][method]=run
print(json.dumps(snapshot,indent=2,sort_keys=True))
