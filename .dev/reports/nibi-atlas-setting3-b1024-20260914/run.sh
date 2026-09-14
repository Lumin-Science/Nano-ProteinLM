#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname -s):${SLURM_JOB_ID:?}" == g27:12162637 ]]
root=/scratch/muchenli/Nano-Protein-LM-nibi-atlas-setting3-b1024-20260914
mode=${1:?pilot or full}
case "$mode" in pilot|full) ;; *) exit 2;; esac
export PAIR_ROOT="$root" PAIR_REPO="${ATLAS_REPO:-$root/source}"
export TRAIN_PYTHON=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/runtime/.venv/bin/python
export EXTERNAL_SRC=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908-run/reports/fir-r02-rope10k-100k-20260906/launch/contact-evaluator-src
export PYTHONPATH="$PAIR_REPO/src:$EXTERNAL_SRC" PYTHONUNBUFFERED=1
export CONTACT_ROOT="$SLURM_TMPDIR/nano-contact-c135bc80"
export HF_HOME=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/hf-cache
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CHECKPOINT_ARCHIVE="/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-atlas-setting3-b1024-20260914/$mode"
prep="$SLURM_TMPDIR/nano-atlas-20260914/$mode"
export DATA_ROOT="$prep/data"
test -f "$prep/DATA_READY.json"
out="$root/$mode-training"
test ! -e "$out"
mkdir -p "$out"
cd "$PAIR_REPO"
"$TRAIN_PYTHON" - "$mode" "$out" "$prep" <<'PY'
import json, os, sys, time, yaml
from pathlib import Path
mode, out, prep = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
c = yaml.safe_load(Path('.dev/configs/nibi/setting3-atlas-b1024-100k.yaml').read_text())
ready = json.loads((prep/'DATA_READY.json').read_text())
assert ready['status'] == 'prepared_unscreened'
if mode == 'pilot':
    c.update(max_steps=100, schedule_steps=100, checkpoint_interval=50, periodic_evaluation_interval=0, log_interval=10)
else:
    assert ready['records'] >= 100000 * 1024 * 1.01
    assert Path(os.environ['PAIR_ROOT'], 'pilot-training', 'PILOT_PASSED.json').is_file()
end = int(Path(os.environ['PAIR_ROOT'], 'ALLOCATION_END_UNIX.txt').read_text())
c['stop_at_unix_time'] = end - 1200
assert c['stop_at_unix_time'] - time.time() > 600
(out/'launch-config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
(out/'LAUNCH.json').write_text(json.dumps(dict(mode=mode, initialization='fresh', global_batch=1024, world_size=8, data=ready, allocation_end_unix=end, stop_at_unix_time=c['stop_at_unix_time'], contamination_screened=False),indent=2)+'\n')
PY
"$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
  -m nanoprotein.train --config "$out/launch-config.yaml" --data-root "$DATA_ROOT" --output-root "$out" > "$out/train.log" 2>&1
report="$PAIR_REPO/.dev/reports/nibi-atlas-setting3-b1024-20260914"
export CHECKPOINT_ARCHIVE="$CHECKPOINT_ARCHIVE/final"
bash "$report/evaluate-checkpoint.sh" "$out/checkpoint-final.pt" "$out/final-evaluation" > "$out/final-evaluation.log" 2>&1
"$TRAIN_PYTHON" -m nanoprotein.checkpoint_audit --checkpoint "$out/checkpoint-final.pt" --data-root "$DATA_ROOT" --restore-optimizer --output "$out/FINAL_CHECKPOINT_VERIFIED.json"
if [[ "$mode" == pilot ]]; then
  "$TRAIN_PYTHON" - "$out" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); p=json.loads((out/'FINAL_CHECKPOINT_VERIFIED.json').read_text())
assert p['optimizer_steps']==100 and p['model_and_optimizer_roundtrip_exact']
assert p['world_size']==8 and p['parameter_count']==170559856
assert p['source_exposure_global']['esm_atlas']['draws']==102400
assert json.loads((out/'final-evaluation/RESULT_VERIFIED.json').read_text())['status']=='passed'
(out/'PILOT_PASSED.json').write_text(json.dumps(p,indent=2)+'\n')
PY
fi
