#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname -s):${SLURM_JOB_ID:?}" == g27:12162637 ]]
root=/scratch/muchenli/Nano-Protein-LM-nibi-atlas-setting3-b1024-20260914
prep="$SLURM_TMPDIR/nano-atlas-20260914/full"
while [[ ! -f "$prep/DATA_READY.json" ]]; do sleep 15; done
mkdir -p "$root/data" "$root/preparation"
rsync -a "$prep/data/" "$root/data/"
cp "$prep/DATA_READY.json" "$prep/DOWNLOAD_PLAN.json" "$prep/PREPARATION.json" "$root/preparation/"
export PYTHONPATH="$root/source/src"
py=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/runtime/.venv/bin/python
"$py" - "$root" <<'PY'
import json, sys
from pathlib import Path
from nanoprotein.data import file_sha256
from nanoprotein.atlas_data import validate_atlas_manifest
root = Path(sys.argv[1]); data = root/'data'
m = validate_atlas_manifest(data)
for source, splits in m['sources'].items():
    for split, receipt in splits.items():
        for name, key in [('tokens.bin','tokens_sha256'),('index.npy','index_sha256')]:
            assert file_sha256(data/source/split/name) == receipt[key]
p = dict(status='passed',data_root=str(data),manifest_sha256=file_sha256(data/'manifest.json'),training_records=m['sources']['esm_atlas']['train']['records'],contamination_screened=False)
(root/'DATA_PRESERVED.json').write_text(json.dumps(p,indent=2)+'\n')
print(json.dumps(p),flush=True)
PY
