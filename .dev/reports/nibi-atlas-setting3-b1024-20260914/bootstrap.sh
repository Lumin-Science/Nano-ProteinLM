#!/usr/bin/env bash
set -euo pipefail
root=/scratch/muchenli/Nano-Protein-LM-nibi-atlas-setting3-b1024-20260914
[[ "$(hostname -s):${SLURM_JOB_ID:?}" == g27:12162637 ]]
cd "$root"
module load StdEnv/2023 gcc/12.3
py=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/runtime/.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
uv pip install --python "$py" --target "$root/prep-libs" --index-url https://pypi.org/simple pylance==11.0.0
PYTHONPATH="$root/prep-libs" "$py" -u - <<'PY'
import json, time, lance
start=time.monotonic()
ds=lance.dataset('s3://esm-protein-atlas/v1/folds/folds_1B.lance',version=3,storage_options={'aws_skip_signature':'true','region':'us-west-2'})
fragments=ds.get_fragments()
print(json.dumps({'rows':ds.count_rows(),'fragments':len(fragments),'schema':str(ds.schema),'first_counts':[(f.fragment_id,f.count_rows()) for f in fragments[:5]]}),flush=True)
n=0
for batch in ds.to_batches(columns=['protein_hash','sequence'],batch_size=65536,limit=100000,batch_readahead=2,fragment_readahead=1):
    n+=batch.num_rows
print(json.dumps({'sample_records':n,'seconds':time.monotonic()-start}),flush=True)
PY
date -u +%Y-%m-%dT%H:%M:%SZ > BOOTSTRAP_READY.txt
