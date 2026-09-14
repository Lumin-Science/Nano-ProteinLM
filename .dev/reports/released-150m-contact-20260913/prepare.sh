#!/usr/bin/env bash
set -euo pipefail
root=/scratch/muchenli/Nano-Protein-LM-released-150m-contact-20260913
export UV_CACHE_DIR="$root/uv-cache" HF_HOME="$root/hf-cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$root"
uv=/home/muchenli/.local/bin/uv
"$uv" venv --python 3.12 --seed "$root/runtime"
"$uv" pip install --python "$root/runtime/bin/python" \
  torch==2.8.0 transformers==4.56.2 huggingface-hub==0.35.0 \
  numpy==2.4.6 scikit-learn==1.9.0 pyarrow==21.0.0 lmdb==2.3.0 PyYAML==6.0.3 \
  einops==0.8.1 kernels pandas polars biotite click psutil scipy
"$root/runtime/bin/python" - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
root=Path('/scratch/muchenli/Nano-Protein-LM-released-150m-contact-20260913')
models={}
for family,repo in [('esm2','facebook/esm2_t30_150M_UR50D'),('e1','Profluent-Bio/E1-150m')]:
    info=HfApi().model_info(repo)
    dest=root/'models'/family
    snapshot_download(repo_id=repo,revision=info.sha,local_dir=dest,allow_patterns=['*.json','*.txt','*.safetensors','README.md','LICENSE*'])
    models[family]={'model_id':repo,'revision':info.sha,'directory':str(dest),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()}}
    print(f'Downloaded {repo} at {info.sha}',flush=True)
(root/'MODEL_SOURCES.json').write_text(json.dumps(models,indent=2)+'\n')
print('PREPARATION_PASSED',flush=True)
PY
