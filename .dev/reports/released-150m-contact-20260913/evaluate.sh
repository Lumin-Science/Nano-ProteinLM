#!/usr/bin/env bash
set -euo pipefail
root=/scratch/muchenli/Nano-Protein-LM-released-150m-contact-20260913
export PYTHONPATH="$root/source/src:$root/E1/src"
export HF_HOME="$root/hf-cache" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$root/inductor-cache"
export TRITON_CACHE_DIR="$root/triton-cache"
py="$root/runtime/bin/python"
dataset=/localscratch/muchenli.12162637.0/nano-contact-c135bc80
external=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908-run/reports/fir-r02-rope10k-100k-20260906/launch/contact-evaluator-src
test -f "$root/MODEL_SOURCES.json"
"$py" -m pip freeze > "$root/runtime-requirements.txt"
run_family() {
    local family="$1" offset="$2"
    mkdir -p "$root/$family"
    CUDA_VISIBLE_DEVICES="$offset" "$py" -u -m nanoprotein.released_contact smoke \
      --family "$family" --model-dir "$root/models/$family" \
      --output "$root/$family/SMOKE.json" > "$root/$family/smoke.log" 2>&1
    CUDA_VISIBLE_DEVICES="$offset" "$py" -u -m nanoprotein.released_contact fit \
      --family "$family" --model-dir "$root/models/$family" \
      --dataset-root "$dataset" --external-src "$external" \
      --output "$root/$family/PROBE.json" > "$root/$family/fit.log" 2>&1
    local pids=() shard
    for shard in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES="$((offset + shard))" "$py" -u -m nanoprotein.released_contact score \
          --family "$family" --model-dir "$root/models/$family" \
          --dataset-root "$dataset" --external-src "$external" \
          --probe-receipt "$root/$family/PROBE.json" --shard-index "$shard" --shard-count 4 \
          --output "$root/$family/shard-$shard.json" > "$root/$family/shard-$shard.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0 pid
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    test "$failed" = 0
    touch "$root/$family/COMPLETED"
}
run_family esm2 0 &
esm2_pid=$!
run_family e1 4 &
e1_pid=$!
failed=0
wait "$esm2_pid" || failed=1
wait "$e1_pid" || failed=1
test "$failed" = 0
echo ALL_EVALUATIONS_COMPLETED
