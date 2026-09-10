#!/usr/bin/env bash
# Execute inside the existing Nibi allocation: migration, qualification, or production.
set -euo pipefail
mode=${1:?migration, qualification or production}
export PAIR_ROOT=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-400k-20260910
export PAIR_REPO="$PAIR_ROOT-run"
export TRAIN_PYTHON=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/runtime/.venv/bin/python
export EXTERNAL_SRC=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908-run/reports/fir-r02-rope10k-100k-20260906/launch/contact-evaluator-src
export DURABLE_ROOT=/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-setting3-b2048-400k-20260910
parent=/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-paired-unique-b2048-100k-20260909/setting3/checkpoint-final.pt
export PYTHONPATH="$PAIR_REPO/src:$EXTERNAL_SRC" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HOME=/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908/hf-cache HF_HUB_OFFLINE=1
export CONTACT_ROOT="$SLURM_TMPDIR/nano-contact-c135bc80"
export DATA_ROOT=$(cat "$PAIR_ROOT/PRODUCTION_DATA_ROOT.txt")
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PRESERVE_MILESTONES=0
cd "$PAIR_REPO"
commit=$(cat "$PAIR_ROOT/SOURCE_COMMIT.txt")
[[ "$(git rev-parse HEAD)" == "$commit" && -z "$(git status --porcelain)" ]]
[[ "$(hostname -s):$SLURM_JOB_ID" == g27:12162637 ]]
test -f "$PAIR_ROOT/DATA_READY.json"
config="$PAIR_REPO/configs/setting3-nibi-b2048-400k.yaml"
if [[ "$mode" == migration ]]; then
  "$TRAIN_PYTHON" -m nanoprotein.data_migration --checkpoint "$parent" \
    --old-data-root /scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909/data \
    --data-root "$DATA_ROOT" --output "$PAIR_ROOT/DATA_MIGRATION.json" --seed 20260910
  exit 0
fi
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]
test -f "$PAIR_ROOT/DATA_MIGRATION.json"
case "$mode" in
  qualification) out="$PAIR_ROOT/qualification"; endpoint=100200;;
  production) out="$PAIR_ROOT/full"; endpoint=400000; test -f "$PAIR_ROOT/QUALIFICATION_PASSED.json";;
  *) exit 2;;
esac
test ! -e "$out"
mkdir -p "$out"
trap 'rc=$?; if ((rc!=0)); then printf "exit_code=%s\n" "$rc" > "$out/FAILED.txt"; fi' EXIT
printf 'NODE=%s\nSLURM_JOB_ID=%s\nSLURM_STEP_ID=%s\nCUDA_VISIBLE_DEVICES=%s\n' "$(hostname -s)" "$SLURM_JOB_ID" "$SLURM_STEP_ID" "$CUDA_VISIBLE_DEVICES" > "$out/launch-identity.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/launch-started-utc.txt"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv > "$out/gpu-map.csv"
args=()
if [[ "$mode" == qualification ]]; then
  args+=(--max-steps 100200 --schedule-steps 100200 --checkpoint-interval 100 --periodic-evaluation-interval 100 --walltime-seconds 3600)
else
  export PRESERVE_MILESTONES=1
  mkdir -p "$DURABLE_ROOT"
  cp "$config" "$DURABLE_ROOT/recipe.yaml"
  cp "$PAIR_ROOT/DATA_PLAN.json" "$PAIR_ROOT/DATA_MIGRATION.json" "$PAIR_ROOT/SOURCE_COMMIT.txt" "$DURABLE_ROOT/"
  cp "$PAIR_ROOT/source.bundle" "$DURABLE_ROOT/source.bundle"
fi
"$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 -m nanoprotein.train \
  --config "$config" --data-root "$DATA_ROOT" --output-root "$out" \
  --resume "$parent" --resume-data-migration "$PAIR_ROOT/DATA_MIGRATION.json" "${args[@]}" > "$out/train.log" 2>&1
"$TRAIN_PYTHON" -m nanoprotein.checkpoint_audit --checkpoint "$out/checkpoint-final.pt" \
  --data-root "$DATA_ROOT" --output "$out/TRAINING_VERIFIED.json" --expected-step "$endpoint" --restore-optimizer > "$out/verify.log" 2>&1
if [[ "$mode" == qualification ]]; then
  resume_out="$PAIR_ROOT/qualification-resume"
  test ! -e "$resume_out"
  "$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 -m nanoprotein.train \
    --config "$config" --data-root "$DATA_ROOT" --output-root "$resume_out" \
    --resume "$out/checkpoint-final.pt" --max-steps 100210 --schedule-steps 100210 --checkpoint-interval 0 --periodic-evaluation-interval 0 > "$PAIR_ROOT/qualification-resume.log" 2>&1
  "$TRAIN_PYTHON" -m nanoprotein.checkpoint_audit --checkpoint "$resume_out/checkpoint-final.pt" \
    --data-root "$DATA_ROOT" --output "$resume_out/TRAINING_VERIFIED.json" --expected-step 100210 --restore-optimizer > "$resume_out/verify.log" 2>&1
  "$TRAIN_PYTHON" "$PAIR_REPO/.dev/reports/nibi-setting3-b2048-400k-20260910/verify_qualification.py"
else
  "$TRAIN_PYTHON" -m nanoprotein.preserve_checkpoint --checkpoint "$out/checkpoint-final.pt" \
    --receipt "$out/TRAINING_VERIFIED.json" --destination "$DURABLE_ROOT"
  bash runs/evaluate_global_checkpoint.sh "$out/checkpoint-final.pt" "$out/evaluations/step-400000" > "$out/final-evaluation.log" 2>&1
  cp "$out/TRAINING_VERIFIED.json" "$out/TRAINING_COMPLETE.json" "$out/run_contract.json" "$out/DATA_COVERAGE.json" "$out/RESUME.json" "$DURABLE_ROOT/"
  rsync -a --exclude='components/' --exclude='*.log' --exclude='*.stdout' --exclude='*.stderr' "$out/evaluations/" "$DURABLE_ROOT/evaluations/"
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/COMPLETE.txt"
