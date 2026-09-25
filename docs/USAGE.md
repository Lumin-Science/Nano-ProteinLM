# Training commands

## Search round

```bash
bash tasks/171m-validation-loss_ar.sh configs/autoresearch/esmc-171m.yaml trial-001 42
```

The command requires prepared data and four matching H100 or L40S GPUs. It selects 1,200 seconds on H100 or 3,600 seconds on L40S. Before the training clock starts, it copies `$DATA_ROOT/training` (about 10 GB with the default 30 shards) to `/tmp`, or to the folder named by `NANOPROTEIN_STAGE_DIR`, because random batch reads from shared network storage can stall training; later runs reuse the copy while its manifest is unchanged. It refuses to overwrite a run directory and evaluates only after training succeeds. `tasks/171m-p-at-l_ar.sh` runs the same training and also scores contact P@L for the P@L task. Each invocation is one round; the search algorithm owns its candidate and seed choices.

## Ordinary training

The commands below assume Hopper GPUs and FlashAttention-3. For the one-hour L40S search budget, use `--attention-backend flash --walltime-seconds 3600` in the ordinary training command below. Set `--peak-bf16-tflops-per-gpu` to the value recorded in a task run's `ENVIRONMENT.json` for hardware-specific utilization reporting. The benchmark task reads these settings automatically.

```bash
uv run --frozen python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m nanoprotein.train --config configs/autoresearch/esmc-171m.yaml \
  --data-root data/training --output-root outputs/training-001 \
  --walltime-seconds 1200 --seed 42
```

Ordinary training and benchmark measurements use the same API. Read the resolved recipe, run_contract.json, DATA_COVERAGE.json and TRAINING_COMPLETE.json to confirm the effective settings and stop reason. Successful training writes checkpoint-final.pt. The recipes prefetch two micro-batches in a background thread without changing their order; each metrics.jsonl record reports `step_data_seconds`, the longest time any GPU process waited for data in that step. The trainer supports time, step and token limits; the measurement command clears step and token caps so the round uses its training-time allowance.

## Owner-run final evaluation

Prepare enough data for the selected mixture and the 24,200,224,761-token target before starting. The example trains the reference on four Hopper GPUs with FlashAttention-3, where it takes about 12 hours. On other GPUs, use `--attention-backend flash` and raise the 16-hour safety time cap; completion is determined by the token target, not this cap. Keep the global batch at 1,024 and use the same seed for every recipe.

```bash
uv run --frozen python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m nanoprotein.train --config configs/test-100k/esmc-171m.yaml --seed 42 \
  --data-root data/training --output-root outputs/final-reference \
  --max-steps none --max-model-tokens 24200224761 --schedule-steps 100000 \
  --walltime-seconds 57600 --attention-backend flash3 --warmup-steps 1000 \
  --micro-batch-size 64 --gradient-accumulation 4 \
  --checkpoint-interval 0 --periodic-evaluation-interval 0
```

Verify that TRAINING_COMPLETE.json reports the token target as the stop reason and at least 24,200,224,761 model tokens. Record the final update's overrun. Repeat for the frozen submitted recipe with the same seed, then use the [evaluation command](EVALUATION.md#evaluate-a-checkpoint) on each final checkpoint. This owner-run comparison does not consume search rounds.
