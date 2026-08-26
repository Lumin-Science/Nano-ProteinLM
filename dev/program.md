# Protein autoresearch

This program adapts the loop in
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) and its
[`program.md`](https://github.com/karpathy/autoresearch/blob/master/program.md):
change the training code, commit, run a fixed-budget experiment, measure one
score, keep improvements, log everything, and repeat.

## Goal

Maximize frozen long-range contact precision at L (**P@L**). Higher is better.

P@L is the only validation and model-selection score. Do not use MLM loss,
P-CORE, or an invented aggregate to keep or discard an experiment.

Use the frozen full 20,775-chain contact evaluator already in this repository:
the same chain manifest, probe split, attention features, contact definition,
top-L rule, and chain-mean reduction. The execution may be parallelized or
otherwise accelerated only after exact score and per-chain parity against the
qualified reference path. Never weaken or approximate the metric or its data.

## Compute budget

Every actual training run gets exactly:

- four GPUs;
- 3,600 seconds of synchronized training-loop wall time.

This is the only model-size, token, FLOP, batch-size, or optimizer-step
constraint. Architecture, optimizer, batch size, compilation, kernels,
precision, checkpointing, packing, and other efficiency improvements are fair
game. A faster candidate is allowed to train for more steps and tokens inside
the same hour.

Startup, the timing smoke run, checkpoint writing, and P@L evaluation are
outside the 3,600-second training clock, but their durations must be logged.

P@L runs on the same four GPUs using 32 deterministic process shards by
default. Before this path is adopted, it must reproduce the qualified
20,775-chain score exactly and finish in less than 600 seconds. Sharding is an
execution detail: every chain, feature, fitted probe, and reduction stays
unchanged.

## Fixed contract

Do not change the training corpus or mixture, tokenizer, masking/loss contract,
four-GPU hardware class, 3,600-second clock, contact metric, dependencies,
`pyproject.toml`, or `uv.lock`.

The only allowed corpus is a deterministic shard prefix of the immutable
`full-open-v2` Hugging Face release. After the timing smoke run, download at
least `estimated_steps × 4 GPUs × micro_batch_size` unique sequence records
with `scripts/download_data.py`; record the resolved Hub commit, download-plan
digest, and local manifest digest. Training must pass the P@L, P-CORE v0.2, and
P-CORE v0.5-alpha-q9 all-splits homology contract and post-write corpus
verification. Never create or use an exact-only or pre-Q9 corpus.

Use `uv sync --frozen` and `uv run --frozen` for every Python command. Never use
`pip`, Conda, or an ambient environment.

The agent may edit model, optimizer, schedule, training, and efficiency code;
experiment configs; and focused verification commands. Keep `program.md`, data preparation,
evaluation, archived results, and the paper unchanged during the loop.

## Setup

1. Confirm the branch is `auto-research`.
2. Run `uv run --frozen ruff check .` and `uv run --frozen ruff format --check .`.
3. Verify the production corpus and contact receipts.
4. Initialize an untracked `results.tsv` with the header below.
5. Run the unchanged code first to establish the one-hour P@L baseline.

Do not propose a candidate before the baseline succeeds.

## Estimate the one-hour iteration budget

Before every actual run, including the baseline, run a short four-GPU smoke
test with the exact candidate code, architecture, optimizer, batch settings,
and production data.

1. Pass initialization, allocator growth, and one-time compilation.
2. Measure at least 50 steady-state optimizer steps.
3. Synchronize all ranks and use the slowest-rank elapsed time.
4. Compute:

   ```text
   seconds_per_step = measured_training_seconds / measured_optimizer_steps
   estimated_steps = ceil(1.02 * 3600 / seconds_per_step)
   ```

5. Set the actual run's step target to `estimated_steps`. Step-based warmup or
   decay must use this estimate.
6. Keep `walltime_seconds: 3600` as the authoritative hard stop.
7. Start actual training from step zero in a fresh output directory. Never
   continue from the smoke checkpoint.

The 2% headroom makes the wall-time guard, rather than an underestimated step
count, the expected stopping condition. If the actual run reaches the estimated
step target materially before one hour, correct the estimate and rerun. Log
both estimated and realized steps.

## Experiment output

Every completed run must report:

```text
p_at_l:
training_seconds:
smoke_seconds_per_step:
estimated_steps:
actual_steps:
model_tokens_M:
num_params_M:
peak_vram_mb:
```

Only `p_at_l` decides whether the candidate improves. The other fields explain
how it used the fixed budget.

## Logging

`results.tsv` is tab-separated and remains untracked:

```text
commit\tp_at_l\tdelta\tsmoke_step_s\testimated_steps\tactual_steps\tmodel_tokens_M\tparams_M\tmemory_GB\tstatus\tdescription
```

Statuses are `baseline`, `keep`, `discard`, and `crash`. The description must
state exactly what changed. Use `NA` for unavailable crash fields, not zero.

Example:

```text
89abcde\t0.171200\t+0.007634\t0.0791\t92804\t90977\t5593.1\t333.0\t38.2\tkeep\tdepth-aware residual initialization
```

## Experiment loop

Repeat until manually stopped:

1. Read the incumbent and `results.tsv`.
2. Choose one architecture, optimizer, or efficiency idea.
3. Implement it as a small, reviewable change.
4. Run `uv run --frozen ruff check .` and `uv run --frozen ruff format --check .`.
5. Commit the candidate before using GPUs.
6. Smoke-test it and calculate its one-hour step estimate.
7. Train from scratch on four GPUs for the synchronized 3,600-second budget.
8. Verify checkpoint, config, data, environment, and uv-lock receipts.
9. Run the frozen full P@L evaluator.
10. Append the result and change description to `results.tsv`.
11. If P@L is strictly higher, mark `keep` and continue from the candidate.
12. If P@L is lower, mark `discard`, revert the candidate while preserving Git
    history, and continue from the incumbent.
13. If P@L is exactly tied, keep the simpler or faster implementation.
14. If training or evaluation fails, mark `crash`, log the cause, and continue.

Prefer one conceptual change per experiment. Once separate changes improve
P@L, combine them and measure the combination.

Every output directory must be fresh. Never overwrite a checkpoint, reuse
embeddings from another checkpoint hash, or edit validation after seeing a
result. Redirect verbose output to logs.

Never call `scancel`, terminate a shared allocation, or kill another user's
process. Runs must stop through their synchronized wall-time guard.

## Never stop

After setup and the baseline succeed, continue proposing, running, and logging
experiments until the human manually interrupts the loop. If ideas run out,
inspect near-misses, simplify successful changes, or combine independently kept
improvements. Never weaken the P@L metric or the four-GPU/one-hour contract.
