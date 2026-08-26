# Protein autoresearch

This is a simple experimental loop for improving ESMC-300M under a fixed
compute budget: change the training recipe, train for the full wall time,
evaluate, and keep only real progress.

## Objective

Every completed experiment reports three metrics:

- **P@L:** frozen long-range contact precision at L; higher is better.
- **Training loss:** mean of the final 100 recorded training MLM losses, or all
  recorded losses when fewer than 100 are available; lower is better.
- **Validation loss:** `sequence_mean_nll` from the frozen held-out MLM
  evaluator; lower is better.

P@L is the primary model-selection metric. Training and validation loss are
required diagnostics and must always be reported, but they do not override a
P@L regression.

Use the frozen full 20,775-chain contact evaluator and frozen validation split.
Do not change their sequences, probe, features, masking seed, metric, or
reduction.

## Fixed experiment contract

Each training run:

- starts from scratch on four GPUs;
- uses `walltime_seconds: 3600`;
- stops through the synchronized wall-time guard; and
- writes the final checkpoint and begins validation-loss and P@L evaluation
  immediately after training stops.

Keep fixed the training corpus and mixture, validation data, tokenizer, GPU
count and hardware class, wall time, evaluators, and dependency lock. The agent
may change the model, optimizer, loss, schedule, batching, training
implementation, efficiency code, and experiment config. Use only `uv sync
--frozen` and `uv run --frozen` for Python commands.

## Setup

1. Work directly on `auto-research` and confirm there are no unrelated working
   tree changes.
2. Run `uv run --frozen ruff check .` and
   `uv run --frozen ruff format --check .`.
3. Verify the fixed data and evaluation receipts.
4. Create an untracked `results.tsv` with the header below.

## End-to-end round command

`runs/autoresearch_round.sh` is the canonical round entry point. It runs
training for exactly one hour, held-out MLM validation, full P@L evaluation,
and summary generation without waiting for another Codex turn.

Example:

```bash
DATA_ROOT="$PWD/data/processed/full-open-v2-4h" \
OUTPUT_ROOT="$PWD/outputs/autoresearch-baseline" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash runs/autoresearch_round.sh
```

Use a fresh `OUTPUT_ROOT` for every attempt. Use customized dirs depending on which compute environment you are at.

## Results

Every completed run must report:

```text
p_at_l:
train_loss:
validation_loss:
training_seconds:
actual_steps:
model_tokens_M:
num_params_M:
peak_vram_mb:
evaluation_seconds:
```

`results.tsv` remains untracked and uses this tab-separated schema:

```text
commit\tp_at_l\tdelta\ttrain_loss\tvalidation_loss\ttraining_seconds\tactual_steps\tmodel_tokens_M\tparams_M\tmemory_GB\tstatus\tdescription
```

Statuses are `baseline`, `keep`, `discard`, and `crash`. Use `NA` for a run
that has no commit or unavailable crash fields. The description must state the
single conceptual change that was tested.

## Waiting for a round

After launching a round, monitor it until it has trained stably for at least 50
optimizer steps. Then stop polling and schedule the next check for roughly when
the run should finish.

At the next check there are only two normal cases:

- If the round is still running, check once that it remains healthy, then wait
  again without continuously watching its output.
- If the round has finished, read `ROUND_SUMMARY.json` and complete the
  keep/discard steps. If it failed, inspect the error, fix the bug, and retry
  with a fresh output directory.

`OUTPUT_ROOT/ROUND_STATE` is only a small convenience receipt: `running`,
`complete`, or `failed`.

## Baseline gate

Before Round 1, run the unchanged current-best configuration through the full
end-to-end command above. Require `state=complete`, inspect
`ROUND_SUMMARY.json`, and append a `baseline` row to `results.tsv`. Do not
propose or run the first candidate until this baseline succeeds.

## Experiment loop

Repeat until manually stopped:

1. Read the incumbent result and `results.tsv`.
2. Think deeply about the results and choose one promising model,
   optimization, loss, or efficiency change.
3. Apply the change directly in the `auto-research` working tree. Do not create
   a branch or commit yet.
4. Run Ruff checks, submit `runs/autoresearch_round.sh` with a fresh output
   directory, and monitor it until it trains stably for at least 50 steps.
5. Follow the waiting rules above. The round command moves directly from the
   wall-time stop into validation and P@L evaluation.
6. Verify the checkpoint, config, data, environment, and lockfile receipts.
7. If P@L is strictly higher than the incumbent, commit the candidate on
   `auto-research`, append a `keep` row with its commit hash and all metrics,
   and continue from it.
8. If P@L is tied or lower, append a `discard` row with commit `NA`, restore
   only the candidate changes, and leave the incumbent code unchanged.
9. If training or evaluation fails, append a `crash` row, record the cause,
   restore the candidate changes, and continue.

Keep one conceptual change per experiment. Continue until the human manually
stops the loop.
