# Protein autoresearch — ESMC-171M validation loss

Minimize `validation_loss`: the frozen held-out MLM evaluator's
`sequence_mean_nll` in `eval-validation/VALIDATION_MLM.json`. Keep a candidate
only when its mean validation-loss improvement exceeds one standard deviation
of its own run-to-run validation loss, using at least two runs per method.
Training loss and full 20,775-chain P@L are required diagnostics; neither
affects selection.

This is the default AutoResearch program on `main`, adapted from
[`program2.md` on `autoresearch-171m-val-loss`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC/blob/autoresearch-171m-val-loss/program2.md).
The local [program2.md](program2.md) preserves the historical campaign
instructions. Existing `program2` paths below keep continuation compatible
with its run records.

## Hard boundaries

- Train from scratch on four L40S GPUs with at most 171,000,000 trainable
  parameters and `walltime_seconds: 3600`. Stop through the synchronized
  wall-time guard, save the final checkpoint, and immediately evaluate it.
- Warm learning rates linearly for exactly 554 optimizer steps, then hold
  every parameter group at its configured peak. No cooldown, cosine decay,
  WSD decay, or other post-warmup LR reduction.
- Freeze the training corpus and mixture, validation data, tokenizer, hardware
  class, GPU count, wall time, dependency lock, and evaluators. Keep evaluation
  sequences, probes, features, masking, seeds, metrics, and reductions fixed;
  never train on held-out data. Validation uses the existing 8 batches of 4
  sequences, context 512, and seed 20260821.
- Model, optimizer, training loss, batching, and training implementation may
  change within these limits. Use `uv sync --frozen` and `uv run --frozen`
  for environment setup and Python commands.

## Isolation and baseline

Create or reuse a separate worktree and branch named `autoresearch-171m-val-loss`.
For a new campaign, branch from `main`; for a continuation, preserve the
existing campaign branch and incumbent. All paths below are relative to that
worktree. Preserve other campaigns' worktrees, branches, launchers, configs,
and results. Share only verified read-only data/evaluation assets; reserve
four idle GPUs for each run.

| Purpose | Path |
| --- | --- |
| Results (untracked) | `.dev/program2/results.tsv` |
| Candidate launcher (untracked) | `.dev/program2/rounds.sh` |
| Candidate configs | `configs/program2/` |
| Checkpoints, evaluation, and run logs | `outputs/program2/<unique-run-id>/` |
| Prepared data and download cache | `.exps/program2/` |
| UV cache | `.uv-cache/program2/` |
| Notes, plots, and continuation state | `.dev/program2/` |

Use `program2-171m` for any tmux session or wakeup identity. Resume only this
campaign's records and incumbent.

Start with `configs/esmc-171m-original.yaml`: the repository's original,
family-scaled ESMC baseline, with 24 layers, width 768, 12 heads, 170,671,168
parameters, AdamW, LayerNorm, RoPE base 10,000, and default residuals and
initialization. Preserve this config and `runs/autoresearch_4xl40s_1h.sh`.
The runner defaults to this original AdamW baseline and supplies the frozen
data release and evaluation contract. Set `CONFIG` explicitly for every
candidate and seed repeat.

Evaluate the baseline with at least two runs at campaign start, before
modifying training code. The command below launches the first run; use copies
under `configs/program2/` changing only the training seed for baseline repeats.

```bash
CONTACT_ROOT=/path/to/frozen-contact-data \
EXTERNAL_SRC=/path/to/evaluation-source \
ARTIFACT_ROOT="$PWD/.exps/program2" \
UV_CACHE_DIR="$PWD/.uv-cache/program2" \
CONFIG="$PWD/configs/esmc-171m-original.yaml" \
OUTPUT_ROOT="$PWD/outputs/program2/baseline-<unique-run-id>" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash runs/autoresearch_4xl40s_1h.sh
```

Replace placeholders for the environment. Unset inherited `DATA_ROOT` and
`DATA_CACHE_ROOT` overrides unless they point to verified read-only inputs.
Candidate launchers use the same command with a config under `configs/program2/`
and a fresh output directory. Never reuse a run directory or alter another
campaign's launcher.

## Results and loop

Each method, including the baseline and incumbent, requires at least N
independent, completed runs from scratch, with **N = 2 by default** and never
fewer than two. Choose distinct training seeds before running and compare
methods on the same seed set; evaluation seeds stay fixed.
Keep the method's code and recipe identical across repeats. Each run receives
the full one-hour budget and a fresh output directory.

Use all repeats to compute arithmetic mean validation loss and sample standard
deviation (`ddof=1`). Define `delta = incumbent_mean - candidate_mean` before
updating the incumbent. Accept only if `delta > candidate_std`; a tie fails.
For exactly two candidate losses, `candidate_std = abs(loss1 - loss2) / sqrt(2)`.
A single run cannot qualify. Equivalently:

```text
candidate_mean_val_loss < current_best_mean_val_loss - candidate_val_loss_std
```

The standard deviation is the candidate's sample SD across training seeds.
This is a selection rule, not a formal significance test.

Create `.dev/program2/results.tsv` only if absent, with this tab-separated header:

```text
method_id\trun_id\tseed\tcommit\tvalidation_loss\tn_runs\tvalidation_loss_mean\tvalidation_loss_std\tdelta\tp_at_l\ttrain_loss\ttraining_seconds\tactual_steps\tmodel_tokens_M\tparams_M\tmemory_GB\tevaluation_seconds\tstatus\tdescription
```

For an older log, preserve its rows, add the new fields, and reconstruct method
summaries from saved runs; complete missing repeats before making decisions.

Record one row per run, repeating the method's `n_runs`, mean, standard
deviation, delta, and decision on its rows after all repeats complete. Use `0`
for the baseline delta and `NA` for unavailable fields or an uncommitted run.
Training loss is the mean of the final 100 recorded MLM losses (all records if
fewer). Copy per-run metrics from
`ROUND_SUMMARY.json`; map `num_params_M` to `params_M` and divide
`peak_vram_mb` by 1024 for `memory_GB`.

1. Require `ROUND_STATE` to show `state=complete`; verify checkpoint, config,
   data, environment, and lockfile receipts for every repeat. Record the initial
   method as `baseline`. Reuse its runs on continuation under the same fixed
   contract; complete any missing repeats before comparing candidates.
2. Test one conceptual change per round in the dedicated worktree. Before each
   launch, including the baseline, run
   `uv run --frozen ruff check nano_protein scripts tests` and
   `uv run --frozen ruff format --check nano_protein scripts tests`.
   Check maintained code; preserve archived source snapshots in `reports/`.
3. Train and evaluate all repeats, verify completion and receipts, and append
   all metrics with method/run IDs, seeds, and a description of the change.
   Commit on `autoresearch-171m-val-loss` as `keep` only when
   `incumbent_mean - candidate_mean > candidate_std`; otherwise record `discard`
   and restore only that candidate's changes. The accepted mean and seed set
   become the new incumbent reference.
4. Record failed training/evaluation or non-finite metrics as `crash`, with the
   cause, and restore only candidate changes. Continue from the incumbent until
   stopped.

## Scale-up verification

Every kept research change needs a scale-up check before claiming that its
improvement transfers to longer training. The current scale-up trains the
171M model family from scratch on four H100s for 100,000 steps at global batch
1,024, then evaluates 4,096 held-out MLM sequences and all 20,775 contact chains.
Compare each recipe with the baseline and preceding recipe under the same
scale-up settings; record pending checks and unsuccessful transfers alongside
completed results. Keep these results separate from the one-hour selection
record. See [the matched recipes](docs/PROGRAM2_SCALEUP.md) and
[scale-up leaderboard](README.md#scale-up-leaderboard).
