# Karpathy-style sequential search

This optional method proposes one candidate at a time, measures it with the selected task and keeps or discards it before proposing another. The shared [AutoResearch protocol](../docs/autoresearch.md) and task define the scientific boundaries, round allowance, compute and evaluation. Other methods can use the same tasks without following this program.

Read the user-selected task and `AGENTS.md`. If no task is selected, stop. Read the original campaign prompt and recent research records on every continuation. Resource instructions in the prompt identify the allocated machine; they do not change the task's measurement budget.

## Start or resume

Work inside the organizer-prepared standalone `autoresearch` clone. Do not create a worktree from the research checkout, fetch another branch or consult previous repository findings. You may create a campaign branch such as `codex/ar-YYMMDD-<name>` from this clean starting point and commit your own changes. Resume the same workspace and journals after sleeping or interruption.

Choose a short campaign name. Keep `results.tsv`, `research.log`, candidate diffs and measurement receipts under `$OUTPUT_ROOT/autoresearch/<campaign>/`. Record the release commit, task, primary metric, seed, hardware, data receipts, round limit and any user-imposed stopping condition before training. Read the TSV header and recent journal entries before every trial; retain failures and discarded results.

Measure the untouched starting recipe first to establish the incumbent. Each task invocation consumes one round, including the baseline, a failed training attempt or an optional repeat. The normal allowance is 72 rounds; a user-requested qualification may stop earlier. Do not launch owner-run final evaluation unless it was requested.

## Resources and measurement

The current sequential-search profile uses four H100 GPUs with FlashAttention-3, 1,200 seconds of training per round and the task's complete fixed evaluation. For the Fir deployment, use the user-specified `fc10219` allocation after checking its current Slurm job ID and available GPUs. Run training through that allocation, never on a login node, and leave its allocation-holding processes untouched. The portable program does not assume that this node or job remains allocated in a later campaign.

Environment and data must be prepared before timing. Use the same storage placement for the baseline and candidates; prefer node-local prepared data when available. Read the coverage receipt and ensure the selected source mixture can finish each run. Do not shorten training, reduce the evaluation population or change protected task scripts to obtain a score.

## One seed per candidate

Use one training seed, **42** by default, for the baseline and each candidate. A user may select another common seed before the campaign. The task script takes that seed as its third argument:

```bash
bash tasks/171m-validation-loss_ar.sh configs/default.yaml trial-001-seed42 42
```

Use the selected task's entry point for other objectives. Keep conditions and the common seed matched across comparisons. Read the completed run's `TRAINING_COMPLETE.json` and `evaluation/EVALUATION.json`; verify that evaluation matches the final checkpoint and includes 4,096 validation proteins plus all 20,775 contact chains. Missing, failed or non-finite measurements cannot qualify a candidate.

One seed does not establish statistical significance or estimate training-seed variance. Record seed-level SD and confidence intervals as `NA`; contact-chain bootstrap intervals describe variation over chains and cannot substitute for training-seed uncertainty. The multi-run summary utility is optional and requires at least two distinct seeds; do not call it for a single run.

## Acceptance decision

The agent decides whether an observed improvement is convincing enough to retain. There is no mandatory two-seed replication or fixed confidence-interval gate. Compare the candidate with the current incumbent on the declared objective, quantify the absolute and relative gap, and explain the decision using the measured gap, training stability, diagnostics, throughput and any observed run-to-run variation. A hypothesis alone is not evidence of improvement.

Keep only a valid candidate that improves the primary metric and has a documented reason for treating the gain as useful. Lower validation loss is better; higher P@L is better. Do not switch objectives after observing scores. Discard regressions and ties. A small or ambiguous gain may be discarded, or the agent may spend remaining rounds on a declared matched repeat of both recipes. Preserve every measurement and the reason for spending those rounds; do not selectively repeat until a candidate wins. Single-seed keeps are provisional search decisions, not claims of statistical significance.

Before each candidate run, write the hypothesis and what result would support keeping it. After evaluation, record why the measured evidence does or does not support that expectation. If the decision standard changes, record the change and its reason explicitly; do not rewrite earlier decisions or measurements.

## Research records

Append one row per attempted measurement to `results.tsv`. Use tab-separated cells on one line, with `NA` for unavailable values. Suggested columns are:

```tsv
timestamp_utc	round_id	trial_id	code_revision	incumbent_id	seed	validation_loss	p_at_l	primary_gap	decision	decision_reason	artifacts
```

Use `baseline`, `keep`, `discard` or `failed` for completed decisions. If optional repeats defer a decision, record `pending` and append the later decision to `research.log`, retaining the original measurements. Define the sign of `primary_gap` so positive means improvement. Record P@L intervals separately if needed; do not label them seed-level intervals.

Append timestamped events to `research.log`: campaign settings, hypothesis, exact command, resolved configuration, failure diagnosis, measured gap, decision and next idea. Save a candidate's diff before training. Commit kept changes and update the incumbent; restore only the discarded candidate's edits, preserving journals, evidence and unrelated files. Inspect a live process or Slurm step before deciding whether an interrupted run needs recovery; a stale log or observation timeout alone is not proof that training stopped.

## Loop and sleep

When invoked with `ar-loop-n-sleep`, start Codex inside tmux. The skill owns `.ar/PROMPT.md`, `.ar/events.tsv` and the delayed wakeup of the same pane; this program owns the search decisions and round ledger. Retain the original prompt and stopping condition across wakeups. A background training or evaluation command must survive the end of a Codex turn, record its exit status and expose a log and process handle for the next check.

After launching a run, confirm that its process is live and its log shows progress before sleeping. On wakeup, inspect the actual process and complete artifacts before taking the next step. Include evaluation in the continuation plan, even when training has finished. Stop at the user limit or exhausted round allowance, leave the selected recipe and decision record clear, and schedule no further wakeup after completion.
