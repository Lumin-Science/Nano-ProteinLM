# Karpathy-style sequential search: reward gate

This optional method proposes one candidate at a time and keeps it only when a fixed reward rule says so. The shared [AutoResearch protocol](../docs/AUTORESEARCH.md) and the selected task define the scientific boundaries, round allowance, compute and evaluation. In the alternative, [karpathy_ar_agent_gate.md](karpathy_ar_agent_gate.md), the agent decides what to keep. Other methods can use the same tasks without following either program.

Read the user-selected task and `AGENTS.md`. If no task is selected, stop. Read the original campaign prompt and recent research records on every continuation. Resource instructions in the prompt identify the allocated machine; they do not change the task's measurement budget.

## Start or resume

Work inside the organizer-prepared clone of the AutoResearch release tag. Do not create a worktree from the research checkout, fetch another branch or consult previous repository findings. You may create a campaign branch such as `ar-YYMMDD-<name>` from this clean starting point and commit your own changes. Resume the same workspace and journals after sleeping or interruption.

Choose a short campaign name. Keep `results.tsv`, `research.log`, candidate diffs and measurement receipts under `$OUTPUT_ROOT/autoresearch/<campaign>/`. Record the release commit, task, reward, hardware, data receipts, round limit and any user-imposed stopping condition before training. Read the TSV header and recent journal entries before every trial; retain failures and discarded results.

Start by measuring the untouched starting recipe on the allocated GPUs with seeds 42 and 43. These two baseline runs do not count toward the round allowance. Every other task invocation consumes one round, including a failed training attempt. The normal allowance is 72 rounds; a user-requested qualification may stop earlier. Do not launch owner-run final evaluation unless it was requested.

## Resources and measurement

The current sequential-search profile uses four H100 GPUs with FlashAttention-3, 1,200 seconds of training per round and the task's complete fixed evaluation. Use the compute allocation named by the user after checking that it is live and exposes four matching GPUs. Run training through that allocation, never on a login node, and leave its allocation-holding processes untouched.

Environment and data must be prepared before timing. Use the same storage placement for every run; prefer node-local prepared data when available. Read the coverage receipt and ensure the selected source mixture can finish each run. Do not shorten training, reduce the evaluation population or change protected task scripts to obtain a score.

## Reward and seeds

The reward is the selected task's score, oriented so that higher is better: the negative of `validation_mlm.sequence_mean_nll` for the validation-loss task, or `contact.precision_at_l` for the P@L task. Every recipe is trained with seed 42 first and seed 43 second. The task script takes the seed as its third argument:

```bash
bash tasks/171m-validation-loss_ar.sh configs/autoresearch/esmc-171m.yaml trial-001-seed42 42
```

Use the selected task's entry point for other objectives. Read the completed run's `TRAINING_COMPLETE.json` and `evaluation/EVALUATION.json`, and verify that evaluation matches the final checkpoint and covers the task's full evaluation population. A missing, failed or non-finite measurement is a failed run, and a failed run discards its candidate.

## Baseline check

The owner measured the untouched starting recipe, `configs/autoresearch/esmc-171m.yaml`, with this release's task command on the four-H100 profile, using seeds 42, 43 and 44:

| Task score | Mean ± sample SD over seeds 42, 43 and 44 |
|---|---:|
| `validation_mlm.sequence_mean_nll` | 2.70552 ± 0.00245 |
| `contact.precision_at_l` | 0.09841 ± 0.00250 |

On that profile, your two baseline runs should land close to these values. A clearly different result points to a setup problem: check the environment, data and GPUs, and tell the user before continuing. On other hardware the scores will differ.

## Keep rule

1. The two baseline runs establish the incumbent. Its statistics are the mean and sample standard deviation of their two rewards.
2. Train each candidate with seed 42. If its reward is at or below the incumbent's mean, discard it after this one round.
3. Otherwise, train it with seed 43 and compute its two-seed mean and sample standard deviation. Keep the candidate only if its mean exceeds the incumbent's mean by more than the larger of the two standard deviations; otherwise discard it.
4. A kept candidate becomes the incumbent, with its two-seed statistics.

With two seeds, the sample standard deviation is `|r42 − r43| / sqrt(2)`. Apply the rule exactly: do not add seeds, rerun a discarded candidate or change the rule after seeing results. A revised version of an idea is a new candidate. If only one round remains, a candidate that passes step 2 cannot be kept.

Before each candidate, write the hypothesis. After each measurement, record the rewards and the step of the rule that decided.

## Research records

Append one row per task invocation to `results.tsv`, using tab-separated cells on one line and `NA` for unavailable values. Suggested columns are:

```tsv
timestamp_utc	round_id	trial_id	seed	code_revision	incumbent_id	score	reward	incumbent_mean	incumbent_sd	candidate_mean	candidate_sd	decision	decision_reason	artifacts
```

Use `baseline`, `pending` (seed 42 passed step 2), `keep`, `discard` or `failed` for the decision. `score` is the task's raw metric and `reward` its oriented value. Give the two baseline rows `round_id` 0, since they do not count toward the allowance.

Append timestamped events to `research.log`: campaign settings, hypothesis, exact command, resolved configuration, failure diagnosis, rewards, decision and next idea. Save a candidate's diff before training. Commit kept changes and update the incumbent; restore only the discarded candidate's edits, preserving journals, evidence and unrelated files. Inspect a live process or Slurm step before deciding whether an interrupted run needs recovery; a stale log or observation timeout alone is not proof that training stopped.

## Loop and sleep

When invoked with `ar-loop-n-sleep`, run inside tmux. The skill owns `.ar/PROMPT.md`, `.ar/events.tsv` and the delayed wakeup of the same pane; this program owns the search decisions and round ledger. Retain the original prompt and stopping condition across wakeups. A background training or evaluation command must survive the end of your turn, record its exit status and expose a log and process handle for the next check.

After launching a run, confirm that its process is live and its log shows progress before sleeping. On wakeup, inspect the actual process and complete artifacts before taking the next step. Include evaluation in the continuation plan, even when training has finished. Stop at the user limit or exhausted round allowance, leave the selected recipe and decision record clear, and schedule no further wakeup after completion.
