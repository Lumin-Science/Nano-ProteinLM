# Karpathy-style sequential search: agent gate

This optional method proposes one candidate at a time, and the agent decides, as a scientist would, whether each one is worth keeping. There is no fixed acceptance rule. The shared [AutoResearch protocol](../docs/AUTORESEARCH.md) and the selected task define the scientific boundaries, round allowance, compute and evaluation. In the alternative, [karpathy_ar_reward_gate.md](karpathy_ar_reward_gate.md), a fixed reward rule decides. Other methods can use the same tasks without following either program.

Read the user-selected task and `AGENTS.md`. If no task is selected, stop. Read the original campaign prompt and recent research records on every continuation. Resource instructions in the prompt identify the allocated machine; they do not change the task's measurement budget.

## Start or resume

Work inside the organizer-prepared clone of the AutoResearch release tag. Do not create a worktree from the research checkout, fetch another branch or consult previous repository findings. You may create a campaign branch such as `ar-YYMMDD-<name>` from this clean starting point and commit your own changes. Resume the same workspace and journals after sleeping or interruption.

Choose a short campaign name. Keep `results.tsv`, `research.log`, candidate diffs and measurement receipts under `$OUTPUT_ROOT/autoresearch/<campaign>/`. Record the release commit, task, primary metric, hardware, data receipts, round limit and any user-imposed stopping condition before training. Read the TSV header and recent journal entries before every trial; retain failures and discarded results.

Establish the incumbent first, by measuring the untouched starting recipe or reusing the reference measurement below. Do not launch owner-run final evaluation unless it was requested.

## Resources and measurement

The current sequential-search profile uses four H100 GPUs with FlashAttention-3, 1,200 seconds of training per round and the task's complete fixed evaluation. Use the compute allocation named by the user after checking that it is live and exposes four matching GPUs. Run training through that allocation, never on a login node, and leave its allocation-holding processes untouched.

Environment and data must be prepared before timing. Use the same storage placement for every run; prefer node-local prepared data when available. Read the coverage receipt and ensure the selected source mixture can finish each run. Do not shorten training, reduce the evaluation population or change protected task scripts to obtain a score.

The task script takes the training seed as its third argument; use seed 42 unless you decide to run another:

```bash
bash tasks/171m-validation-loss_ar.sh configs/autoresearch/esmc-171m.yaml trial-001-seed42 42
```

Use the selected task's entry point for other objectives. Read the completed run's `TRAINING_COMPLETE.json` and `evaluation/EVALUATION.json`, and verify that evaluation matches the final checkpoint and covers the task's full evaluation population. A missing, failed or non-finite measurement cannot support a keep.

## Reference measurement

The owner measured the untouched starting recipe, `configs/autoresearch/esmc-171m.yaml`, with this release's task command on the four-H100 profile, using seeds 42, 43 and 44:

| Task score | Mean ± sample SD over seeds 42, 43 and 44 |
|---|---:|
| `validation_mlm.sequence_mean_nll` | 2.70552 ± 0.00245 |
| `contact.precision_at_l` | 0.09841 ± 0.00250 |

On that profile you may use these values as the baseline instead of measuring it; its three seeds also show the starting recipe's seed-to-seed spread. Reusing them runs nothing, so it consumes no round; record a `baseline` row with `reused` in `decision_reason`. Measure the baseline yourself on the L40S profile or if your starting recipe differs from the release.

## Goal and budget

The recipe you finally select goes to the owner-run [final evaluation](../docs/AUTORESEARCH.md#final-evaluation): training from scratch to 24,200,224,761 non-padding tokens at global batch 1,024, with 1,000 warmup steps and a constant learning rate afterwards, then scoring the task's metric. A search round is a much shorter proxy for that test: 20 minutes of training at global batch 256. Aim for changes that will still help in the final evaluation, not only in the proxy.

Every task invocation consumes one round of the task's search allowance: the baseline, each candidate, each extra seed or repeat, and each failed attempt. How you spend the allowance is your decision. Weigh each new idea against checking or refining one you already have, and note the rounds used and remaining in every decision.

## Evidence from each run

- **Task score:** in `evaluation/EVALUATION.json`. For the validation-loss task, the per-source and median losses (`source_sequence_mean_nll`, `sequence_median_nll`) show where a change helps.
- **Training curve:** `metrics.jsonl` logs the training loss every 10 optimizer steps, together with the learning rate, gradient norm, tokens per second and MFU.
- **Run totals:** `TRAINING_COMPLETE.json` records the optimizer steps, model tokens and training seconds reached within the time limit, the peak GPU memory and the parameter count.
- **The change itself:** the resolved configuration and the candidate's diff.

## Deciding what to keep

Keep a change when the evidence convinces you that it will improve the final evaluation. How you reach that judgement is up to you; think like a scientist. After each measurement, write what the evidence shows and your decision with its reasoning.

The examples below show the kind of reasoning that can help. They are illustrations, not steps you must follow:

- Before a run, you might write down what you expect and what you would do with a clear gain, a small gain or no gain, so the result is easier to interpret afterwards.
- A candidate beats the incumbent by several times the seed-to-seed spread you have observed, with a healthy training curve: keeping it is reasonable.
- A candidate is slightly better, within that spread, and adds complexity: you might discard it, or run one more matched seed for both recipes if the idea matters to you.
- You see promise but are unsure how large the improvement is: one more matched seed for both recipes shows whether the gain survives a different initialization and data order.
- A new component destabilizes training at the current learning rate: the idea may deserve another round or two on that setting before you judge it, while the incumbent stays unchanged.
- Most of a gain comes from higher throughput: the round reaches more tokens, which helps its score but not the fixed-token final evaluation.
- A candidate learns faster early, but its training curve flattens near the end: the gain may shrink with longer training, and the final evaluation also uses a larger batch.
- Loss spikes or growing gradient norms late in a run may get worse with longer training.
- Later evidence suggests that an earlier keep was noise: reverting it is an option.
- Two recipes perform about equally: the simpler one is usually the better keep.
- Late in the allowance, confirming the incumbent or trying cheap refinements may be worth more than a large new idea.

## Research records

Append one row per task invocation, and one for a reused baseline, to `results.tsv`, using tab-separated cells on one line and `NA` for unavailable values. Suggested columns are:

```tsv
timestamp_utc	round_id	trial_id	idea_id	seed	code_revision	incumbent_id	score	primary_gap	decision	decision_reason	artifacts
```

Use `baseline`, `keep`, `discard`, `repeat`, `iterate`, `revert` or `failed` for the decision, and `pending` when a declared repeat defers it. `idea_id` groups the rounds spent on one idea. Define the sign of `primary_gap` so that positive means improvement.

Append timestamped events to `research.log`: campaign settings, hypothesis and prediction, exact command, resolved configuration, failure diagnosis, evidence, decision with its reasoning and next idea. Save a candidate's diff before training. Commit kept changes and update the incumbent; restore only the discarded candidate's edits, preserving journals, evidence and unrelated files. Inspect a live process or Slurm step before deciding whether an interrupted run needs recovery; a stale log or observation timeout alone is not proof that training stopped.

## Loop and sleep

When invoked with `ar-loop-n-sleep`, run inside tmux. The skill owns `.ar/PROMPT.md`, `.ar/events.tsv` and the delayed wakeup of the same pane; this program owns the search decisions and round ledger. Retain the original prompt and stopping condition across wakeups. A background training or evaluation command must survive the end of your turn, record its exit status and expose a log and process handle for the next check.

After launching a run, confirm that its process is live and its log shows progress before sleeping. On wakeup, inspect the actual process and complete artifacts before taking the next step. Include evaluation in the continuation plan, even when training has finished. Stop at the user limit or exhausted round allowance, leave the selected recipe and decision record clear, and schedule no further wakeup after completion.
