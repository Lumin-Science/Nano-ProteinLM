<div class="ai">

> Archived copy of `autoresearch/program.md` as used by AutoResearch round 2 (commit `28b742c`, September 13, 2026), which ran each candidate with `N=2` training seeds. The text below is unchanged. The current one-seed method is [autoresearch/program.md](../../autoresearch/program.md); both rounds are described in [AUTORESEARCH_BASELINE.md](../AUTORESEARCH_BASELINE.md#previous-two-seed-pipeline).

</div>

# AutoResearch loop

Read the user-selected task and the project's `AGENTS.md`. If no task is selected, stop. The task defines the objective, reward, evaluation commands and editable scope; this document defines the loop and records.

## Start or resume

Choose `ar_run_name` to briefly summarize the task and user prompt.

- Use an isolated worktree under the main checkout's `.worktrees/<branch-name>/` directory. If the current branch does not match `ar-YYMMDD-<ar_run_name>`, create a worktree with a branch of that form; otherwise reuse its existing worktree and derive the date and `ar_run_name` from the branch. Run all campaign edits and commands from that worktree.
- Follow the task's setup and baseline requirements, then evaluate the starting implementation to establish the incumbent. Keep the scoring reference fixed.
- Keep `results.tsv`, `research.log` and trial artifacts under `autoresearch/<ar_run_name>/`. Resume existing journals; reuse results only when code, data, evaluation settings and environment still meet the task's requirements.

## Evaluations per candidate

Run `N` seed evaluations per candidate, where the task specifies `N` and evaluations may run in parallel when resources permit; all `N` must succeed for a valid run.

## Research records

A trial is one candidate evaluated `N` times. Use an ID such as `trial-001` and save its diff, commands and all evaluation outputs under `autoresearch/<ar_run_name>/<trial_id>/`.

**`results.tsv`**: append one row per finished trial, including the baseline and failed trials.

For example:

```tsv
timestamp_utc	trial_id	code_revision	incumbent_id	n_evaluations	completed_evaluations	samples_per_system	reward(mean)	ci95_low	ci95_high	improvement	status	description	artifacts
```

This is an example; add or adapt columns for the selected task and describe them in `research.log`. Keep headers and rows aligned, cells on one line without embedded tabs, and use `NA` for unavailable values while preserving historical measurements.

- `n_evaluations` / `completed_evaluations`: planned `N` / successful complete evaluations.
- `reward(mean)`: mean evaluation reward in this example; use the task's aggregation rule when specified.
- `ci95_low` / `ci95_high`: 95% confidence bounds calculated as the task specifies; for mean rewards, default to `mean ± t(0.975, N−1) × s / sqrt(N)`, where `s` is the sample standard deviation of the `N` complete evaluation rewards (denominator `N−1`). This assumes independent, approximately normal rewards and `N >= 2`.
- `improvement`: candidate reward minus the reward of `incumbent_id`.
- `status`: `baseline`, `keep`, `discard` or `failed`. A failed gate or incomplete evaluation has no qualifying reward, interval or improvement.
- `artifacts`: campaign-relative trial directory. Retain measured regressions as completed trials.

**`research.log`**: append timestamped events using `timestamp_utc trial_id event message`. Record the task, `N`, baseline and evaluation settings at campaign start; for each trial, record the hypothesis, decision, evidence, failures and next idea. Log interruptions and policy changes, preserving history across restarts.

Read the TSV header and recent entries in both journals at startup and before each trial; consult older entries and artifacts when needed to avoid repeating work.

## Keep Rule

Keep a candidate only when `candidate.ci95_low > incumbent.mean` and `candidate.mean > incumbent.ci95_high`.

## Iterate

1. Review the incumbent and history, propose a change within the task's editable scope, and log the hypothesis before running.
2. Check the diff against task boundaries and run relevant checks. Do not change protected files or evaluation rules to qualify a candidate.
3. Run the complete evaluation `N` times with fixed candidate code and conditions, using a fresh output directory each time. Retain every result, including failures.
4. Compare the aggregated reward with the incumbent using the task's objective, diagnostics and uncertainty. Remeasure unclear results consistently and retain every attempt.
5. Decide whether to keep the current improvement using the keep rule; a tie is not a keep. Commit kept changes when using Git and update the incumbent; for discarded candidates, restore only their changes and retain the evidence.
6. Append the trial row and decision to the journals. Continue within the user's budget, leaving the incumbent and next idea clear when stopping.

Follow the task's instructions for any final or owner-run evaluation.
