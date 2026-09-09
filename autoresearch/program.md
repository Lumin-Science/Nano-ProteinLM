# AutoResearch loop

Read the task selected by the user in `tasks/` and the repository's `AGENTS.md`.
This document guides the research loop. The task defines the question, score,
budget, repeats, commands, boundaries and Test of Progress; take concrete
settings from that task and its scripts, not from historical experiments.

## Start or resume

- Use a dedicated research branch/worktree. Preserve unrelated changes and other
  campaigns; choose a campaign name instead of reusing a historical branch name.
- Keep `results.tsv`, `research.log` and trial artifacts under
  `OUTPUT_ROOT/<campaign>/`, following the recording format below. Create the
  journals only if absent; on resume, read their history before scheduling work.
- Measure the task's starting recipe with all required repeats before changing
  it. This establishes the incumbent. Reuse earlier measurements only when the
  code, data, hardware and evaluation protocol match; otherwise rebaseline.

## Research records

A trial is one candidate recipe evaluated across all required repeats. Use a
stable ID such as `trial-001`; save its recipe, candidate diff, raw logs and run
outputs under `OUTPUT_ROOT/<campaign>/<trial_id>/`.

**`results.tsv`** is the score ledger: append one row per finished trial,
including the baseline and failed trials. Use this tab-separated header:

```tsv
timestamp_utc	trial_id	code_revision	incumbent_id	repeat_ids	scores	mean	std	improvement	status	description	artifacts
```

`repeat_ids` and `scores` are comma-separated lists in matching order, preserving
every seed/repeat's primary score. `mean`, `std` and `improvement` follow the
comparison rule below; `incumbent_id` identifies the trial being compared against.
Use `baseline`, `keep`, `discard` or `failed` for `status`, and `NA` for missing
values. Failed or incomplete trials retain available scores but have no qualifying
mean, SD or improvement. Keep cells on one line without embedded tabs; link
detailed diagnostics through the campaign-relative `artifacts` directory.

**`research.log`** is the reasoning journal: append one timestamped line per trial
event, using `timestamp_utc trial_id event message`. Before a trial, log `start`
with its hypothesis and incumbent; after it, log the decision, evidence, failure
cause if any, and next idea. Record the task path/version and measurement context
when the campaign starts, and log interruptions or policy changes as they happen.
Preserve both journals across restarts and discarded candidates.

At startup/resume and before choosing every new trial, read the TSV header and
the tails of both journals. Read older entries or linked artifacts when needed
to understand the incumbent or avoid repeating an earlier experiment:

```bash
campaign_dir="$OUTPUT_ROOT/<campaign>"
head -n 1 "$campaign_dir/results.tsv"
tail -n 20 "$campaign_dir/results.tsv" "$campaign_dir/research.log"
```

## Iterate

1. Read the history tails above and inspect the incumbent. Propose one clear
   change and explain why it could improve the task's score within its compute
   budget; record the hypothesis in `research.log` before running it.
2. Review the diff against the task boundaries yourself. Check relevant code,
   resolved settings and model size; run focused tests before expensive work.
   A command succeeding does not establish scientific validity. Do not change
   the task, measurement scripts or evaluation to make a candidate qualify.
3. Run the task's ordinary train/simulate/evaluate command for every required
   repeat, using fresh run directories and the same code/recipe across repeats.
   Match seeds and measurement conditions to the incumbent.
   Read the logs and completion records: verify the budget, data, hardware and
   evaluation actually match the task. Record incomplete or failed runs with
   their cause; they cannot supply a candidate score.
4. Compare the task's primary score using all planned repeats. Report diagnostics
   separately; do not substitute a better-looking diagnostic or cherry-pick seeds.
   For repeated-run scores, use the arithmetic mean and sample SD (`ddof=1`).
5. Default retention rule: keep a candidate only
   when its improvement over the incumbent mean exceeds its own sample SD.
   Improvement is `incumbent_mean - candidate_mean` for a lower-is-better score,
   and the reverse for a higher-is-better score. A tie is not a keep. This is a
   search heuristic, not a significance test; record any user-directed policy
   change before comparing candidates.
6. Append the trial row to `results.tsv` and its decision/evidence to `research.log`.
   Commit a kept change on the research branch and update the incumbent. For a
   discarded candidate, restore only its changes while retaining its recipe,
   logs and measurements.
   Continue within the user's budget; leave the incumbent and next hypothesis
   clear when stopping or handing off.

Task compliance and research decisions belong to the agent. The task script is
an executable measurement recipe, not a boundary adjudicator or an agent loop.
Test of Progress is run separately by the benchmark owner when requested.
