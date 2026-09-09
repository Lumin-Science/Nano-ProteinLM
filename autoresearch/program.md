# AutoResearch loop

Read the task selected by the user in `tasks/` and the repository's `AGENTS.md`.
This document guides the research loop. The task defines the question, score,
budget, repeats, commands, boundaries and Test of Progress; take concrete
settings from that task and its scripts, not from historical experiments.

## Start or resume

- Use a dedicated research branch/worktree. Preserve unrelated changes and other
  campaigns; choose a campaign name instead of reusing a historical branch name.
- Keep candidate recipes, run outputs and a small experiment journal under
  `OUTPUT_ROOT/<campaign>/`. Record the task version, code revision, hypothesis,
  recipe, seeds/repeats, per-run scores, mean, sample SD, decision and artifact paths.
- Measure the task's starting recipe with all required repeats before changing
  it. This establishes the incumbent. Reuse earlier measurements only when the
  code, data, hardware and evaluation protocol match; otherwise rebaseline.

## Iterate

1. Inspect the incumbent and results. Propose one clear change and explain why
   it could improve the task's score within its compute budget.
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
5. Default retention rule, carried over from `program2.md`: keep a candidate only
   when its improvement over the incumbent mean exceeds its own sample SD.
   Improvement is `incumbent_mean - candidate_mean` for a lower-is-better score,
   and the reverse for a higher-is-better score. A tie is not a keep. This is a
   search heuristic, not a significance test; record any user-directed policy
   change before comparing candidates.
6. Record `keep`, `discard`, or `failed` with the evidence. Commit a kept change
   on the research branch and update the incumbent. For a discarded candidate,
   restore only its changes while retaining its recipe, logs and measurements.
   Continue within the user's budget; leave the incumbent and next hypothesis
   clear when stopping or handing off.

Task compliance and research decisions belong to the agent. The task script is
an executable measurement recipe, not a boundary adjudicator or an agent loop.
Test of Progress is run separately by the benchmark owner when requested.

Adapted from [program2.md](https://github.com/Lumin-Science/Nano-Protein-LM/blob/9db6e85f3453a4d9ed25b775c649dbfe630d60d0/program2.md)
on `autoresearch-171m-val-loss`. Its protein-specific settings and campaign paths
remain historical; the selected task is the authority for the current experiment.
