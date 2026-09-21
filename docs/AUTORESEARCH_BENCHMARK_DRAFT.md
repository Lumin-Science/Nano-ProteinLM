<div class="ai">

# AutoResearch benchmark standard: discussion draft

</div>

<div class="ai">

This is a proposal for the next benchmark version, not a change to active task scripts, the current keep rule or historical results. The current contracts remain [the validation-loss task](../tasks/171m-validation-loss.md), [the P@L task](../tasks/171m-p-at-l.md) and [the research loop](../autoresearch/program.md). The immediate default change is the owner's selection of separate Q/K/V Muon updates, documented in the [promotion record](../.dev/reports/cck-contact-ablations-100k-20260919/DEFAULT_PROMOTION.md).

</div>

<div class="ai">

## 1. Freeze a benchmark version before searching

</div>

<div class="ai">

Each campaign should record one protocol identifier with the baseline code and resolved-config hashes, dataset revision and materialized-manifest hashes, tokenizer, evaluation code and split receipts, dependencies, hardware and attention backend, parameter bound, training budget, seed list, score direction, checkpoint rule and acceptance rule. The baseline is an immutable snapshot of the default at campaign start. Updating `configs/default.yaml` starts a new baseline version; it does not relabel old results or allow a new implementation to reuse an old score. Keep a separately frozen reference for longitudinal comparisons.

</div>

<div class="ai">

Carry forward the current ±5% actual-trainable-parameter bound and protected data, tokenizer, evaluation, dependency and budget-accounting boundaries. Define permitted code/configuration changes explicitly. Preserve Stage-1 context and source mixture, exclude pretrained initialization and held-out training, and reject dummy parameters or changed scoring. Recipe innovation remains permitted inside that declared scope; a successful measurement command does not itself establish compliance.

</div>

<div class="ai">

## 2. Separate search efficiency from transfer performance

</div>

<div class="ai">

| Comparison | Question | Proposed primary budget | Primary checkpoint |
|---|---|---|---|
| Search | How much quality can a recipe achieve within the same training time? | Retain one synchronized training hour on four L40S GPUs per seed | End of the declared training-time budget |
| Confirmation | Does a selected change retain its benefit under a longer matched run? | Fix actual model-token exposure, batch, context and schedule across candidates | Declared token endpoint |
| Agent comparison | Which agent produces the best valid improvement under the same research resources? | Fix campaign GPU-hours, elapsed-time allowance, trial/retry allowance, agent model/settings and permitted tools | Best candidate selected under the frozen campaign rule, then independently confirmed |

</div>

<div class="ai">

The current owner-run Test of Progress uses a fixed 24.20B-token budget on four H100s per seed. The recent CCK study used 100k steps at batch 1,024 and consumed exactly 24,196,983,520 model tokens per arm; that is a matched fixed-step study, not the identical Test of Progress budget. Steps alone cease to establish equal exposure if batch size, sequence lengths or sampling change. Keep fixed-time, fixed-token and historical fixed-step results in separate tables, with actual steps, sequences, model tokens, training time and total elapsed time reported for each.

</div>

<div class="ai">

Define timing boundaries in the protocol. Preserve the current synchronized training-loop timer, which counts ordinary training overhead and excludes setup, final saving and evaluation. Periodic evaluation pauses and checkpoint frequency must be fixed across candidates. Report excluded time and total GPU occupation separately. Use the actual time budget rather than converting a short smoke-test speed into a stopping-step estimate. Record bounded endpoint overrun and the stop reason; a safety cap or node failure is not a completed benchmark run.

</div>

<div class="ai">

Use exclusive allocations of the declared GPU type/count and record CPU allocation, thread settings, backend, precision and environment qualification. Specify whether data caches start cold or are warmed consistently; keep evaluator caching rules and identity checks fixed. Remeasure a baseline when hardware or execution conditions change, and record repeated unchanged-recipe runs to expose execution variability. L40S/FA2 and H100/FA3 results belong to different hardware conditions even when their model and data recipes match.

</div>

<div class="ai">

## 3. Declare one reward and one checkpoint-selection rule

</div>

<div class="ai">

Choose P@L or validation MLM loss as the primary reward before a campaign; always report both. Do not change the objective after observing which metric improved. P@L stays a fraction in machine-readable records, with percentage points used for displayed differences. Keep MLM validation size and masking fixed within a benchmark version: the current search uses 32 sequences, while the long-run confirmation uses 4,096. Their loss estimates must be labeled by protocol rather than pooled.

</div>

<div class="ai">

Use the declared budget endpoint for the primary score. Periodic P@L every 10k updates and best-P@L checkpoints are useful diagnostics and continuation artifacts; best-over-training is a separate result. If checkpoint selection is part of a benchmark, freeze its frequency and selection split before running, then evaluate the selected checkpoint on a separate untouched confirmation split. Do not call a repeatedly consulted search panel an unseen final test set. Keep the existing frozen panel for historical continuity; any additional confirmation split needs its own version and decontamination receipts.

</div>

<div class="ai">

## 4. Distinguish training-seed variation from evaluation uncertainty

</div>

<div class="ai">

Use the same predefined training seeds for candidate and incumbent, retain every run, and report paired per-seed differences as well as means and sample SDs. Two seeds can remain a low-cost screen. For a proposed default promotion, use a fresh, preregistered seed set with at least three paired runs, preferably five if the budget permits. These counts are a proposed resource policy, not a guarantee that a confidence interval is precise. Record a practically meaningful improvement threshold in the primary metric's units before testing.

</div>

<div class="ai">

For confirmation, propose a paired across-seed interval on the candidate-minus-reference difference and require its lower bound to exceed the predefined threshold; reverse the difference for loss so positive always means improvement. Specify the estimator, assumptions, number of seeds and stopping rule before execution. A Student-t interval assumes independent, approximately normal seed differences. Chain bootstrap measures variation across evaluated proteins conditional on trained models; it cannot substitute for training-seed replication. Shared-chain comparisons should preserve chain pairing. Repeated adaptive selection needs an independent confirmation stage; a nominal per-trial interval alone is not a campaign-wide false-positive guarantee. The existing two-sided-overlap keep rule remains in effect until a replacement is explicitly adopted.

</div>

<div class="ai">

## 5. Make data capacity and interruptions auditable

</div>

<div class="ai">

Before training, verify sufficient unused records in every source for the longest permitted candidate under its sampling mixture, including a declared margin. Freeze resampling policy: either reject exhaustion or report and account for repeated exposure explicitly. Preserve split and homology exclusions. For matched ablations, retain the same sampler seeds and compare source counts, sequences and model-token histories at shared steps. The recent study's 110,160,958 available records were sufficient for 102.4M draws per run; availability must be checked again when the budget or batch changes.

</div>

<div class="ai">

Define restart policy before the campaign. An interrupted run may continue only with verified model, optimizer, sampler and RNG state and cumulative budget accounting, or it may restart from zero under a declared rerun allowance. Preserve incomplete attempts and record discarded compute. Do not select the most favorable retry. Recipe errors are invalid trials; infrastructure failures are separately labeled, with both consuming the resources specified by the agent-comparison budget. Changing data, optimizer partitioning or schedule creates a new experiment rather than an invisible continuation.

</div>

<div class="ai">

## 6. Require matched controls before attributing a gain

</div>

<div class="ai">

Compare the previous default, the proposed default and one-component removals under the same long-run contract. For two proposed additions, a four-arm design identifies the old default, each addition alone and both together. The completed three-arm CCK study omitted the old default with both additions disabled: it measured the benefit of separate Q/K/V conditional on centering, and the benefit of centering conditional on separate Q/K/V. It did not directly measure separate Q/K/V's standalone gain over the previous default. The owner's present selection is supported by nearly tied P@L and slightly lower MLM loss for the simpler Q/K/V-only recipe, with one-seed uncertainty still unresolved.

</div>

<div class="ai">

## 7. Keep the evidence and implementation reviewable

</div>

<div class="ai">

Every valid result should carry code/config/data/environment identities, exact commands, start and stop reasons, actual budget use, all seed metrics, evaluator settings, final and selected-checkpoint hashes, and a boundary review. All planned seeds and evaluations must complete before a score qualifies. Archive unsuccessful and interrupted trials alongside successful ones. Promote only the implementation actually needed by the chosen recipe, preserve old configuration behavior and resume compatibility, and remeasure the new baseline for the next campaign. Keeping GitHub on main does not require deleting local experimental worktrees or historical evidence.

</div>

<div class="ai">

## Decisions to settle before implementing this standard

</div>

<div class="ai">

The owner still needs to select the confirmation seed count and compute allowance, the minimum useful P@L/loss improvement, the independent confirmation panel, and whether agent comparisons constrain total elapsed time as well as GPU-hours. The next useful step is a short protocol document that fixes those choices and explicitly versions any replacement for the current keep rule.

</div>
