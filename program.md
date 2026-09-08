# Protein AutoResearch task contract — v1

**Task:** `protein-embedding-recipe-v1`. The five sections below define the
experiment. [tasks/protein-embedding.yaml](tasks/protein-embedding.yaml) is its
structured counterpart; the [shared task standard](docs/AUTORESEARCH_TASK_STANDARD.md)
explains how to use the same format for other tasks, including OpenMM.
The operational appendix retains the existing research commands and ledger.
[program2.md](program2.md) is an immutable historical campaign definition.

## 1. Research question

**Which training-recipe changes produce more useful protein representations
at approximately fixed model size and data?** Search under a fixed training-time
budget, then test whether improvements transfer to a matched token budget.

The comparator is the repository's original ESMC-like AdamW baseline. Held-out
masked language-model (MLM) loss is the search proxy. The practical test asks for
both better MLM loss and better long-range contact P@L. P@L uses attention maps;
it does not by itself establish better performance on every downstream embedding
task. P-CORE supplies additional frozen-representation diagnostics.

## 2. What the codebase establishes

| Component | Established implementation |
|---|---|
| Baseline | [Original ESMC-171M](configs/esmc-171m-original.yaml): 24 layers, width 768, 12 heads, FFN 2048, 170,671,168 parameters; AdamW, LayerNorm, RoPE 10k, untied embeddings |
| Training inputs | Pinned, decontaminated UniRef90, MGnify and OMG/IMG data, fixed tokenizer, mixture sampling and masked-target corruption |
| Training | Distributed BF16, FA2/FA3, synchronized wall-time and step stopping, full checkpoints/optimizer state; model, optimizer, batching and loss experiments |
| Evaluation | Frozen sequence-mean MLM evaluator; full 20,775-chain contact evaluation with fixed probe procedure and bootstrap; checkpoint/data/environment receipts |
| Evidence | [Two-seed research history](reports/program2/README.md) and [completed 100k-step tests](README.md#test-leaderboard), with their original protocols preserved |

The public recipe is an ESMC-like reconstruction, not the paper's exact corpus
or calibrated recipe. **The current trainer stops on time or steps, not tokens.**
Section 5 defines the requested new token-budget protocol; its runner adapter
must be implemented and verified before that phase can be launched. The task
YAML is a specification, not input to `nano_protein.train`.

## 3. Research protocol and search score

| Item | Fixed definition |
|---|---|
| Execution | Train from scratch on **4 NVIDIA L40S GPUs**, then evaluate the final checkpoint |
| Budget per seed | **3,600 seconds** of synchronized training-loop wall time; no evaluation inside this budget |
| Repeats | **N=2 by default**, seeds **42 and 43**; same predeclared seed set for baseline, incumbent and candidate |
| Baseline batch / context | Global batch 256; context 512. Candidate batch size may change; context and sampling semantics stay fixed |
| LR schedule | Exactly **554 optimizer steps** of linear warmup, then constant peak LR in every parameter group |
| Search metric | `sequence_mean_nll` from `eval-validation/VALIDATION_MLM.json`, **minimize** |
| Search evaluation | **32 fixed held-out sequences**, 8 batches × 4, context 512, seed **20260821**, equal source weights |
| Required diagnostics | Training loss, full 20,775-chain P@L, tokens/steps, parameter count, memory and timing |

Call this the **search score**, rather than reward. It is separate from the
candidate's backward training loss. Within each validation sequence, average
cross-entropy over its masked targets; then average over sequences. The method
score is the arithmetic mean of these per-run losses over all N training seeds.
Compute sample SD over seeds with `ddof=1`.

Keep the existing acceptance rule:

```text
improvement = incumbent_mean_validation_loss - candidate_mean_validation_loss
keep iff all runs are valid and improvement > candidate_sample_sd
```

The incumbent is the last accepted recipe, not the lowest noisy observation.
A tie fails; a single run cannot qualify. Training loss and P@L do not affect
research selection. The rule is a noise-margin heuristic, not a significance
test. Report all repeats and failures; never select favorable seeds or rerun to
replace an unfavorable completed result. A change to N or the rule requires a
new protocol version and a matched baseline/incumbent comparison.

The clock starts at the synchronized pre-loop barrier. Batching, transfers,
forward/backward, optimization, communication, logging and lazy first-use work
inside the loop count. Setup, final checkpoint serialization and subsequent
evaluation do not. Stop at the first optimizer-update boundary after the limit,
record the actual duration/overrun, and require `stop_reason=walltime` plus
`ROUND_STATE=complete`. No checkpoint selection by evaluation score.

Contact evaluation uses the fixed 16-chain probe fit / 4-chain selection and
20-chain refit procedure; all 20,775 reporting chains stay outside probe fitting.
Pair sampling uses seed 20260819; chain ordering and the 5,000-resample 95%
bootstrap use seed 20260820. Evaluation assets, seeds and reductions are frozen.
See [the evaluator contract](docs/EVALUATION.md).

## 4. Experiment boundaries

These rules apply **during candidate research**. Contract maintenance between
campaigns may change them, with a new version and comparable baseline. All paths
are repository-relative. Protected entries win; unlisted files are protected.
The [task YAML](tasks/protein-embedding.yaml) contains the complete path/key lists.

| May change | Purpose |
|---|---|
| `nano_protein/model.py` | Model architecture, initialization and parameter grouping, within the parameter bound |
| `nano_protein/train.py` | Optimizer, backward objective and training execution, subject to the frozen semantics below |
| `nano_protein/batch_balance.py`, `nano_protein/flash_attention.py` | Batch redistribution and attention execution under the same data/model contract |
| `nano_protein/experiments/**`, `configs/candidates/**`, `tests/candidates/**` | Candidate helpers, configs and additional tests |
| `.dev/program2/**`, fresh `outputs/program2/<run-id>/**` | Candidate wrappers, notes and new immutable output records |

| Must not change | Protected paths or assets |
|---|---|
| Task definition and evidence | `program.md`, `program2.md`, `tasks/**`, `README.md`, `docs/**`, `reports/**`; previous run outputs |
| Frozen baseline/history configs | `configs/esmc-171m-original.yaml`, `configs/program2/**`, `configs/program2_h100_100k/**`; other configs are protected unless explicitly allowed |
| Data and tokenizer | `nano_protein/data.py`, `nano_protein/sharded_data.py`, `nano_protein/tokenizer.py`; downloaded corpora, manifests, mixture/source weights and evaluation fixtures |
| Evaluator and execution contract | `nano_protein/evaluate.py`, `nano_protein/pcore_task.py`, `nano_protein/contact_cache.py`, `runs/**`, `scripts/**`, external evaluation source |
| Schedule and lifecycle | `nano_protein/schedule.py`, `nano_protein/resume.py`, `nano_protein/periodic_evaluation.py`; budget, sampling and receipt logic inside editable training code |
| Environment and existing tests | `pyproject.toml`, `uv.lock`, `.gitmodules`, `tests/test_*.py` |

Additional invariants:

- Keep **actual trainable parameters within ±5% of 170,671,168**: inclusive
  **162,137,610–179,204,726**. Count real model tensors; no dummy parameters.
  The historical 142M narrowed-FFN runs predate this rule and cannot qualify
  under v1. Compensating changes to width/depth are allowed within the bound.
- Keep the corpus release and its verified prepared data fixed. Stage-1 source
  weights are UniRef90 0.36, MGnify 0.11, OMG/IMG 0.54, normalized by their sum.
  Preserve context 512, tokenizer, crop/mask semantics and evaluation exclusion.
  No new data, distillation targets, pretrained initialization or held-out training.
- Hyperparameters, optimizer, trainable model and backward loss may change.
  Batch-size/accumulation changes and redistribution are allowed; replacing,
  filtering or resampling examples to favor benchmark outcomes is not.
- Editable training code cannot alter the clock, stopping criteria, token meter,
  frozen scheduler, data semantics, metric extraction or provenance checks.
  Shared model code cannot detect evaluation identities or substitute answers.
  Preserve normal checkpoint loading and inference semantics.
- Keep hardware class/count and the campaign-pinned environment fixed. Use
  `uv sync --frozen` / `uv run --frozen`. Record any allowed execution backend
  choice within that environment. Other hardware is a separate comparison.
- Pin the code/contract, corpus, external evaluator, dependency and environment
  receipts before running; use fresh outputs and retain failures. Shared input
  caches are read-only. Existing checks plus source review enforce this policy;
  neither a file list nor schema validation is a complete runtime sandbox.

## 5. Test protocol and success criteria

The **test** measures transfer at fixed training exposure. Freeze the candidate
commit and transfer configuration before training from scratch. Apply the same
protocol to the original baseline and each preceding cumulative recipe.

| Item | v1 test definition |
|---|---|
| Training budget per seed | **24,200,224,761 global non-padding model tokens**, including BOS/EOS, excluding padding and evaluation |
| Token meter | Sum input `attention_mask` across ranks/microbatches; count consumed inputs once, independently of recomputation or rank redistribution |
| Stop boundary | First completed optimizer update at or above target; report actual tokens and overrun, less than one update (at most 524,288 tokens at batch 1024/context 512) |
| Hardware / execution | **4 H100**, BF16, FA3; matched pinned environment |
| Transfer settings | Global batch **1,024**, context **512**, Stage 1, base LR **5e-4**, base WD **0.01**, **1,000-step warmup**, then constant LR |
| Candidate group settings | Preserve and report its declared optimizer-group multipliers; freeze before tests |
| New v1 repeats | **N=2**, seeds **42, 43**, matched across all tested recipes; report every seed |
| MLM evaluation | **4,096 sequences**, 1,024 batches × 4, context 512, seed 20260821; fixed sequence-mean NLL |
| Contact evaluation | All **20,775 chains**; same frozen probe, chain order and 5,000-resample chain bootstrap |
| Checkpoint | Final budget endpoint, including full optimizer/runtime state for future continuation |

The exact token target matches the observed exposure in the completed four-H100
comparison. **100k steps is historical context, not the new stop condition**;
new seeds can reach the token limit at different steps. A safety wall-time limit
is nonbinding: if it stops a run before the token budget, that run is incomplete.
A token-stop adapter and audited endpoint receipts are required before launch.

For each named comparator, define:

```text
loss_gain = comparator_mean_validation_loss - candidate_mean_validation_loss
contact_gain = candidate_mean_p_at_l - comparator_mean_p_at_l
successful transfer iff all runs are valid and loss_gain > 0 and contact_gain > 0
```

Report the two metrics separately, mean ± sample SD across training seeds,
paired seed deltas, and each seed's chain-bootstrap P@L CI. A tie or one-metric
regression does not meet this two-metric criterion. State whether a result beats
both the baseline and preceding recipe, or only one comparator. This directional
criterion is not a significance claim; chain CIs do not establish repeatability
across training seeds.

This is **confirmation at a larger training budget on reused evaluation assets**,
not a blind holdout: research already reports P@L on these chains and uses the
same held-out MLM reservoir. Final test metrics cannot select checkpoints or
modify the declared recipe during that test. Follow-up changes start a new
research iteration and retain their evaluation history.

The [published Test Leaderboard](README.md#test-leaderboard) remains a historical
100k-step, batch-1024, **single-seed (20260824)** comparison. It is not retroactively
an N=2 token-stopped test. Its Settings 1/2/3/5 use the recorded R02-based recipe;
Setting 1 contains more changes than the research Muon-only ablation. Keep that
qualification when comparing search and test results.

## Operational appendix: research isolation, execution and records

Create a separate worktree/branch, `autoresearch-171m-val-loss-<campaign-id>`,
from a pinned `main` revision for a new v1 campaign. Continue existing results
only under the same contract and seed set; otherwise establish a fresh baseline.
All paths below are relative to that worktree. Preserve historical campaigns'
branches, launchers, configs and results, including `autoresearch-171m-val-loss`. Share only verified read-only data/evaluation assets; reserve
four idle GPUs for each run.

| Purpose | Path |
| --- | --- |
| Results (untracked) | `.dev/program2/results.tsv` |
| Candidate launcher (untracked) | `.dev/program2/rounds.sh` |
| Candidate configs | `configs/candidates/` |
| Checkpoints, evaluation, and run logs | `outputs/program2/<unique-run-id>/` |
| Prepared data and download cache | `.exps/program2/` |
| UV cache | `.uv-cache/program2/` |
| Notes, plots, and continuation state | `.dev/program2/` |

Use `program2-171m-<campaign-id>` for any tmux session or wakeup identity. Resume only this
campaign's records and incumbent.

Start with `configs/esmc-171m-original.yaml`: the repository's original,
family-scaled ESMC baseline, with 24 layers, width 768, 12 heads, 170,671,168
parameters, AdamW, LayerNorm, RoPE base 10,000, and default residuals and
initialization. Preserve this config and `runs/autoresearch_4xl40s_1h.sh`.
The runner defaults to this original AdamW baseline and supplies the frozen
data release and evaluation contract. Set `CONFIG` explicitly for every
candidate and seed repeat.

Evaluate the baseline with at least two runs at campaign start, before
modifying training code. The command below launches the first run; use the pinned
`configs/program2/baseline_seed42.yaml` and `baseline_seed43.yaml` for the
default repeats. Run the second command with `baseline_seed43.yaml` and a fresh
output path. New seed identities require predeclared baseline copies.

```bash
CONTACT_ROOT=/path/to/frozen-contact-data \
EXTERNAL_SRC=/path/to/evaluation-source \
ARTIFACT_ROOT="$PWD/.exps/program2" \
UV_CACHE_DIR="$PWD/.uv-cache/program2" \
CONFIG="$PWD/configs/program2/baseline_seed42.yaml" \
OUTPUT_ROOT="$PWD/outputs/program2/baseline-<unique-run-id>" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash runs/autoresearch_4xl40s_1h.sh
```

Replace placeholders for the environment. Unset inherited `DATA_ROOT` and
`DATA_CACHE_ROOT` overrides unless they point to verified read-only inputs.
Candidate launchers use the same command with a config under `configs/candidates/`
and a fresh output directory. Never reuse a run directory or alter another
campaign's launcher.

### Results and loop

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
   Commit on the campaign branch as `keep` only when
   `incumbent_mean - candidate_mean > candidate_std`; otherwise record `discard`
   and restore only that candidate's changes. The accepted mean and seed set
   become the new incumbent reference.
4. Record failed training/evaluation or non-finite metrics as `crash`, with the
   cause, and restore only candidate changes. Continue from the incumbent until
   stopped.
