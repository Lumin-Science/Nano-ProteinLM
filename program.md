# Protein architecture and optimizer autoresearch

Status: ready for harness setup; experimentation has not started.

Branch: `auto-research`

Frozen parent: `89e995b1b9a6e7a516fbf2c90a714a182c50ef25`

## Purpose

Autonomously discover model-architecture and optimizer changes that improve the
quality of a frozen protein sequence embedding under a fixed compute budget.
The target is biological transfer, not masked-language-model loss by itself.

This program adapts the small-surface, fixed-budget, baseline-first experiment
loop in [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and its
[`program.md`](https://github.com/karpathy/autoresearch/blob/master/program.md).
The retained principles are:

- freeze data preparation and evaluation;
- expose a deliberately small mutable training surface;
- run the unmodified baseline first;
- give every experiment the same declared compute budget;
- commit, run, measure, and record every idea;
- keep simple improvements and reject complexity without evidence;
- use `uv` as the only Python environment and command runner.

There is one important departure from the language-model example. Protein
embedding quality is a vector of biological-transfer and contact metrics. No
new scalar aggregate may be invented merely to make autonomous selection easy.
The loop uses a preregistered dominance gate for screening and the exact
P-CORE-Q4 plus full contact protocol for promotion.

## Scientific objective

Find a sequence-only encoder that, at matched training compute:

1. improves the frozen-probe representation vector;
2. improves or preserves long-range contact P@L;
3. does not materially regress held-out MLM NLL;
4. remains operationally feasible on four A100 80GB GPUs;
5. is simple enough to audit, reproduce, and eventually release.

The current local reference is the fresh 84,000-step ESMC-300M Stage-1 run:

| Quantity | Frozen reference |
|---|---:|
| Parameters | 332,997,184 |
| Filled residues | 5,123,026,423 |
| Held-out MLM NLL | 2.50497 |
| P-CORE-Q4 v0.3 | 30.66093 |
| Full contact P@L | 0.16357 |
| Training seed | 20260822 |
| Checkpoint SHA-256 | `b00b3d3784b9d08cd960f766441c8fa0a79827b10630aae702dd2c7a8561700c` |

This reference is a local scaling baseline, not a quality target already met:
released ESMC-300M scores 37.4847 on the same local P-CORE-Q4 protocol and
0.5387 full contact P@L.

## Claim boundary

An autonomous screen can produce a **provisional candidate**, never a release
claim. A promoted candidate requires the exact evaluation, multiple training
seeds, immutable receipts, and matched-compute accounting described below.

Enzyme Commission and Human PPI are always reported when the exact suite runs,
but both remain quarantined and cannot select a model. P-CORE v0.5-alpha is also
diagnostic only until a replacement panel passes qualification. Do not optimize
against quarantined or unqualified tasks.

## Immutable experiment contract

The following are human-owned and read-only during autoresearch:

- `uv.lock`, `pyproject.toml`, and the installed dependency set;
- `nano_protein/data.py`, `nano_protein/tokenizer.py`, and all corpus builders;
- `nano_protein/evaluate.py`, evaluation scripts, split ledgers, task data, null
  references, bootstrap rules, probe classes, metrics, and trust labels;
- the production data manifest, decontamination receipts, sequence order, source
  pool, tokenizer, 15% mask-all corruption, and sequence-mean MLM loss;
- `configs/esmc_300m_stage1_4xa100_4h.yaml` and
  `configs/esmc_300m_stage1_4xa100_16h.yaml`;
- the archived `results/` baseline receipts and the paper in `report/`;
- GPU count, GPU class, screening duration, evaluation examples, evaluation
  profile, and seed assigned to each rung.

Never edit evaluation code to make a candidate look better. Never replace a
failed metric with a proxy after seeing the result. Never change data, context,
masking, batch semantics, or the compute clock in the same comparison unless
that factor is the preregistered object of a separate study.

## Mutable surface

Keep experimental diffs inside this surface:

- `nano_protein/model.py` for architecture and initialization;
- `nano_protein/train.py` for optimizer construction, gradient handling, and
  optimizer-step semantics only;
- `nano_protein/schedule.py` for learning-rate or optimizer schedules;
- an optional new `nano_protein/optim.py` containing an in-tree optimizer;
- `configs/autoresearch/` for experiment-only configurations;
- focused tests under `tests/` for every new contract.

Do not add a dependency. If an optimizer cannot be implemented clearly with the
uv-locked PyTorch and NumPy stack, do not test it. Do not edit architecture and
optimizer in the same first-order trial. Establish their effects separately;
combine only retained changes.

## Required setup before the loop

1. Confirm `git branch --show-current` prints `auto-research` and that the
   worktree started at frozen commit `89e995b`.
2. Run `uv sync --frozen`. Do not use `pip`, Conda, or an ambient environment.
3. Run the complete fast test suite with `uv run --frozen pytest -q`.
4. Verify the production corpus rather than rebuilding it. The accepted data
   root must contain the same manifest, corpus-verification, and homology-
   exclusion digests as the frozen 16-hour receipt.
5. Add a screen configuration under `configs/autoresearch/`. It must use Stage
   1, context 512, the 36:11:54 source mixture, four A100s, BF16, the frozen
   tokenizer/loss, a fresh output root, and a synchronized 1,800-second training
   clock. Start with the baseline architecture, AdamW recipe, and seed 20260823.
6. Make the screen self-terminating through the trainer's synchronized clock.
   Startup, checkpoint writing, and evaluation remain outside training time.
7. Initialize an untracked tab-separated `autoresearch/results.tsv` with the
   schema below. Never commit logs, checkpoints, caches, or raw data.
8. Run the unmodified 1,800-second baseline before proposing any change. Then
   repeat the same baseline with seed 20260824. These two rows calibrate
   training-seed variation and the screen's numerical tolerances.

Do not begin candidate experiments if either baseline crashes, hashes the wrong
data, uses the wrong number of GPUs, fails tests, or produces incomplete routine
evaluation coverage.

## Fixed screening rung

Every screen uses:

- four A100 80GB PCIe GPUs;
- exactly 1,800 synchronized training-loop seconds;
- the verified 9M-protein Stage-1 pool;
- context length 512 and source mixture 36:11:54;
- one declared seed, initially 20260823;
- held-out MLM plus the frozen `speedrun` evaluation profile;
- a fresh, content-addressed output directory;
- the same uv lock and CUDA/Torch contract.

The fixed wall clock intentionally rewards designs that turn the same hardware
time into better representations. Always report optimizer steps, filled
residues, model tokens, estimated FLOPs, parameter count, tokens/s, and peak
memory so a quality gain can be distinguished from a throughput tradeoff.

Architecture screens should remain in the 300M class, defined initially as
300M--366M trainable parameters. Anything outside that band is a separate
scaling study and cannot replace the 300M incumbent from a single screen.

## Screening measurements

Record this frozen vector for every completed checkpoint:

- held-out sequence-mean MLM NLL, lower is better;
- remote-homology balanced accuracy, higher is better;
- FLIP2 hydro Spearman rho, higher is better;
- frozen contact-subset P@L, higher is better;
- filled residues and model tokens;
- optimizer steps and estimated training FLOPs;
- tokens/s and peak CUDA memory;
- parameter count and wall-clock training seconds.

The routine representation tasks have no P-CORE aggregate. Human PPI must not
be reintroduced into the routine gate.

## Keep, confirm, discard

After the two baseline seeds, derive one tolerance per screen metric as the
larger of the observed absolute seed difference and the evaluator's documented
numerical tolerance. Freeze those tolerances in the first two rows of the
experiment log; do not revise them after seeing candidates.

A seed-20260823 trial is a `provisional-keep` only when all of these hold:

1. tests, training, checkpoint receipts, and every routine metric complete;
2. no biological metric regresses by more than its frozen tolerance;
3. at least two of remote homology, FLIP2, and contact improve by more than
   their frozen tolerances, or one improves by at least twice its tolerance
   while the other two do not regress beyond tolerance;
4. MLM NLL does not regress by more than its frozen tolerance;
5. peak allocated memory is at most 76 GiB and no rank approaches OOM;
6. the change has a clear mechanism and a reviewable diff.

Run every provisional keep unchanged with seed 20260824. It becomes `keep` only
if the mean two-seed biological vector still satisfies the same rule and the
direction of each claimed gain agrees across seeds. Otherwise mark it
`inconclusive` or `discard`.

Do not hide a regression inside a weighted average. When two candidates are
non-dominated, retain both as separate hypotheses until the next rung. Prefer
the simpler candidate when metrics are within tolerance.

## Promotion ladder

The autonomous screen is Rung 0. Promotion proceeds in increasing cost:

| Rung | Training | Evaluation | Required decision evidence |
|---|---|---|---|
| 0 | 1,800 s, seeds 20260823/24 | MLM + frozen routine vector | provisional screen only |
| 1 | matched 21,000-step baseline FLOPs, at least two seeds | all six raw probes, Q4, full contact | candidate must beat the matched baseline without trusted-task regression |
| 2 | matched 84,000-step baseline FLOPs, at least two seeds | exact full suite, throughput and memory | local 300M candidate decision |
| 3 | independent locked rebuild | exact full suite plus release audit | release-eligible evidence |

At Rungs 1--3, match the baseline's estimated `6 * parameters * model_tokens`
budget and report optimizer steps, residues, and wall time. For an unchanged
parameter count this reduces to the same model-token/step budget; when an
architecture changes parameter count, adjust its token cap rather than quietly
granting more FLOPs. A candidate cannot advance on lower MLM NLL alone. It must
improve P-CORE-Q4 and contact jointly, with no material regression on any
trusted Q4 task under paired evaluation uncertainty. Training-seed uncertainty
is reported separately and requires multiple independently initialized runs.

## Architecture research queue

Change one conceptual factor per first-order experiment. Start with hypotheses
that are specific to the current implementation:

1. **Residual scaling discrepancy.** Compare the released
   `sqrt(n_layers / 36)` convention with the paper's stated `sqrt(n_layers)`
   convention and a depth-aware initialization that preserves update scale.
2. **Depth/width allocation.** Within the 300M parameter band, trade layers for
   width while retaining 64-dimensional heads and report tokens/s and contact.
3. **FFN allocation.** Vary the rounded SwiGLU expansion while compensating
   depth or width to remain in band.
4. **Normalization.** Test bias-free LayerNorm or RMSNorm consistently at the
   attention/FFN pre-norm sites. Do not silently alter Q/K normalization in the
   same trial.
5. **Q/K normalization granularity.** Compare full-projection and per-head Q/K
   normalization with matched dimensions.
6. **Initialization.** Test residual-aware output projection scaling and
   embedding/head initialization without changing optimizer settings.
7. **MLM head.** Compare the released two-layer GELU head with a simpler tied or
   linear head, accounting for parameters and ensuring the encoder, not a large
   head, receives the benefit.
8. **Attention/FFN balance.** Only after the above, test grouped-query or other
   attention variants that remain compatible with the qualified packed kernel.

Avoid large bundles such as "new norm + new optimizer + new width." They cannot
identify causality and are difficult to revert or reproduce.

## Optimizer research queue

Use the frozen architecture while calibrating optimizer changes:

1. **AdamW calibration.** Search peak learning rate and weight decay jointly,
   preserving the declared LR-times-decay hypotheses where relevant.
2. **Momentum dynamics.** Test beta1, beta2, and epsilon in bounded factorials;
   report early instability and final gradient norms.
3. **Warmup and decay.** Compare warmup fractions and WSD/cosine decay while
   keeping the 1,800-second clock fixed. The schedule must be defined in steps
   or synchronized progress without rank-local time decisions.
4. **Parameter groups.** Test separate learning rates or decay for embeddings,
   norms, attention/FFN matrices, and the MLM head. Every grouping rule must be
   name-stable and covered by a test.
5. **Gradient control.** Compare global clipping with a clearly specified
   adaptive rule; log unclipped and applied norms.
6. **Hybrid Muon/AdamW.** If justified, implement a small in-tree Muon-style
   update for eligible two-dimensional transformer matrices and retain AdamW
   for embeddings, norms, biases, and the output head. Test optimizer state,
   DDP parity, checkpoint round-trip, and memory before a GPU run.
7. **State and update precision.** Explore optimizer-state precision only after
   numerical parity and resume correctness are tested. Never trade silent
   instability for apparent throughput.

Reject optimizer ideas that cannot resume exactly from their own checkpoint or
that make the run receipt unable to reconstruct parameter groups and schedule.

## Experiment procedure

Repeat until manually stopped:

1. Inspect the current branch, incumbent commit, and `results.tsv`.
2. State one falsifiable hypothesis and the expected mechanism.
3. Change only the allowed files and add or update focused tests.
4. Run `uv run --frozen pytest -q`.
5. Commit the candidate before training.
6. Launch it in a fresh output root and let the synchronized training cap stop
   it naturally. Redirect verbose output to a run log rather than flooding the
   agent context.
7. Verify the checkpoint, config, data, environment, and uv-lock digests before
   reading metrics.
8. Run the frozen routine evaluator without editing it.
9. Append one TSV row, including crashes and incomplete runs.
10. Apply the keep/confirm/discard rule.
11. If discarded, preserve the experiment commit and revert it with a new
    commit. Do not rewrite history or use `git reset --hard`.
12. Continue from the strongest simple non-dominated incumbent.

Never delete or overwrite an output root. Never reuse cached embeddings across
different checkpoint hashes. Never compare a partially completed evaluator to
a complete one.

## HPC and process safety

- Never call `scancel`, requeue, release, or terminate an allocation.
- Never kill an outer Slurm process or another user's process.
- Use a self-terminating training clock and the scheduler's natural job limit.
- If an experiment overruns or hangs, preserve logs and report it; do not end a
  shared allocation to recover.
- Do not assume a GPU is free from low utilization alone. Verify ownership and
  allocation before launching.
- A naturally ended allocation does not authorize changing the scientific
  budget or continuing on different hardware under the same result row.

## Result log

`autoresearch/results.tsv` is tab-separated and untracked. Use this header:

```text
commit\texperiment\tparent\ttrack\tseed\tbudget_s\tparams_m\tsteps\tfilled_residues\ttrain_flops\tmlm_nll\tremote_ba\tflip2_rho\tcontact_pal\tpeak_gib\ttokens_s\tstatus\tdescription
```

Allowed statuses are `baseline`, `provisional-keep`, `keep`, `discard`,
`inconclusive`, and `crash`. A crash uses `NA` for unavailable metrics rather
than fabricated zeros. Descriptions must name the single factor changed.

For every row, retain machine-readable receipts in controlled storage with:

- candidate and parent commit;
- exact diff and hypothesis;
- config, data-manifest, checkpoint, evaluator, and uv-lock hashes;
- Python, Torch, CUDA, GPU, rank count, and attention backend;
- all raw metrics, timings, memory, source counts, and stop reason;
- the keep/discard decision and its rule-based justification.

## Simplicity rule

All else equal, simpler is better. Keep a deletion that matches the incumbent.
Require a larger, repeatable biological gain for a complex optimizer, custom
kernel path, or multi-file abstraction. Lines of code are not the only cost:
checkpoint compatibility, test burden, memory, throughput, and ease of
independent reproduction all count.

## Stop conditions

Pause candidate generation and surface the evidence when any of these occurs:

- the frozen harness or data receipt is invalid;
- routine metrics fail to discriminate the two frozen baseline seeds reliably;
- three consecutive candidates fail for the same infrastructure reason;
- a required action would edit data/evaluation, change dependencies, terminate
  an allocation, or exceed authorized compute;
- a candidate reaches Rung 1 and requires the expensive exact campaign.

Reaching a stop condition is not permission to weaken the protocol. Preserve
the worktree and logs, state the blocker precisely, and wait for human action.

## Definition of success

The autoresearch program succeeds only when it produces a small, reproducible
architecture or optimizer change that survives two screening seeds, improves
the exact P-CORE-Q4/contact decision at matched compute, passes all tests and
receipts, and remains understandable enough to merge deliberately into the
locked main line.
