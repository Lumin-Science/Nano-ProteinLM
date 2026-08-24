# P-CORE v0.5 alpha: preregistered validation probes

Status: **scientist-requested execution, frozen before new-task rankings**  
Frozen: **2026-08-22**  
Protocol: `pcore-v0.5-alpha-q9`  
Environment: all Python environments and commands are managed by `uv`.

This campaign implements and exercises the proposed nine-task evaluator, then
compares the released ESMC-300M, ESMC-600M, and ESMC-6B checkpoints. It is a
bounded, one-shot validation campaign, not a benchmark-tuning exercise. The
five proposed additions remain provisional until the qualification controls in
`FORMULATION.md` pass.

## Frozen shared contract

- Evaluate the same immutable row IDs for every model.
- Use final-layer residue states before the LM head; remove special/padding
  tokens; mean-pool valid residues for global tasks.
- Use deterministic non-overlapping windows of at most 2,046 residues and a
  residue-count-weighted protein mean.
- Cache by model revision/weight digest, sequence digest, layer, pooling,
  window policy, dtype, and hidden dimension.
- Do not fine-tune encoders, search layers/pooling, or add nonlinear heads.
- Standardized logistic/ridge probes use fixed `C=1`/`alpha=1`; cosine kNN uses
  `k=5` with similarity weighting. These values are not changed after results.
- Probe seeds are `20260822`, `20260823`, and `20260824`; embeddings are shared.
- The primary standardized logistic and ridge solvers are convex and deterministic
  under this implementation. `20260822` is therefore the point-estimate seed;
  `20260823` and `20260824` are reserved for preregistered null/permutation
  controls rather than redundant refits of an identical convex solution.
- Exact reports use 10,000 paired hierarchical bootstrap resamples with frozen
  resample IDs. Development smokes may use 100 resamples and are never reported
  as final estimates.
- Preserve native task metrics. The alpha summary reports median and lower-
  quartile null-normalized gain, but no alpha result can promote a model.

## Probe 1 — source, manifest, and split reconstruction

**Assumption.** Each proposed task can be reconstructed from an authoritative,
versioned artifact with enough biological groups and no prohibited probe-role
overlap.

**Action.** Build metadata and sequence manifests for all nine tasks. Every
receipt records source URL/revision, byte hash, row count, unique sequence and
residue counts, split counts, label prevalence, biological group counts, and
exact split intersections. For restricted data, publish acquisition metadata
and hashes rather than payloads.

**Falsifiers.** Missing official artifact/version, irreproducible split,
prohibited overlap, underpowered hard population, or a protocol that cannot be
implemented from the released fields.

**Consequence.** A failed task remains executable only as `blocked` or
`quarantined`; it receives no fabricated value and the nine-task summary is
marked incomplete.

## Probe 2 — evaluator and null-control smoke

**Assumption.** The implementation respects split boundaries, fits only on
train/validation data, reproduces deterministic results, and returns null-like
performance when labels are permuted at the biological-unit level.

**Action.** Run unit tests plus one small real-data smoke per task. Repeat each
smoke exactly; run a frozen target-permutation control; validate result schemas,
finite metrics, task membership, row hashes, and cache resumption.

**Falsifiers.** Non-determinism, test-aware fitting, prediction/label mismatch,
target permutations materially above the preregistered null tolerance, or a
receipt that can be combined with a different manifest/model contract.

**Consequence.** Stop that task before released-model evaluation and record the
failure in the validation report; do not simplify the task in response to a
model result.

## Probe 3 — released ESMC three-scale panel

**Assumption.** The proposed evaluator can measure meaningful differences among
released ESMC-300M, ESMC-600M, and ESMC-6B representations under one protocol.

**Action.** Extract each unique sequence once per released checkpoint, fit the
frozen readouts for the three probe seeds, and compute task-native metrics and
paired biological-unit intervals. Reuse the already frozen Q4 artifacts only
when their row, split, metric, pooling, and model contracts exactly match this
alpha protocol; otherwise rerun them.

**Falsifiers.** A model-specific population, incomplete task receipt, changed
readout, silent OOM row deletion, or results that cannot be reproduced from the
recorded hashes.

**Consequence.** Report the complete raw vector and missing/failed tasks. Do
not impute, backfill from a different protocol, or rank by an incomplete q9
summary.

## Probe 4 — interpretation and qualification triage

**Assumption.** Observed differences survive simple controls and are not solely
length, composition, sequence identity, node degree, mutation count, one large
landscape, or one protein family.

**Action.** Report task-specific classical/confound baselines available from
the frozen manifests, seed sensitivity, paired deltas, effective group counts,
and obvious hardest-stratum retention. Compare strict scale monotonicity only as
a diagnostic; it is not an admission rule.

**Falsifiers.** Mature-model gains vanish against a confound baseline, model
ordering is probe-seed unstable, a few groups dominate, or a candidate cannot
distinguish the released scales within uncertainty.

**Consequence.** Assign `provisional`, `release_only`, `quarantined`, or
`blocked` with the observed reason. Only a later full qualification campaign
may assign `selection` to a new task.

## Required output

`VALIDATION_REPORT.md` must use the structure assumption → probe → observation
→ verdict → consequence, include exact commands, environment/model/dataset
receipts, all raw task metrics and intervals, paired model deltas, runtime and
peak-memory measurements, null/confound results, failures, and the next
qualification actions. It must dual-report the frozen Q4 result for continuity
and must call the new result `alpha`, not production P-CORE.

## Source-gated implementation clarification (before model rankings)

The authoritative artifacts forced the following concrete choices. They were
frozen after source inspection and before any new-task released-model score was
computed.

- CATH v4.4 S20 contributes 11,180 domains from 1,593 H groups with at least
  two domains. Within each H group, 20% (at least one) are hash-selected as
  queries and the rest are gallery. The primary metric macro-averages AP over
  H groups rather than letting large folds dominate.
- FLIP2 is bounded without looking at targets to at most 4,000 train, 1,000
  official-validation, and 5,000 test rows per landscape. The exact landscape
  score is `0.5*((Spearman+1)/2) + 0.5*NDCG`; landscapes are median-aggregated
  within each dataset and datasets are median-aggregated into the task vote.
  Its interval resamples datasets and then landscapes within each sampled
  dataset; it measures benchmark-panel uncertainty, not per-variant assay
  uncertainty.
- MegaScale uses only the official `train`, `val`, and `test` rows from
  `dataset3_single`: 50,000 hash-selected train, 10,000 hash-selected
  validation, and all 28,172 test variants. Wild type is reconstructed by
  reversing the declared single mutation, features are mutant minus wild-type
  embeddings, and non-test parents in a test parent's MMseqs2 30%-identity
  cluster are removed before bounding.
- CAID2 Disorder-PDB is split 80/20 by protein hash for probe training and
  validation; all 319 CAID3 Disorder-PDB proteins remain temporal test.
  Any CAID2 identifier or exact sequence appearing in CAID3 is removed before
  probe fitting (this excludes `DP02732` from CAID2 train). Unlabelled `-`
  residues are excluded. The primary AP is macro-averaged over eligible test
  proteins; pooled AP and MCC remain diagnostics.
- PRING uses the official Human BFS node split and fixed released negatives.
  MMseqs2 at 30% identity and 80% bidirectional coverage removes non-test nodes
  sharing a cluster with test nodes. The full 64,038-pair test remains, but the
  task is provisional because the release supplies one negative realization,
  not the proposed ten. Its AUPRC interval uses a symmetric two-endpoint
  identity-cluster bootstrap: every pair receives the product of its sampled
  endpoint-cluster weights, so uncertainty is invariant to pair orientation.
- The CAFA5 MF no-knowledge population shrinks from 541 to 110 proteins under
  the frozen 30%-identity/80%-query-coverage screen against MF-labelled
  training proteins. This fails the preregistered minimum of 500, so CAFA is
  blocked, receives no model score, and makes the alpha q9 aggregate incomplete.
- Released-model extraction uniformly uses final ESMC backbone states with
  cross-protein length-aware batching. The path skips the unused LM head and
  all-layer hidden-state tuple. Batched bf16 outputs are not mixed with legacy
  serial caches; old Q4 results are dual-reported as continuity results only.
