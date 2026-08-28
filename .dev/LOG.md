# Development log

This is the public development log. It records what was tried, what worked,
what did not work, and what remains open. This log and the canonical technical
report are tracked; detailed working notes, raw receipts, cluster launchers,
and local paths remain gitignored under `.dev/`.

Results here are fixed-checkpoint measurements, not claims about training-seed
uncertainty unless explicitly stated. The public contracts remain
[`DATA.md`](../docs/DATA.md), [`EVALUATION.md`](../docs/EVALUATION.md),
[`BASELINES.md`](../docs/BASELINES.md), and
[`AUTORESEARCH.md`](../docs/AUTORESEARCH.md).

## 2026-08-28 — Public/private development boundary

**Worked**

- Consolidated the public development history into this log.
- Restored detailed development artifacts to the repository-local `.dev/`
  workspace so experiments remain recoverable without publishing local setup.
- Kept raw task metrics and negative results visible rather than reporting only
  an aggregate.

**Did not work / changed**

- A large public `dev/` tree mixed active plans, superseded proposals, generated
  files, raw receipts, and workstation-specific launch scripts. It was not a
  stable public contract and is no longer tracked.

## 2026-08-28 — Technical-report consolidation

**Worked**

- Rebuilt the technical report as one six-page Nano-ESMC paper with
  one corpus pipeline, one evaluation contract, and one consolidated baseline
  and qualification section.
- Made the 665,970,495-protein public release the only supported corpus in the
  main narrative and retained the older 9-million-protein runs only as a
  clearly labeled historical duration check.
- Standardized the active evaluator name to P-CORE and the unfinished expansion
  to P-CORE Next alpha. Historical receipt version names remain in this log
  where needed for traceability.

**Did not carry forward**

- Removed the legacy six-task aggregate, duplicated baseline tables,
  hypothetical eight-H100/seven-day planning, superseded release gates, and
  repeated data-processing explanations from the active paper.
- Preserved the original source and rendered PDF in the ignored development
  archive rather than treating them as current documentation.

## 2026-08 — Training-corpus reconstruction

**Worked**

- Reconstructed the UniRef90, MGnify, and JGI-role source pipeline from
  4,180,155,464 input records to 765,290,002 source-wise 70%-identity
  representatives.
- Froze a 317,000-sequence protected evaluation union and removed 21,653 exact
  matches plus 78,439,147 additional homologs from the representative pool.
- Published 665,970,495 decontaminated training proteins in 565 immutable
  training shards. The complete accounting is in [`DATA.md`](../docs/DATA.md).

**Did not work / limitation**

- The ESMC paper's July 2023 JGI snapshot is not available as a reproducible
  public download. Nano-ESMC therefore uses the public OMG/IMG arm as a
  surrogate.
- The surrogate provides 324,923,979 70%-identity representatives versus
  approximately 2 billion reported for ESMC's JGI arm. This roughly
  1.68-billion-representative gap is the main corpus-scale difference and
  prevents a claim of exact data reproduction.

## 2026-08-22 — P-CORE v0.5-alpha qualification

The alpha campaign reconstructed six candidate sources, froze common
manifests, evaluated the released ESMC-300M, ESMC-600M, and ESMC-6B
checkpoints, and used 10,000 paired biological-unit bootstrap resamples.

| Candidate / primary metric | ESMC-300M | ESMC-600M | ESMC-6B | Outcome |
|---|---:|---:|---:|---|
| CATH 4.4 macro-H mAP | 0.1170 | 0.1067 | 0.1153 | Diagnostic; not scale-monotone |
| PRING Human AUPRC | 0.7035 | 0.7001 | 0.7113 | Provisional; bootstrap intervals failed centering |
| FLIP2 hierarchical score | 0.7823 | 0.7581 | 0.7805 | Diagnostic; limited independent landscapes |
| MegaScale median-parent Spearman | 0.6835 | 0.7267 | 0.7207 | Qualification candidate |
| CAID3 macro-protein AP | 0.7292 | 0.7341 | 0.7340 | Temporal diagnostic; released scales unresolved |
| CAFA5 molecular function NK30 | blocked | blocked | blocked | Only 110 hard test proteins survived the frozen screen; minimum was 500 |

**Worked**

- All three released checkpoints completed against identical task manifests.
- MegaScale was the only new task with a clear paired 6B-over-300M signal:
  +0.0372 with a 95% interval of [0.0064, 0.1171].
- The campaign exposed task-specific failures instead of hiding them through
  imputation or aggregate-score changes.

**Did not work / decision**

- No runnable candidate was strictly monotone across all three model scales.
- PRING used one negative realization and produced non-centered percentile
  intervals, so its apparent signal cannot qualify a model.
- CAFA5 failed the preregistered population-size gate; the gate was not relaxed
  after seeing the data.
- CATH, FLIP2 shift, and CAID3 did not establish reliable scale separation.
- No P-CORE v0.5 aggregate or released-model ranking was issued. The existing
  four-task core remains the reference while a successor is qualified.

## 2026-08-21 — Four-task P-CORE reference

The retained receipts label this evaluator **P-CORE-Q4 v0.3**. No result
artifact labeled “P-CORE v0.4” was found, so the values are disclosed under
their recorded name rather than retrospectively relabeled.

| Trusted task / metric | ESMC-300M | ESMC-600M | ESMC-6B |
|---|---:|---:|---:|
| Remote homology / balanced accuracy | 0.1159 | 0.1152 | 0.1186 |
| Secondary structure / residue macro-F1 | 0.8315 | 0.8407 | 0.8780 |
| DeepLoc2 / macro average precision | 0.6442 | 0.6568 | 0.6942 |
| FLIP2 Hydrophobic Core / Spearman correlation | 0.4132 | 0.4276 | 0.4616 |
| **P-CORE-Q4 v0.3** | **37.4847** | **38.1553** | **40.6011** |

**Worked**

- The four-task aggregate is ordered ESMC-300M < ESMC-600M < ESMC-6B.
- Secondary structure, DeepLoc2, and FLIP2 improve at every released scale.
- Tasks use frozen encoders, fixed low-capacity readouts, held-out tests, and
  biological-unit resampling.

**Did not work / decision**

- Remote homology is nearly flat and dips slightly at 600M; it remains useful
  as a raw axis but is weak evidence by itself.
- Enzyme Commission produced 0.7174, 0.7096, and 0.0237 macro-AP across 300M,
  600M, and 6B. The implausible 6B collapse made it ineligible for selection.
- The reconstructed Human PPI test had only 237 pairs and produced 0.8155,
  0.8026, and 0.8345 average precision. Its small, easy, non-monotone behavior
  made it ineligible for selection.
- EC and Human PPI remain reported diagnostics but do not enter P-CORE-Q4.

## 2026-08-21 to 2026-08-22 — Local training-duration check

| Metric | Local 4 h | Local 16 h | Released ESMC-300M |
|---|---:|---:|---:|
| P-CORE-Q4 v0.3 | 25.4609 | 30.6609 | 37.4847 |
| Full contact P@L | 0.1013 | 0.1636 | 0.5387 |

**Worked**

- Extending the local ESMC-300M-class run from four to sixteen hours improved
  all four trusted P-CORE tasks.
- Full contact P@L improved by 0.0623; the fixed-checkpoint paired 95% interval
  was [0.0614, 0.0632].

**Did not work / decision**

- The runs used different seeds, so their paired evaluation intervals do not
  estimate training-seed uncertainty.
- The sixteen-hour checkpoint remained below released ESMC-300M on P-CORE-Q4,
  contact P@L, and three of four trusted task metrics. It was not promoted as a
  reproduction-quality model.

## 2026-08-27 — One-hour AutoResearch round

**Worked**

- Under the fixed four-L40S, one-hour contract, the retained training changes
  improved contact P@L from 0.0925 to 0.0960.
- The retained recipe combined learned residual/input routing, parameter-free
  transformer RMSNorm, depth-scaled residual initialization, and a final-20%
  learning-rate cooldown.

**Did not work / limitation**

- The +0.0036 improvement is a single-round, fixed-checkpoint result. It is not
  yet a multi-seed release claim.
- Rejected candidates remain development evidence; they are not presented as
  supported public configurations.

## 2026-08-28 — Fast contact evaluation promoted to main

**Worked**

- Replaced the default parallel contact path with the exact AutoResearch fast
  path: fit the frozen probe once, bind it to the checkpoint and contact
  manifest, reuse it across shards, and score only nonzero L1-probe channels.
- Added an optional receipt-bound static scoring cache with a full digest
  preflight. Omitting the cache retains probe reuse and sparse scoring.
- Added regression tests showing that cached top-L scoring matches the stable
  reference rule and sparse scoring matches the all-channel implementation.
- Made the exact one-hour incumbent an opt-in public preset while preserving
  the original ESMC-compatible recipe as the speedrun default.
- Separated schedule progress from the emergency optimizer-step stop so a
  nonbinding cap cannot silently move the cooldown.

**Did not carry forward**

- The AutoResearch `prefix_data` and `round2_data` helpers reconstruct only the
  old parent-plus-Q9 campaign prefix. They do not rebuild the current 665M
  release from raw sources, so presenting them as the public reconstruction
  path would be misleading. The current release-manifest decontamination
  validator remains supported; a future raw rebuild must be parent-independent
  and reproduce the complete protected evaluation union.

## Open evaluation work

1. Requalify the current four trusted tasks under the same gates used for new
   candidates.
2. Complete MegaScale confound, weak-model, and probe-stability controls.
3. Repair PRING with a preregistered uncertainty estimator and ten negative
   realizations.
4. Decide prospectively whether CAFA5 should be replaced or dropped without
   tuning a replacement against the observed alpha results.
5. Freeze aggregation only after task admission, then shadow the successor
   beside the current four-task reference before release.

## Entry format

Future entries should record the date, hypothesis, frozen comparison, raw
metric and uncertainty, what worked, what failed, and the keep/reject decision.
Do not add private paths, credentials, scheduler details, or results that
cannot be traced to a retained receipt.
