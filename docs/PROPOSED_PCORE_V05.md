# Proposed P-CORE v0.5-q9

Status: **proposed evaluation core; not yet internally qualified**  
Purpose: select frozen protein-sequence embedding models as reliably as
possible across complementary biological capabilities.  
Machine-readable contract: [`configs/pcore_v05_proposed_q9.yaml`](../configs/pcore_v05_proposed_q9.yaml)

## Recommendation

Adopt a nine-task proposed core, versioned as `pcore-v0.5-proposed-q9`. Here,
`v0.5` is the protocol generation and `q9` is the number of proposed core tasks.
This avoids calling a nine-task suite “Q5,” which would conflict with the
existing meaning of Q4 as a four-task core.

The proposal retains the four currently trusted biological axes and adds five
better-targeted axes or replacements. EC and the current 237-pair Human PPI
remain raw quarantined diagnostics. Long-range contact P@L remains an
independent structural promotion axis and is not counted twice in the core.

## Proposed core task vector

| # | Axis | Frozen task | Input/readout | Primary metric | Biological uncertainty unit |
|---:|---|---|---|---|---|
| 1 | Evolution | TAPE remote homology, fold holdout | Mean embedding; standardized multinomial logistic | Balanced accuracy | SCOP family/superfamily group |
| 2 | Local structure | TAPE/NetSurfP2 secondary structure; CB513 primary | Residue embedding; standardized multinomial logistic | Residue macro-F1; Q3 accuracy secondary | Protein |
| 3 | Global structure | CATH 4.4 experimental-domain remote retrieval | Mean + L2 normalization; cosine retrieval, no fitted probe | H-level mAP | CATH H-superfamily |
| 4 | Localization | Official DeepLoc2 five homology partitions | Mean embedding; one-vs-rest logistic | Macro-AUPRC | Protein/homology cluster |
| 5 | Molecular function | CAFA5 MF no-knowledge, at most 30% identity hard set | Mean embedding; frozen cosine-kNN transfer; sparse linear secondary | Weighted Fmax; protein AP secondary | Protein, then species |
| 6 | Interaction | PRING Human node-disjoint plus 30%-cluster-disjoint test | Symmetric `[abs(a-b); a*b]`; logistic | AUPRC; partner Recall@50 secondary | Held-out protein/community |
| 7 | Fitness shift | FLIP2 predefined engineering-shift panel | Mean mutant embedding; ridge | Hierarchical split/dataset aggregate of Spearman and NDCG | Dataset, landscape, then group |
| 8 | Stability transfer | MegaScale parent- and 30%-cluster-held-out ΔΔG | `mean(mutant)-mean(WT)`; ridge | Median per-parent Spearman | Parent/family |
| 9 | Disorder | CAID3 Disorder-PDB temporal test | Residue embedding; binary logistic | Average precision; MCC secondary | Protein |

All task identifiers, dataset snapshots, row IDs, labels, splits, biological
groups, negative seeds, and source hashes must be frozen before a result is
called `pcore-v0.5-proposed-q9`.

## Shared frozen-embedding contract

- Use the final encoder layer before the language-model head.
- Remove BOS, EOS, and padding positions.
- Use one deterministic long-sequence windowing policy for every model.
- Use residue means for global protein tasks; do not search layers or pooling.
- Permit only linear/logistic, ridge, or a preregistered cosine/kNN readout.
- Fit normalization and probes on probe-training data only.
- Select regularization only on the fixed validation split.
- Apply the selected already-fitted probe once to test; do not refit on test or
  choose features from test performance.
- Bootstrap frozen test predictions; do not refit the encoder inside bootstrap
  replicates.
- Embed every unique protein once and reuse content-addressed embeddings across
  tasks and protein pairs.

## Contamination and population contract

Every model is evaluated on the same fixed rows. Never remove a different test
subset for each model and compare the resulting scores directly.

For every test sequence, report its maximum exposure to the relevant model's
pretraining corpus in four strata:

1. exact sequence match;
2. at least 50% identity;
3. at least 30% identity;
4. post-model-cutoff sequence or annotation.

Also report an all-model common-unexposed intersection when it retains adequate
biological groups. This is a secondary clean ranking, not a replacement for the
fixed primary population. Future local training intended to make a clean v0.5
claim should homology-screen the union of every probe-fit, validation, and test
sequence before materializing its training corpus. Existing checkpoints must
be labeled exposure-audited rather than retroactively called decontaminated.

## Task normalization and summary

Each task first emits its complete native metric and interval. It then emits a
null-normalized gain `G_t`:

- bounded higher-is-better metric: `(metric - null) / (1 - null)`;
- Spearman correlation: `rho`, with null `0`;
- lower-is-better error, if ever used: `1 - error / null_error`.

Do not count related subtasks as separate votes. Q3/Q8, CATH hierarchy levels,
FLIP2 splits, and multiple negative seeds are aggregated inside their parent
task first. Each of the nine biological axes receives one vote.

The proposed headline is a robust three-part result rather than one opaque
number:

1. `median_gain = 100 * median(G_t)` — typical cross-task quality;
2. `lower_quartile_gain = 100 * Q25(G_t)` — weak-region quality;
3. the nine-task native metric/gain vector with paired intervals.

The historical geometric mean may be reported for continuity but is not the
primary v0.5 proposal. A task at or below its null is an explicit guardrail
failure and cannot be hidden by the median.

## Proposed model-comparison rule

For equal-compute candidate `B` against reference `A`, use identical task rows,
probe seeds, negative seeds, and hierarchical bootstrap resamples. Promote `B`
only if all conditions hold:

1. all nine task receipts are complete and pass their null guardrails;
2. `median_gain(B) - median_gain(A) >= 1.0` percentage point;
3. the paired 95% interval for the median-gain difference has lower bound above
   zero;
4. `lower_quartile_gain(B) >= lower_quartile_gain(A)`;
5. no task has a gain regression worse than five percentage points whose paired
   95% interval is wholly below zero;
6. contact P@L, reported separately, has no material statistically supported
   regression;
7. held-out MLM, throughput, memory, and inference cost are reported as
   guardrail/cost axes, not mixed into biological representation quality.

If no model passes, retain the reference. Report Pareto relationships even when
the scalar rule selects a winner.

## Required reliability controls

Every task report should include:

- target-permutation or label-permutation control at the correct biological
  level;
- random encoder and AA-composition/k-mer baselines;
- deliberately undertrained checkpoint;
- released ESMC-300M/600M/6B and at least two external pLM baselines when
  available;
- task-specific confound baselines: degree and identity for PPI; mutation count
  and substitution identity for fitness/stability; taxonomy and length for
  localization/disorder; sequence-similarity retrieval for structure/function;
- at least 10,000 paired hierarchical bootstrap replicates for exact reports;
- probe/negative-seed sensitivity and effective biological-group count.

Strict parameter-scale monotonicity is not required. A task remains suspicious
when it cannot distinguish random or severely undertrained encoders, collapses
under its hard/confound-matched stratum, or changes model ordering under allowed
probe or negative-sampling choices.

## Other reported evaluations

- **Separate promotion axis:** full long-range contact P@L.
- **Quarantined raw diagnostics:** current EC and current PEER Human PPI.
- **Release-only candidates:** PRING cross-species, ProteinGym few-shot,
  CAID3 Binding-IDR, AsEP epitope, BioLiP2 temporal binding sites, SCOPe
  retrieval, and DeepLoc2.1 membrane type.
- **Not part of this protein-only core:** protein–ligand affinity tasks that
  require a ligand encoder or learned fusion model.

## Interpretation boundary

This proposal is ready to guide implementation and evaluation design today. It
is not a claim that every proposed dataset has already passed local leakage,
power, licensing, or weak–strong discrimination tests. Until those checks are
run, publish the protocol as **proposed** and dual-report the existing Q4 score.

