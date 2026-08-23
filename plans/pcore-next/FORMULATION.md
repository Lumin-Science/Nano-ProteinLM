# P-CORE Next: qualification plan for better protein-embedding evaluation

Status: **Proposed core delivered**. The scientist clarified that the current
goal is a high-reliability proposed evaluation core, not completed benchmark
qualification. The concrete proposal is
[`docs/PROPOSED_PCORE_V05.md`](../../docs/PROPOSED_PCORE_V05.md); the research
and pilot program below is optional future validation and does not block use of
the proposal as a design specification.

The submitted review, [`pasted-text.txt`](PROJECT_BRIEF.md), motivates the
candidate portfolio. Its factual claims, dataset versions, licenses, counts,
and proposed protocols remain research obligations until checked against
primary sources and local artifacts.

## Planning page 1 — What scientific question are we actually pursuing?

### Scientific Question contract

| Field | Specification | Status |
|---|---|---|
| Higher-level scientific question | Which frozen protein-sequence representation contains broadly reusable biological information that transfers under genuinely difficult evolutionary, structural, functional, interaction, fitness, stability, localization, and disorder shifts? | **confirmed** |
| Concrete input | Final-layer residue embeddings from a frozen sequence encoder, with one common tokenization/windowing contract and sequence or symmetric pair features derived without encoder updates. | **confirmed** |
| Anticipated output/interface | A versioned proposed evaluator that emits a raw task vector, biological-group uncertainty, trust/qualification status, exposure strata, and a conservative selection summary while preserving task receipts and hashes. | **confirmed** |
| Combined boundaries and nearest exclusions | Sequence-only encoder evaluation; fixed linear/logistic/ridge or preregistered retrieval readouts; no fine-tuning, MLP, learned pooling, per-model layer search, ligand/fusion model, model-specific test deletion, or silent benchmark mutation. Contact P@L stays separate. New tasks have zero selection weight until qualified. | **confirmed** |

No project-specific scientific choice remains unresolved on this page. The
output schema is provisional because the aggregation and final number of
trusted tasks must be chosen using qualification evidence, not preference.

## Planning page 2 — What evidence would make that scientific goal credible?

### Evaluation contract

| Field | Specification | Status |
|---|---|---|
| General evaluation goal | Demonstrate that the suite separates weak from mature representations for biological reasons, remains stable to allowed probe choices, exposes rather than hides contamination and confounds, and produces reproducible model-selection decisions with biological-group uncertainty. | **confirmed** |
| Metrics | Keep the four Q4 tasks as anchors. Treat CATH retrieval, CAFA5-MF hard function, PRING Human PPI, MegaScale family-held-out stability, and CAID3 disorder as provisional candidates. Qualify each with controls, target permutations, exposure/confound strata, effective-group power, probe stability, orthogonality, paired uncertainty, and explicit admission rules. Compare aggregation rules only after task qualification. | **research-needed** |
| Training data | Existing models use UniRef90, MGnify, and OMG/IMG mixtures with the current Q4 exclusion union. Any definitive future P-CORE-Next training run must protect the frozen Q4 plus release-candidate evaluation union before corpus materialization. Existing checkpoints cannot retroactively be called clean for new tasks; they receive exposure-stratified reporting. | **confirmed**, with the new protected union **research-needed** |
| Test data | Existing Q4 manifests remain frozen. Every proposed dataset/version, label snapshot, biological unit, split, license, and effective test-group count must be reconstructed from primary sources. A single common test population is used across models; exact/50%/30% identity and temporal-exposure strata are annotations, not model-specific filters. | **research-needed** |
| Baselines | Required families are AA composition/k-mer, sequence-similarity retrieval where appropriate, random encoders, a controlled undertraining trajectory, released ESMC-300M/600M/6B, ESM-2-650M, ProtT5-XL-U50, and task-specific confound baselines such as PPI degree-only or mutation-count-only fitness. Exact artifacts and feasible model panel require verification. | **proposed** and **research-needed** |

The scientist already supplied candidate test data and baselines through the
review. They are captured above as explicit research obligations rather than
silently treated as verified or accepted.

## The central correction to the submitted Q5 proposal

Do not build nine evaluators and then ask whether the resulting leaderboard
looks sensible. First build a **benchmark qualification harness** whose object
of study is each benchmark. The harness decides whether a candidate is:

1. `selection`: allowed to influence promotion;
2. `release_only`: scientifically informative but too correlated, saturated,
   narrow, or expensive for selection;
3. `diagnostic`: useful for debugging or capability discovery only;
4. `quarantined`: executable and reportable but presently invalid for claims;
5. `rejected`: outside the sequence-embedding objective or irreparably weak.

Use `P-CORE Next` or `pcore-v0.5-alpha` during development. Because Q4 currently
means four trusted tasks, the final protocol suffix must report how many tasks
actually pass: for example, `pcore-v0.5-q7` or `pcore-v0.5-q9`. “Q5” must not
simultaneously mean version five and a nine-task core.

## Concrete execution plan

### Phase 0 — Freeze the incumbent and the decision rules (2 working days)

Deliverables:

- immutable Q4 dataset, split, probe, solver, seed, bootstrap-group, and model
  artifact receipts;
- a reconciliation note for the local 237-pair Human-PPI artifact versus the
  227-test-pair count quoted in the submitted review;
- a `benchmark_registry.schema.json` defining dataset version, source hashes,
  sequences/accessions, labels, split roles, biological groups, exposure
  strata, readout, null, metric, and trust state;
- a written rule that Q4 remains authoritative through alpha, beta, and RC
  qualification; contact remains separate.

Gate P0: every existing result can be reaggregated from frozen receipts and no
task name stands in for a version or split hash.

### Phase 1 — Primary-source and artifact feasibility (3–5 working days)

For CATH, CAFA5-MF, PRING, MegaScale, and CAID3, produce one dataset card and a
metadata-only build receipt before embedding anything. Verify:

- primary paper, official download, version/snapshot date, upstream license,
  redistribution route, and required attribution;
- actual sequence/label counts after the proposed filters;
- exact duplicates and train/validation/test cluster intersections;
- effective biological groups, label prevalence, length/taxonomy/species
  composition, and missing-label policy;
- whether the proposed split can be reproduced deterministically and whether
  all test labels were unavailable at the claimed temporal cutoff;
- residue totals and pooled/token storage estimates.

Fail fast when an official artifact cannot be pinned, redistribution is not
legal, the hard test population is underpowered, or the split cannot implement
the claimed biological holdout. No model is embedded in Phase 1.

Gate P1: each candidate is either `feasible-alpha` with a hashed manifest and
known open questions, or rejected/deferred with a recorded reason.

### Phase 2 — Build one common qualification panel (3 working days plus embedding)

Freeze model artifacts before inspecting candidate-task rankings:

- AA composition and k-mer controls;
- three random seeds of one architecture-compatible encoder;
- a small controlled trajectory at 0%, 1%, 10%, 30%, and 100% of a fixed token
  budget, preferably a 55–100M ESMC-compatible model with at least two seeds;
- the completed local four-hour 300M checkpoint as a deliberately weak system;
- released ESMC-300M, ESMC-600M, and ESMC-6B;
- ESM-2-650M and ProtT5-XL-U50;
- task-specific classical/confound baselines.

If suitable local intermediate checkpoints already exist, use them only after
hashing their data and training contracts. Do not start or alter a training job
merely to fill this panel. All Python dependencies are added through
`pyproject.toml`, locked with `uv`, and executed with `uv sync --frozen`.

The embedding layer is content-addressed by model hash, tokenizer/config hash,
sequence hash, pooling/windowing contract, precision, and hidden dimension.
Embed each unique sequence once and reuse it across probes.

Gate P2: every model/control has a complete artifact receipt and every task can
address a common immutable embedding index.

### Phase 3 — Run the three highest-information falsification pilots

#### Pilot A: PRING leakage and negative sensitivity (first)

Run the original node-disjoint split, the proposed 30%-cluster-disjoint split,
and degree-matched negative variants. Embed proteins once; stream pair features.
Compare prevalence, degree-only, sequence-identity, random, undertrained, and
mature-model baselines over ten frozen negative seeds.

Immediate stop conditions:

- degree-only or sequence-identity baselines explain most of the pLM gain;
- model ordering is unstable across negative seeds or reasonable regularization;
- fewer than 500 held-out proteins or 50 independent bootstrap blocks survive;
- the mature-versus-undertrained paired interval loses significance after
  degree/identity matching.

If it fails, retain interaction as a release diagnostic and investigate graph
retrieval; do not search for a friendlier negative sampler.

#### Pilot B: CAFA hard-set existence and identity dependence (second)

First build the temporal Molecular-Function/no-knowledge/at-most-30%-identity
target set without fitting a probe. Verify the annotation timeline, ontology
snapshot, target count, species mix, label depth/frequency, and known-annotation
exclusions. Only then compare fixed cosine/kNN transfer, sparse linear readout,
sequence similarity, term frequency, random, undertrained, and mature models.

Immediate stop conditions:

- fewer than 500 effective test proteins remain;
- the claimed temporal target is annotation-new but not biologically hard;
- almost all representation gain is confined to the 20–30% identity stratum;
- hierarchy/unknown-label policies change model ordering.

#### Pilot C: MegaScale parent leakage (third)

Select roughly 100 family-disjoint parents and 50k–100k variants. Compare
random-variant, parent-disjoint, and parent-plus-30%-cluster-disjoint splits.
Use mutant-minus-WT embeddings, parent-median Spearman, pooled metrics only as
secondary output, and parent/family bootstrap.

Immediate stop conditions:

- performance collapses to null under parent holdout;
- a few parents dominate the aggregate or interval;
- mutation-count or AA-substitution controls match mature representations;
- held-out parents are too few or too homogeneous for a stable comparison.

Gate P3: only pilots with a surviving causal interpretation proceed. Failed
tasks receive zero selection weight; there is no replacement search inside the
same test results.

### Phase 4 — Add orthogonal structure and disorder candidates (1 week)

Construct CATH experimental-PDB-domain retrieval and CAID3 Disorder-PDB only
after the first three pilots. For CATH, test whether low-capacity cosine
retrieval adds information beyond remote homology. For CAID3, use residue-level
linear readout, protein bootstrap, and temporal/family-aware training data.

Gate P4:

- a candidate must pass the universal qualification rules below;
- if its cross-model score correlation with an existing selection task is
  above 0.90 and it does not change any controlled weak-versus-strong decision,
  it becomes release-only rather than receiving duplicate selection weight;
- a residue task must show effective protein-level, not residue-count-level,
  statistical power.

### Phase 5 — Full benchmark qualification (1–2 weeks)

Run each surviving candidate and each Q4 anchor through the same qualification
matrix. Freeze resample IDs so every model comparison is paired.

#### Universal admission rules

All rules are provisional thresholds to preregister before task results are
opened. Adjusting them after seeing a favored model requires a new benchmark
major version.

1. **Split integrity:** zero exact or prohibited-cluster overlap across probe
   roles; labels and biological groups are bound to immutable row IDs.
2. **Effective power:** at least 200 independent protein/family groups by
   default; at least 500 targets for CAFA; at least 500 held-out proteins and 50
   blocks for PPI; at least 50 held-out parents/families for MegaScale.
3. **Target control:** across at least 20 biologically valid permutations, the
   median normalized gain must stay within 0.02 of null and its 95th percentile
   below 0.05.
4. **Weak–strong separation:** at least three mature encoders must beat every
   random encoder, and the paired 95% biological-group interval for the median
   mature-minus-undertrained difference must have lower bound above zero. The
   median normalized effect must also exceed the larger of 0.05 or twice its
   bootstrap half-width.
5. **Training validity:** across controlled token-budget checkpoints, the
   median task-versus-training-progress Spearman over seeds should be at least
   0.5, with the final checkpoint better than both random and 1%-trained. Strict
   300M/600M/6B monotonicity is not required.
6. **Probe stability:** model-pair decisions may reverse in at most 10% of
   allowed seed/regularization perturbations; test data never choose layer,
   pooling, feature family, regularization, k, or threshold.
7. **Confound survival:** the mature-model advantage must remain positive in
   the hardest preregistered identity/taxonomy/length/degree/mutation/label-
   frequency stratum and retain at least half of the unmatched gain. Otherwise
   quarantine pending explanation.
8. **Common population:** all models are scored on the same rows. Exposure is
   reported as exact, at least 50%, at least 30%, and post-cutoff strata; no
   model receives a custom-cleaned test set.
9. **Orthogonality:** a highly correlated task needs independent evidence that
   it changes a scientifically meaningful selection decision; otherwise it is
   release-only.
10. **Cost and reproducibility:** a clean `uv sync --frozen` run reconstructs
    the probe result from hashes; exact residue count, storage, wall time,
    throughput, memory, and failure policy are recorded.

Gate P5: a signed qualification matrix assigns one trust state to every task.
Passing validity is necessary but not sufficient for selection; redundancy and
cost may still make a valid task release-only.

### Phase 6 — Choose aggregation from shadow decisions, not aesthetics (1 week)

Do not automatically replace the current geometric mean with the review's
median/lower-quartile proposal. On Q4 anchors plus qualified candidates, compare:

- current null-normalized geometric mean with task-at-null guardrails;
- median normalized gain plus lower quartile;
- Pareto eligibility followed by median gain, with lower quartile as a
  worst-region tie-break;
- raw-vector paired superiority with no scalar primary score.

Evaluate each rule on random, undertrained, trajectory, released, and external
models using paired bootstraps. Score the rule itself for rank stability,
sensitivity to one noisy/saturated task, consistency across model families,
and frequency of selecting a Pareto-dominated model. The preferred default is
the simplest rule that never hides a failed task and does not flip decisions
under plausible task removal or bootstrap resampling.

Gate P6: freeze one rule before using a new local training result for promotion.
Publish the raw vector, median, lower quartile, and contact axis regardless of
which scalar—if any—is primary.

### Phase 7 — Shadow deployment and major-version release (minimum 3 decisions)

For at least three real candidate comparisons, dual-report frozen Q4 and the
new RC. Q4 remains authoritative. Record every agreement/disagreement and
whether the disagreement traces to desired new biology, noise, confounding, or
aggregation. Do not tune the RC to agree with a favored candidate.

Before the next training run intended to support a P-CORE-Next clean claim,
freeze and homology-screen the union of every RC probe-fit, validation, and test
sequence. Training already running or completed before this union is frozen
remains a Q4-era/engineering baseline and is not relabeled.

Gate P7: release `pcore-v0.5-qN` only after three shadow decisions, a clean
locked rebuild, source/license approval, model-independent manifests, and an
independent review of one full result. Any later change to rows, splits,
ontology snapshot, negative construction, readout semantics, or weighting
requires a new major benchmark version.

## Deliverable topology

The eventual implementation should keep code, data, and evidence separable:

```text
benchmark_registry/       # one versioned YAML task card per benchmark
manifests/                 # accession/row/split/group/exposure hashes
evaluators/                # fixed readouts and metrics
qualification/            # controls, permutations, confounds, paired reports
protocols/                 # aggregation and trust contracts
results/                   # immutable model x task receipts
```

GitHub should host code, schemas, small manifests, tests, and result receipts.
Hugging Face should host redistributable frozen datasets, larger manifests,
embedding indexes when permitted, and model cards. Restricted upstream data
should be represented by acquisition scripts, accessions, and hashes—not copied
into a public repository.

## Schedule and decision ownership

| Milestone | Earliest elapsed time | Decision produced |
|---|---:|---|
| P0 Q4 freeze | Day 2 | Incumbent is reproducible and historically interpretable. |
| P1 feasibility | End of week 1 | Each proposed task is feasible, deferred, or rejected without model results. |
| P2 panel/cache | Early week 2 | Common controls and embeddings are ready. |
| P3 three pilots | End of week 2 or 3 | PRING, CAFA, and MegaScale survive or fail their strongest falsifier. |
| P4 CATH/CAID | Week 3 or 4 | Orthogonal structure/disorder evidence is available. |
| P5 qualification | Week 4–6 | Trust states and failure reasons are frozen. |
| P6 aggregation | Week 6 | RC rule and task count are preregistered. |
| P7 shadow/release | After at least three candidate decisions | P-CORE v0.5-qN may replace Q4. |

Codex can implement schemas and pilots after explicit research launch. The
scientist owns task admission, aggregation choice, and final promotion policy.

## Evidence that would change this plan

- Primary sources show a proposed dataset, split, count, or license differs
  materially from the submitted review.
- A cheaper candidate provides the same biological axis with better effective
  groups, temporal validity, or redistribution terms.
- The qualification model panel is infeasible because released artifacts or
  compute are unavailable; the panel must then be reduced before results are
  viewed, with lower claim strength.
- The existing Q4 anchors fail the same universal qualification rules; they
  should be reclassified rather than grandfathered indefinitely.
- Multiple candidate tasks are strongly correlated, in which case the target
  suite should contain fewer than nine selection tasks.

## What later research must resolve versus what a pilot must test

Research must resolve primary-source provenance, licenses, official artifact
versions, exact task definitions, temporal semantics, known leakage findings,
and feasible classical/model baselines. Local pilots must test actual filtered
counts, split integrity, effective groups, computational cost, permutation
behavior, confound sensitivity, weak–strong separation, probe stability,
orthogonality, and aggregation decision stability.
