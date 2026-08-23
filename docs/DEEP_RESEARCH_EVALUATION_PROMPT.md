# Deep-research prompt: evaluating frozen protein embeddings

Copy the prompt below into ChatGPT Deep Research.

---

Act as a senior protein-machine-learning evaluation researcher. Design a
scientifically defensible, public, and computationally practical benchmark
suite for selecting the best **frozen protein sequence embedding model**. Work
as of 22 August 2026 and use primary sources wherever possible: original
papers, official benchmark/dataset pages, and official code repositories.
Provide inline citations and a linked bibliography. Distinguish sourced facts,
your inferences, and recommendations.

## Project context

We train ESMC-style sequence encoders at approximately 300M and 600M
parameters. The encoder is frozen for downstream evaluation. Our current probe
path takes final-layer residue vectors, mean-pools valid residues for protein
tasks, standardizes features using probe-train data only, selects logistic
`C` or ridge `alpha` on validation, applies the selected already-fitted probe
once to test, and bootstraps fixed test predictions by biological group.
Secondary structure probes residue vectors directly. Long-range contact P@L is
a separate attention-based structural axis.

The current trusted selection core contains:

- remote homology: fold-holdout balanced accuracy;
- secondary structure: residue macro-F1 on CB513;
- DeepLoc2: five-fold multilabel macro average precision;
- FLIP2 hydro low-to-high: Spearman correlation.

Two tasks still run and report raw metrics but are quarantined from selection:

- Enzyme Commission: released ESMC-300M/600M reach about 0.71 macro-AP, while
  released ESMC-6B collapses to about 0.024 despite an integrity audit;
- Human PPI: only 237 reconstructed test pairs, non-monotonic released scale
  ordering (600M < 300M < 6B), a narrow released range of 0.0319 AP, and a
  severely undertrained local checkpoint only 0.0176 AP below released 300M.

Our four trusted null-normalized tasks form a versioned P-CORE-Q4 geometric
mean. It is scale-ordered on released ESMC-300M/600M/6B
(37.4847/38.1553/40.6011). We will not silently add a new task to this score:
every replacement must first pass a benchmark-qualification study.

Pretraining sources are UniRef90, MGnify, and OMG/IMG. Evaluation
decontamination screens every probe-fit, validation, and test sequence against
the actual training candidate pool using exact sequence hashes plus MMseqs2 at
at least 30% identity and at least 80% bidirectional coverage. Flag benchmarks
whose contamination cannot be controlled under this design.

## Research questions

1. What distinct properties of a protein embedding should be measured beyond
   the current suite? Separate at least: evolutionary/fold information,
   residue-level structure, global structure, molecular function, localization,
   protein-protein or protein-ligand interaction, mutational fitness/stability,
   disorder and binding regions, metric geometry/retrieval, few-shot label
   efficiency, and robustness to length/species/family distribution shift.
2. Which current public benchmarks can measure each property using frozen
   embeddings with a linear, ridge, k-nearest-neighbor, retrieval, or similarly
   low-capacity readout? Include promising recent benchmarks and relevant
   standards such as TAPE, PEER, ProteinGym, FLIP/FLIP2, CAFA/GO/EC resources,
   CATH/SCOPe-style fold splits, localization resources, interaction resources,
   and disorder/binding datasets, but do not assume any is suitable.
3. Which benchmarks are saturated, leaky, too small, label-noisy, dominated by
   taxonomy, unavailable for redistribution, or so probe-sensitive that they
   should remain diagnostic rather than select a model?
4. What should replace the current Human-PPI task? Prioritize family-disjoint
   protein splits, realistic hard negatives, substantially larger test sets,
   connected-component-safe splitting, leakage controls, and evidence that the
   task separates random, undertrained, and strong pretrained encoders.
5. What should replace or repair the EC task? Consider label ontology/version,
   train/test identity thresholds, unseen labels, macro-AP instability,
   multilabel eligibility policy, label frequency, and alternatives based on
   GO/CAFA or function retrieval.
6. Which evaluation methods test embedding geometry without fitting a large
   supervised probe—for example family/fold retrieval, nearest-neighbor transfer,
   clustering, alignment-free similarity, or few-shot curves—and how should
   they avoid rewarding trivial length or taxonomy signals?

## Required qualification standard

For every candidate task, propose a preregistered qualification experiment
using at least: a random/untrained encoder, a deliberately undertrained
checkpoint, multiple checkpoints along training, released ESMC-300M/600M/6B,
and at least two independent strong external protein language models where
licensing permits. A task is eligible for model selection only if it has:

- a biologically defensible, leakage-resistant split and explicit
  pretraining-decontamination route;
- adequate test size/effective biological groups and uncertainty;
- expected discrimination between negative controls and strong models;
- stable qualitative ranking across probe seeds and reasonable regularization;
- a target-permutation control near its declared null;
- no material dependence on sequence length, taxonomy, duplicate families, or
  easy negative construction;
- public data/code or a reproducible acquisition path with usable terms;
- affordable runtime for release evaluation, with a validated cheaper proxy if
  proposed for routine checkpoints.

Do not require monotonic parameter-scale ordering as an axiom—larger models can
legitimately regress—but treat unexplained non-monotonicity as a signal that
must be separated into model behavior versus benchmark/probe failure. Specify
the experiment that makes that distinction. Do not create one aggregate score
until task validity and rank behavior have been established.

## Required deliverables

1. An executive recommendation naming a **minimum selection suite** of roughly
   6–10 tasks, a broader release-only suite, and raw diagnostic tasks.
2. A landscape table of at least 20 candidate evaluations with columns for:
   biological capability, dataset/version and primary citation, unit and size,
   split logic, train/validation/test role, readout and pooling, primary metric,
   null baseline, uncertainty unit, contamination risk, known saturation or
   leakage issue, license/access, estimated embedding/probe cost, and your
   selection/diagnostic/reject verdict.
3. A fully specified replacement plan for Human PPI and for EC, including the
   exact negative construction or label policy, family/identity split,
   bootstrap unit, minimum effective test size, and reinstatement criteria.
4. For every recommended task, a protocol card defining input embeddings,
   pooling/pair features, permitted probe class, regularization selection,
   metric, split use, null, bootstrap, failure/NaN policy, and what scientific
   claim the score does and does not support.
5. A benchmark-qualification matrix covering negative controls, scale models,
   seed/probe sensitivity, power, confound tests, and pass/fail thresholds.
6. Two compute plans: a routine checkpoint gate and an exact release gate.
   Estimate GPU embedding time/storage plus CPU probe time and identify tasks
   that can share a content-addressed embedding cache. State assumptions.
7. A versioned migration plan from P-CORE-Q4: which tasks stay, which new tasks
   remain provisional, how much evidence is required before inclusion, how old
   scores remain comparable, and how to avoid benchmark overfitting.
8. A prioritized implementation backlog with the first three pilot studies
   most likely to falsify weak benchmark choices quickly.

Be critical. Report contradictory evidence and negative findings. Prefer a
smaller set of high-validity, orthogonal evaluations over a large collection of
correlated leaderboards. End with a concise list of unresolved questions that
require dataset-owner clarification or a local pilot rather than literature
research.

---
