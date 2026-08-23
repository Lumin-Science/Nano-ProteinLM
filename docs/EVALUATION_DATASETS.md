# Exact evaluation datasets and split use

This document defines which data enter each metric. The authoritative
machine-readable receipt is `EVALUATION_SPLIT_LEDGER.json`, built from the exact
six probe payloads and contact payloads. Probe fitting, hyperparameter selection, and
final scoring are separate roles:

- **Probe fit:** fits a linear/ridge/contact probe. It never supplies a headline
  score.
- **Validation:** selects `C`, ridge `alpha`, or the contact-probe setting. It
  never supplies a headline score.
- **Test:** is touched only after selection and supplies the reported metric and
  bootstrap uncertainty.

## Frozen split ledger

| Evaluation | Probe-fit data | Validation data | Final test data | Reported test metric | Selection status |
|---|---:|---:|---:|---|---|
| Held-out MLM | Training corpus only | 4,096 representatives/source (12,288 total in the current contract) | None | Sequence-mean NLL and perplexity on validation; a guardrail, not a downstream test | Guardrail only |
| Remote homology | 12,312 proteins | 736 proteins | 718 fold-holdout proteins | Balanced accuracy; family-group bootstrap | P-CORE-Q4 |
| Secondary structure | 8,678 proteins | 2,170 proteins | CB513: 513 records, 434 unique sequences | Residue macro-F1; protein bootstrap; Q3 accuracy diagnostic | P-CORE-Q4 |
| Enzyme Commission | 15,551 proteins | 1,729 proteins | 720 proteins from the `<30%` identity test column | Macro average precision over eligible labels; Bayesian label/group bootstrap | **Quarantined** |
| DeepLoc2 | Three of five partitions per fold | The partition after the test partition | One of five partitions; every protein is test exactly once | Macro average precision pooled over all five held-out folds | P-CORE-Q4 |
| Human PPI | 35,669 pairs / 6,844 unique proteins | 315 pairs / 277 unique proteins | 237 pairs / 227 unique proteins | Average precision; connected-pair-component bootstrap | **Quarantined** |
| FLIP2 hydro low-to-high | 9,974 variants | 2,493 variants flagged as validation within the training set | 12,468 high-fitness test variants | Spearman correlation; variant-group bootstrap | P-CORE-Q4 |
| Long-range contact | 16 PDB chains | 4 PDB chains | 20,775 chains (20,758 unique sequences) | Mean precision at L; chain bootstrap | Separate promotion axis |

DeepLoc2 has no single permanent validation/test split. With test partition
`k`, validation is `(k + 1) mod 5`, and the other three partitions fit the
probe. The partition sizes are 5,963, 5,451, 5,731, 5,696, and 5,462 proteins,
for 28,303 total. Consequently every DeepLoc2 sequence belongs to both a
validation role and a test role across the five-fold evaluation.

The contact manifest is selected deterministically from the 2024-02-28 PDB
snapshot. The first 16 frozen chains fit the logistic attention probe, the next
four select its regularization, and the remaining 20,775 chains form the final
P@L evaluation. Test chains never select the probe.

## Frozen-embedding probe path

The encoder is never updated by these tasks. For a sequence, final-layer
residue vectors are deterministically windowed and mean-pooled over valid
residues to produce one protein vector. Remote homology uses a multinomial
logistic probe; EC and DeepLoc2 use one-vs-rest multilabel logistic probes;
FLIP2 uses ridge regression. Secondary structure skips pooling and applies the
linear classifier to each residue vector. Human PPI concatenates the symmetric
pair features `abs(z_a - z_b)` and `z_a * z_b` before logistic regression.

The feature standardizer is fit on probe-train data only. The probe is fit on
that same split, `C` or ridge `alpha` is selected on validation, and the selected
probe is applied once to test. It is not retrained on validation. Bootstrap
intervals resample frozen test predictions by biological group; they do not
refit the encoder or probe. Thus each number measures information linearly
accessible from a frozen embedding under one declared split and probe family,
not an end-to-end fine-tuned model.

## Decontamination scope

Two digest-bound FASTAs are emitted:

- `evaluation_validation_test.fasta`: 70,305 unique protected sequences and
  24,497,158 residues.
- `evaluation_all_splits.fasta`: 116,841 unique sequences and 35,467,998
  residues, including probe-fit data.

The first 300M campaign uses the stronger all-splits FASTA for homology
screening. Its production gate freezes a complete eligible prefix of 5,000,000
train candidates plus 32,768 validation candidates per source; the materialized
corpus needs 3,000,000 plus 4,096 per source. The source receipts bind each
candidate FASTA to its original-FASTA scan boundary, and preparation fails if it
would cross that screened boundary. This screens every sequence that can enter
the actual four-hour corpus while leaving a separate whole-reservoir audit off
the launch-critical path.

MMseqs2 17-b804f removes a candidate representative when it aligns at least
30% sequence identity with at least 80% coverage of both the evaluation query
and training target (`--cov-mode 0`). Exact normalized-sequence SHA-256 matches
are also removed independently. The final receipt reports excluded
representatives by source and proves that the validation/test exclusion set is
a subset of the all-splits exclusion set. A second post-preparation verifier
scans the actual mmap indices and requires zero exact/homology intersections and
zero train/validation overlap for every source.

The corpus-derived MLM validation partition is created only after external
evaluation exclusions. It is SHA-partitioned from the remaining source
representatives and is disjoint from training by exact digest. It currently has
no separate test partition, because it is an optimization guardrail rather than
the evidence for representation quality.
