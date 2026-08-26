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

## Dataset and publication provenance

Every evaluation payload is an adaptation of a named public source. The source
publication establishes where the sequences, structures, or labels originated;
the frozen split ledger below establishes exactly how this repository uses
them. Source-paper metrics are not silently imported: all reported values are
recomputed with the probes and metrics declared here.

| Evaluation piece | Exact lineage used here | Original source publication(s) |
|---|---|---|
| Held-out MLM | A SHA-partitioned validation sample created after evaluation exclusion from the same UniRef90, MGnify, and OMG/IMG representative reservoirs used for training. It is an internal guardrail, not a published downstream benchmark. | Suzek et al., [*UniRef clusters*](https://doi.org/10.1093/bioinformatics/btu739) (2015); Richardson et al., [*MGnify in 2023*](https://doi.org/10.1093/nar/gkac1080) (2023); Cornman et al., [*The OMG dataset*](https://doi.org/10.1101/2024.08.14.607850) (2024). |
| Remote homology | The TAPE-distributed fold-classification payload: SCOP 1.75 fold labels from the DeepSF construction, using the 12,312/736/718 train/validation/fold-holdout split. | Hou, Adhikari, and Cheng, [*DeepSF*](https://doi.org/10.1093/bioinformatics/btx780) (2018); Rao et al., [*Evaluating Protein Transfer Learning with TAPE*](https://proceedings.neurips.cc/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html) (2019). |
| Secondary structure | The TAPE/NetSurfP-2.0 train and validation payloads with CB513 as the primary test set. The current ledger contains 513 records representing 434 unique normalized sequences. | Klausen et al., [*NetSurfP-2.0*](https://doi.org/10.1002/prot.25674) (2019); Cuff and Barton, [the original CB513 publication](https://pubmed.ncbi.nlm.nih.gov/10081963/) (1999); Rao et al., [TAPE](https://proceedings.neurips.cc/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html) (2019). |
| Enzyme Commission | A sequence-only adaptation of the TorchDrug/TorchProtein `EnzymeCommission` artifact, using its `<30%` identity test column. The task descends from DeepFRI's PDB-chain EC benchmark; obsolete/missing chains and this repository's normalization explain why local counts must come from the ledger, not a paper table. | Gligorijević et al., [*Structure-based protein function prediction using graph convolutional networks*](https://doi.org/10.1038/s41467-021-23303-9) (2021); Zhang and Xu, [TorchProtein dataset record](https://doi.org/10.5281/zenodo.6622158) (2022). |
| DeepLoc2 | The official `multisub_5_partitions_unique.csv` with all five homology-aware partitions. Each partition rotates once through test and once through validation. | Thumuluri et al., [*DeepLoc 2.0*](https://doi.org/10.1093/nar/gkac278) (2022). |
| Human PPI | The PEER release of Pan's human interaction set: HPRD-derived positive pairs and negatives formed from proteins assigned to different subcellular locations, followed by PEER's sequence-redundancy filtering and split. | Pan, Zhang, and Shen, [*Large-scale prediction of human protein-protein interactions*](https://doi.org/10.1021/pr100618t) (2010); Xu et al., [*PEER*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e467582d42d9c13fa9603df16f31de6d-Abstract-Datasets_and_Benchmarks.html) (2022). |
| FLIP2 Hydro low-to-high | The official Hydrophobic Core `low_to_high` fitness split. It pools variants of three wild types and trains below the landscape-wide median before testing above it. | Didi et al., [*FLIP2*](https://doi.org/10.64898/2026.02.23.707496) (2026). |
| Long-range contact | Experimentally determined structures from a frozen 2024-02-28 RCSB PDB snapshot, converted into the ESM attention-to-contact protocol and the ESMC long-range P@L definition. The 16/4/20,775 split is this repository's deterministic reconstruction. | Berman et al., [*The Protein Data Bank*](https://doi.org/10.1093/nar/28.1.235) (2000); Rao et al., [*Transformer protein language models are unsupervised structure learners*](https://openreview.net/forum?id=fylclEqgvgd) (2021); Candido et al., [*Language Modeling Materializes a World Model of Protein Biology*](https://doi.org/10.64898/2026.06.03.729735) (2026). |

The local evaluation deliberately changes some readouts from their source
papers so that every released encoder is compared under one frozen,
low-capacity protocol. Remote homology uses balanced accuracy; secondary
structure uses residue macro-F1 with Q3 accuracy retained only as a diagnostic;
EC and DeepLoc2 use macro average precision; Human PPI uses average precision;
and FLIP2 uses Spearman correlation. The encoder is frozen in every case.
Therefore, compare model rows within this repository's tables, not directly to
headline results in the source publications.

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
