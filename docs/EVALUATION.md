# Evaluation contract

Training checkpoints are evaluated on three levels:

1. Held-out sequence-mean MLM negative log-likelihood on hash-disjoint cluster
   representatives.
2. Frozen representation probes summarized by P-CORE.
3. Long-range contact P@L using all-layer/all-head symmetrized attention maps,
   a logistic probe fit on 16 structures and selected on four structures, Cβ
   distance below 8 Å (Cα for glycine), sequence separation at least 24, and
   top-L precision.

All structures and structural contact labels for the P@L evaluation come from
the frozen 2024-02-28 RCSB Protein Data Bank snapshot. This is also the PDB
population protected during training-corpus decontamination.

Following the ESMC paper, full long-range contact precision at L (P@L) over the
frozen 20,775-chain population is the primary AutoResearch validation and model
selection axis. P-CORE and held-out MLM are reportable reference panels; they
do not override a contact P@L regression.

## Released ESMC checkpoint P@L

| Released model | ESMC paper P@L-LR (95% CI) | Our full 20,775-chain P@L |
|---|---:|---:|
| ESMC-300M | 0.552 ± 0.002 | 0.5387 |
| ESMC-600M | 0.589 ± 0.002 | 0.5803 |
| ESMC-6B | 0.725 ± 0.002 | running; exact result pending |

The paper and this reconstruction both source structures from the RCSB Protein
Data Bank and use the 2024-02-28 snapshot date. Our immutable build receipt
records 216,478 source structures and 781,077 extracted protein chains before
clustering. Both evaluations use a 20,775-chain population with the same
published 40%-identity clustering, length truncation, long-range-contact
definition, and filtering rules. However, the paper does not publish the
ordered chain identifiers, raw-file digests, or a manifest digest. We therefore
cannot prove that our 20,775 chains are identical item-for-item to Biohub's;
the columns are protocol-matched measurements, not a claim of a byte-identical
evaluation set. Our completed 1,024-chain diagnostic is not substituted for the
running 6B full-population measurement.

## Evaluation execution

`EVAL_PROFILE=full` runs all six frozen representation-probe contracts and
aggregates the four trusted tasks into P-CORE. It embeds protein means for all
108,215 sequences but writes residue embeddings only for the 11,411
secondary-structure sequences. The six tasks run as atomic, restartable
subprocesses with bounded parallelism, followed by a digest-checked reduction.
Secondary structure performs four full-residue LBFGS fits and is not suitable
for a short training gate.

The full contact evaluation uses the exact fast P@L path by default. It fits
the frozen probe once, binds the coefficients to the checkpoint and contact
manifest, and reuses that receipt across every deterministic inference shard.
Because the selected L1 probe is sparse, inference transfers and scores only
nonzero attention channels. Contact shards can run concurrently with
representation embedding. The merger restores the global SHA-ranked chain
order and performs the same 5,000-replicate chain bootstrap over all 20,775
rows; these execution changes do not alter the examples, probe, ordering, or
metric.

Static contact labels and eligible-pair geometry may also be cached once. The
optional cache is bound to the source payload and contact-manifest digests and
must pass a complete preflight hash check before inference. Without a cache,
the same fast probe-reuse and sparse-scoring path reads the frozen source
payloads directly.

Component receipts (`VALIDATION_MLM.json`, `CONTACT.json`, diagnostic
embedding, and per-task JSON) are written atomically. A later failure does not
erase completed work, and the runner reuses completed components on restart.
Output roots are checkpoint-specific; never point a different checkpoint at an
existing evaluation directory.

Execution improvements retained on `main` include cross-protein residue-budget
batching, secondary-structure-only residue caches, bounded parallel probe
processes, one-time contact-probe fitting, sparse contact scoring, an optional
receipt-bound contact cache, contact sharding, deterministic global P@L merge,
and atomic receipts. They change execution only; the examples, probes, row
ordering, metrics, and bootstrap remain fixed.

Build and verify the optional static cache once:

```bash
uv run --frozen python scripts/build_contact_scoring_cache.py \
  --dataset-root "$CONTACT_ROOT" \
  --external-src "$EXTERNAL_SRC" \
  --output-root "$CONTACT_SCORING_CACHE_ROOT"
```

Then run the full fast contact evaluation with four GPUs:

```bash
CONTACT_ROOT=/path/to/frozen-contact-data \
EXTERNAL_SRC=/path/to/evaluation-source \
CONTACT_SCORING_CACHE_ROOT=/path/to/contact-scoring-cache \
EVAL_GPUS=0,1,2,3 \
  bash runs/evaluate_p_at_l_parallel.sh
```

Omit `CONTACT_SCORING_CACHE_ROOT` to disable only the static cache. Probe reuse,
sparse scoring, deterministic sharding, and exact aggregation remain enabled.

Baseline training recipes, compute-matched comparisons, and fairness caveats
are documented in [`BASELINES.md`](BASELINES.md).

## Dataset provenance and split contract

The authoritative machine-readable receipt is `EVALUATION_SPLIT_LEDGER.json`,
built from the exact probe and contact payloads. The source publication defines
where each sequence, structure, or label originated; this table defines exactly
how the repository uses it.

- **Probe fit** fits a linear, ridge, or contact probe and never supplies a
  reported test score.
- **Validation** selects `C`, ridge `alpha`, or the contact-probe setting and
  never supplies a reported test score.
- **Test** is touched only after selection and supplies the reported metric and
  bootstrap uncertainty.

| Evaluation | Dataset lineage and publication | Probe fit | Validation | Final test | Metric and role |
|---|---|---|---|---|---|
| Held-out MLM | SHA-partitioned representatives from the post-exclusion UniRef90, MGnify, and OMG/IMG reservoirs. Sources: Suzek et al., [UniRef](https://doi.org/10.1093/bioinformatics/btu739); Richardson et al., [MGnify](https://doi.org/10.1093/nar/gkac1080); Cornman et al., [OMG](https://doi.org/10.1101/2024.08.14.607850). | Training corpus only | 4,096 representatives per source; 12,288 total | None | Sequence-mean NLL and perplexity; guardrail |
| Remote homology | TAPE-distributed SCOP 1.75 fold classification from DeepSF. Sources: Hou et al., [DeepSF](https://doi.org/10.1093/bioinformatics/btx780); Rao et al., [TAPE](https://proceedings.neurips.cc/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html). | 12,312 proteins | 736 proteins | 718 fold-holdout proteins | Balanced accuracy; family-group bootstrap; **P-CORE** |
| Secondary structure | TAPE/NetSurfP-2.0 train and validation payloads with CB513 as test. Sources: Klausen et al., [NetSurfP-2.0](https://doi.org/10.1002/prot.25674); Cuff and Barton, [CB513](https://pubmed.ncbi.nlm.nih.gov/10081963/); Rao et al., [TAPE](https://proceedings.neurips.cc/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html). | 8,678 proteins | 2,170 proteins | CB513: 513 records; 434 unique sequences | Residue macro-F1; protein bootstrap; **P-CORE** |
| Enzyme Commission | Sequence-only adaptation of the TorchDrug/TorchProtein `EnzymeCommission` artifact and its `<30%` identity test column. Sources: Gligorijević et al., [DeepFRI](https://doi.org/10.1038/s41467-021-23303-9); Zhang and Xu, [TorchProtein record](https://doi.org/10.5281/zenodo.6622158). | 15,551 proteins | 1,729 proteins | 720 proteins | Macro average precision; Bayesian label/group bootstrap; **quarantined** |
| DeepLoc2 | Official `multisub_5_partitions_unique.csv` with all five homology-aware partitions. Source: Thumuluri et al., [DeepLoc 2.0](https://doi.org/10.1093/nar/gkac278). | Three of five partitions per fold | Partition after the test partition | One of five partitions; every protein is test once | Macro average precision pooled over five folds; **P-CORE** |
| Human PPI | PEER release of Pan's HPRD-derived human interaction set with released negatives and redundancy-filtered split. Sources: Pan et al., [human PPI](https://doi.org/10.1021/pr100618t); Xu et al., [PEER](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e467582d42d9c13fa9603df16f31de6d-Abstract-Datasets_and_Benchmarks.html). | 35,669 pairs; 6,844 proteins | 315 pairs; 277 proteins | 237 pairs; 227 proteins | Average precision; connected-component bootstrap; **quarantined** |
| FLIP2 Hydro low-to-high | Official Hydrophobic Core `low_to_high` fitness split pooling variants of three wild types. Source: Didi et al., [FLIP2](https://doi.org/10.64898/2026.02.23.707496). | 9,974 variants | 2,493 variants within the training set | 12,468 high-fitness variants | Spearman correlation; variant-group bootstrap; **P-CORE** |
| Long-range contact | Experimentally determined structures from the frozen 2024-02-28 RCSB PDB snapshot, adapted to the ESM attention-to-contact protocol and ESMC long-range definition. Sources: Berman et al., [PDB](https://doi.org/10.1093/nar/28.1.235); Rao et al., [ESM contacts](https://openreview.net/forum?id=fylclEqgvgd); Candido et al., [ESMC](https://doi.org/10.64898/2026.06.03.729735). | 16 chains | 4 chains | 20,775 chains; 20,758 unique sequences | Mean precision at L; chain bootstrap; **primary selection axis** |

Source-paper headline metrics are not substituted for repository measurements.
Every released-checkpoint P@L value is recomputed with the frozen payload,
probe, and metric declared here. Remote homology uses balanced accuracy;
secondary structure uses residue macro-F1 with Q3 accuracy retained only as a
diagnostic; EC and DeepLoc2 use macro average precision; Human PPI uses average
precision; and FLIP2 uses Spearman correlation. The encoder remains frozen.

DeepLoc2 has no single permanent validation/test split. With test partition
`k`, validation is `(k + 1) mod 5`, and the other three partitions fit the
probe. The partition sizes are 5,963, 5,451, 5,731, 5,696, and 5,462 proteins,
for 28,303 total.

The contact manifest is selected deterministically from the 2024-02-28 PDB
snapshot. The first 16 chains fit the logistic attention probe, the next four
select its regularization, and the remaining 20,775 chains form the final P@L
evaluation. Each probe chain contributes at most 4,096 true-contact pairs and
4,096 non-contact pairs. After regularization selection, the probe is refit on
the 20 reserved chains and frozen. Test chains never fit or select the probe.
The ESMC paper specifies a small set of 20 training structures but does not
describe an internal fit/selection subdivision in its methods paragraph; the
explicit 16/4 split is this repository's frozen, leakage-resistant
formalization of those 20 structures.

## P-CORE

P-CORE summarizes the four trusted frozen-representation tasks: remote
homology, secondary structure, DeepLoc2, and FLIP2 Hydro low-to-high. Each raw
metric is converted to null-normalized task skill, and the four skills are
combined with a geometric mean. Enzyme Commission and Human PPI remain visible
diagnostics but are not part of P-CORE.

The values below were recomputed for the released Biohub ESMC checkpoints using
this repository's exact embeddings, splits, probes, and metrics. They were not
copied from the ESMC paper. Checkpoint revisions are `a59b831…` for 300M,
`a7e8201…` for 600M, and `45b0fa5…` for 6B.

| P-CORE task / metric | ESMC-300M | ESMC-600M | ESMC-6B |
|---|---:|---:|---:|
| Remote homology / balanced accuracy | 0.1159 | 0.1152 | 0.1186 |
| Secondary structure / residue macro-F1 | 0.8315 | 0.8407 | 0.8780 |
| DeepLoc2 / macro average precision | 0.6442 | 0.6568 | 0.6942 |
| FLIP2 Hydro low-to-high / Spearman correlation | 0.4132 | 0.4276 | 0.4616 |
| **P-CORE** | **37.4847** | **38.1553** | **40.6011** |

P-CORE is a reporting panel. It is not the current AutoResearch selection
metric.

### Experimental P-CORE expansion

P-CORE v0.5 alpha evaluates five additional runnable tasks, but it is not a
production aggregate and does not replace the four-task P-CORE score above.
The proposed CAFA5 task failed its preregistered population-size gate, and the
PRING task remains provisional.

| Alpha task | Primary metric | Status and source |
|---|---|---|
| CATH 4.4 remote retrieval | Macro H-superfamily mean average precision | Runnable diagnostic; [CATH 4.4](https://doi.org/10.1093/nar/gkae1087) |
| PRING Human C3-30 interaction | AUPRC | Provisional diagnostic; [PRING](https://github.com/SophieSarceau/PRING) |
| FLIP2 engineering shift | Hierarchical Spearman/NDCG score | Runnable bounded diagnostic; [FLIP2](https://doi.org/10.64898/2026.02.23.707496) |
| MegaScale family-screened stability | Median per-parent Spearman | Runnable bounded diagnostic; [MegaScale](https://doi.org/10.1038/s41586-023-06328-6) |
| CAID2-to-CAID3 Disorder-PDB | Macro-protein average precision | Runnable temporal diagnostic; [CAID3](https://doi.org/10.1002/prot.70045) |
| CAFA5 molecular function NK30 | Weighted Fmax | Blocked: 110 test proteins remained after the 30%-identity screen, below the preregistered minimum of 500; [CAFA5](https://doi.org/10.64898/2026.04.27.716980) |

## Contact P@L

Contact P@L uses attention maps rather than final-layer frozen embeddings. For
every chain, all layer/head attention planes are symmetrized, and a logistic
probe is fit on 16 structures and selected on four held-out structures. A true
contact is a Cβ distance below 8 Å (Cα for glycine); long range means a sequence
separation of at least 24. The score is the mean precision among the top L
eligible residue pairs for a chain of evaluated length L.

### Where the 20,775 chains come from

This is a post-filter population, not a convenient round-number sample. Every
structure and contact label comes from the 2024-02-28 RCSB PDB snapshot; the
P@L dataset is not sourced from UniRef90, MGnify, or OMG/IMG. The pipeline
extracts protein chains from that snapshot and applies the ESMC paper's contact
preprocessing. Chains are clustered at 40% sequence identity with MMseqs2
Linclust using 80% shorter-sequence coverage
(`--min-seq-id 0.4 -c 0.8 --cov-mode 1 --cluster-mode 2 --kmer-per-seq 100`).
Twenty structures are reserved for probe construction; this repository freezes
them as 16 probe-fit and four probe-validation chains. Evaluation sequences are
truncated to 510 residues, affecting 3,425 chains, and structures with fewer
than L true long-range contacts are removed. The remaining manifest has exactly
20,775 evaluation chains representing 20,758 unique sequences.

Released-model results are reported once in the compact full-population table
at the beginning of this document. The fixed 1,024-chain subset remains an
execution diagnostic only; a full value is never estimated from that subset.

## Quarantined tasks

Enzyme Commission and Human PPI remain executable and their raw metrics are
reported, but neither may influence model selection or P-CORE. EC has an
unresolved cross-scale anomaly: released ESMC-300M and 600M score about 71
null-normalized skill while 6B collapses to 1.607 despite a clean integrity
audit. Human PPI has only 237 test pairs, released-model scale ordering is
non-monotonic, and the four-hour undertrained checkpoint (0.7979 AP) sits close
to released ESMC-300M (0.8155 AP). This is inadequate discrimination for a
promotion gate.

Human PPI can return only after a preregistered replacement adds substantially
more family-disjoint test pairs, hard negatives, a leakage audit, and
scale/checkpoint ranking validation. EC requires independent reproduction and
repair before reinstatement.

## Frozen-embedding probe path

The encoder is never updated by these tasks. For a sequence, final-layer
residue vectors are deterministically windowed and mean-pooled over valid
residues to produce one protein vector. Remote homology uses a multinomial
logistic probe; EC and DeepLoc2 use one-vs-rest multilabel logistic probes;
FLIP2 uses ridge regression. Secondary structure skips pooling and applies a
linear classifier to each residue vector. Human PPI concatenates the symmetric
pair features `abs(z_a - z_b)` and `z_a * z_b` before logistic regression.

The feature standardizer and probe are fit on probe-training data only. `C` or
ridge `alpha` is selected on validation, and the selected probe is applied once
to test; it is not retrained on validation. Bootstrap intervals resample frozen
test predictions by biological group and do not refit the encoder or probe.

The complete evaluation-union construction, exact-match and homology exclusion
statistics, validation selection, and released-shard receipts are documented
once in [`DATA.md`](DATA.md).
