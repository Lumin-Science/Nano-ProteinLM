# Data contract

## Public release

The supported training artifact is
[`LuminScience/LuminBench-Nano-ESMC`](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC).
Production pins immutable commit
[`bd38448d`](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC/tree/bd38448d50d8f426d7b9bd4410b53159ea001259),
whose independently verified manifest contains 665,970,495 training proteins:

| Released source arm | Training proteins | Training residues | Parquet shards |
|---|---:|---:|---:|
| UniRef90 | 74,175,974 | 24,582,966,441 | 92 |
| MGnify | 328,949,335 | 61,464,625,352 | 229 |
| OMG/IMG | 262,845,186 | 65,256,646,612 | 244 |
| **Total** | **665,970,495** | **151,304,238,405** | **565** |

Each source also has one 4,096-protein validation shard, for 12,288 validation proteins in total. The complete artifact contains 568 train/validation shards and occupies 109,661,312,410 compressed bytes. Setup downloads 30 training shards by default (29,979,351 proteins), plus all validation shards. `bash scripts/setup.sh --training-shards N` selects 3–565 whole training shards; choose a fresh `DATA_ROOT` for another selection. The direct data API also supports a requested sample budget. Both routes use checksum-bound source prefixes and always include complete MLM validation. The setup command separately installs the [frozen P@L bundle](#frozen-contact-evaluation-data).

The companion [raw clustering release](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC-RAW)
preserves the source-specific 70%-identity representative FASTAs and cluster
membership maps before evaluation decontamination and final packing. Use the
processed release above for training; the raw release supports inspection of
the preceding clustering stage.

## Sizing a training download

Counts below come from the pinned manifest and the ordinary `plan_shards` API.
The three source counts are ordered UniRef90 / MGnify / OMG-IMG; every selection
also includes all three MLM validation shards.

| Training shards | Source allocation | Training proteins | Stored training residues | Parquet GB | Parquet + prepared stores GB |
|---|---|---:|---:|---:|---:|
| 7 — historical sequential-search corpus | 3 / 1 / 3 | 7,109,469 | 1,879,045,806 | 1.32 | 3.51 |
| 30 — setup default | 13 / 3 / 14 | 29,979,351 | 8,053,052,338 | 5.62 | 15.00 |
| 105 — 100k × 1,024 | 46 / 8 / 51 | 103,867,089 | 28,185,691,687 | 19.64 | 52.40 |
| 209 — 100k × 2,048 | 91 / 16 / 102 | 206,909,262 | 56,102,947,191 | 39.09 | 104.30 |
| 565 — full release | 92 / 229 / 244 | 665,970,495 | 151,304,238,405 | 109.66 | 290.27 |

Storage estimates use decimal GB and include MLM validation: the cache holds compressed Parquet, and prepared stores use one byte per residue plus a 44-byte index entry per protein (offset, length and digest). Array headers and receipts add a small amount. The P@L v2 archive is 167,956,554 bytes and expands to 663,147,593 bytes; reserve about 0.9 GB with filesystem overhead. Allow roughly 20 GB for 30 shards, 60 GB for 105, 120 GB for 209 or 320 GB for the full release, with separate space for dependencies, checkpoints and evaluation outputs.

At 100k steps, global batches 1,024 and 2,048 sample 102.4M and 204.8M proteins respectively. The 105- and 209-shard rows cover the expected draws under the normalized 36:11:54 mixture before sampling headroom. Source selection is stochastic. The trainer checks each source with 1% headroom by default and stops on exhaustion unless the recipe explicitly permits resampling; 30 shards are insufficient for either no-repeat run.

For the default mixture at 100k steps and batch 1,024, use `bash scripts/setup.sh --training-samples 103424000` in a fresh `DATA_ROOT`; batch 2,048 needs `--training-samples 206848000`. These budgets include the default 1% headroom, and the planner rounds each source up to whole shards. Changed mixtures require their own source-coverage check. `DATA_COVERAGE.json` records the resolved budget and policy. `data_resampling: allow` enables intentional repeated-data experiments, which must report their reuse; headroom alone is not a guarantee against stochastic exhaustion.

**Stored residues are not the training token budget.** Stage 1 crops proteins
to at most 510 residues and adds BOS/EOS; padding is excluded from model tokens.
The completed 100k-step, batch-1,024 default-recipe run sampled 102.4M proteins and
processed 24,200,224,761 model tokens. This defines the 24.20B-token final-evaluation
target; actual steps can differ with another sequence-length mix.
Data can be reused across seeds and recipes without downloading it again.

Historical campaigns retain their seven-shard corpus. Use `bash scripts/setup.sh --training-shards 7` in a dedicated `DATA_ROOT` to reproduce those records. The current [AutoResearch protocol](AUTORESEARCH.md#design-space) permits data selection and source-mixture changes within the provided corpus; record the selected shards, mixture and source exposure for each recipe. Existing prepared roots retain their saved shard count, so select a fresh root to change it.

## Nano-ESMC production funnel

The following table mirrors Table 5 of the [technical report](../.dev/report/main.pdf) and is the central
ledger for the complete path from pinned upstream objects to the public
Hugging Face commit. Rejection columns are disjoint accounting categories.
For OMG/IMG, source assignment occurs before the length and ambiguity filters.

| Source | Original records | MGnify-origin excluded | `<60 aa` excluded | `>20%` ambiguous excluded | Quality eligible | Distinct sequences in source | 70%-identity representatives | After evaluation decontamination |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UniRef90 | 170,669,877 | 0 | 4,755,787 | 29,797 | 165,884,293 | 165,884,293 | 92,230,941 | 74,181,938 |
| MGnify | 729,215,663 | 0 | 117,372,608 | 26,021 | 611,817,034 | 611,788,129 | 348,135,082 | 330,444,705 |
| OMG/IMG | 3,280,269,924 | 1,048,850,563 | 145,672,826 | 0 | 2,085,746,535 | 963,673,186 | 324,923,979 | 282,202,559 |
| **Source-view total** | **4,180,155,464** | **1,048,850,563** | **267,801,221** | **55,818** | **2,863,447,862** | **1,741,345,608** | **765,290,002** | **686,829,202** |

### 1. Pin the source objects

| Arm | Pinned input | Integrity pin |
|---|---|---|
| UniRef90 | official UniRef 2023_02 archive | 211,819,312,677 bytes; MD5 `353681f464572bb199fa032f714d4669` |
| MGnify | official Protein DB 2023_02 `mgy_clusters.fa.gz` | 83,473,342,442 bytes; MD5 `332d36d2a943bdb769237a03e050ed03` |
| OMG/IMG | all 959 objects in `tattabio/OMG` | 1,253,813,127,320 bytes; every path, size, and LFS SHA-256 pinned |

### 2. Normalize, assign the OMG source, and quality-filter

Whitespace and terminal `*` characters are removed, letters are uppercased,
and unsupported characters are mapped to `X`. Sequences shorter than 60 amino
acids or with more than 20% non-canonical residues are rejected.

The OMG payload contains both IMG/JGI-style and MGnify/ENA-derived accession
classes. Numeric accessions of at least seven digits and `Ga`, `IMG`, or `JGI`
prefixes define the IMG/JGI-role arm. `ERZ`, `ERR`, `ERS`, `ERP`, and `MGY`
accessions are counted as **MGnify-origin excluded**. They are valid records,
not quality failures; they are excluded so MGnify is not sampled once through
the dedicated MGnify arm and again through the OMG/IMG arm. An unrecognized OMG
accession fails the build instead of being silently assigned.

### 3. Collapse exact duplicates and reduce source redundancy

Accepted normalized sequences are SHA-256 partitioned into 256 buckets. Exact
sequence strings are globally deduplicated while every source membership and
original identifier is retained. There are 1,661,993,387 globally unique
normalized sequences; the source-membership views sum to 1,741,345,608 because
one sequence can occur in more than one source.

Each source view is then clustered independently with MMseqs2 Linclust at 70%
sequence identity and 80% shorter-sequence coverage
(`--cov-mode 1 --cluster-mode 2`).

“Distinct sequences in source” means distinct normalized sequence strings that
carry that source membership after global exact deduplication. The source rows
are not mutually exclusive: the same sequence may occur in more than one arm.

The final column removes 21,653 exact evaluation matches and 78,439,147
additional homologs against the final 317,000-sequence protected union. These
78,460,800 sequential decontamination rejections are not the entire difference
between the representative and released-training counts: cross-source ownership,
the storage-length gate, and validation selection reduce the final training
release to 665,970,495 proteins, as itemized in step 5.

### 4. Freeze the protected evaluation union

Every probe-fit, validation, gallery/reference, test, and blocked candidate
split available at release-build time is protected—not only the headline test
sets. After exact cross-task deduplication, this union contains 317,000 unique
sequences and 88,031,573 residues. It includes all 20 probe chains and all
20,775 reported contact-evaluation chains from the frozen 2024-02-28 RCSB
Protein Data Bank snapshot.

### 5. Remove exact matches and homologs

Exact sequence digests are excluded independently. MMseqs2 then searches with
the evaluation union as query and all training representatives as targets. A
representative is excluded at 30% or greater sequence identity, at least 80%
coverage of both query and target, and E-value at most 0.001:

```text
--min-seq-id 0.30 -c 0.80 --cov-mode 0 -e 0.001 -s 7.5 --max-seqs 1000000
```

Keeping the evaluation proteins on the query side is part of the contract:
audits found that reversing query and target lost valid pairs in the MMseqs2
heuristic prefilter. The one-million-candidate cap is checked for every query.

The immutable release manifest accounts for the exclusions sequentially. Thus
the homology column below means additional homology rejections after exact
matches, and the cross-source column means duplicate non-owner memberships not
already assigned to an earlier exclusion.

| Source | 70% representatives | Evaluation exact | Evaluation homology | Cross-source non-owner | Length | Validation | Final training |
|---|---:|---:|---:|---:|---:|---:|---:|
| UniRef90 | 92,230,941 | 17,200 | 18,031,803 | 0 | 1,868 | 4,096 | 74,175,974 |
| MGnify | 348,135,082 | 1,737 | 17,688,640 | 1,491,154 | 120 | 4,096 | 328,949,335 |
| OMG/IMG | 324,923,979 | 2,716 | 42,718,704 | 19,353,277 | 0 | 4,096 | 262,845,186 |
| **Total** | **765,290,002** | **21,653** | **78,439,147** | **20,844,431** | **1,988** | **12,288** | **665,970,495** |

Before sequential exclusion accounting, the cross-source ownership receipt
finds 24,657,087 shared representative sequences and 25,478,264 duplicate
source memberships. The first owner in UniRef90, MGnify, OMG/IMG order is kept,
so source mixing cannot overweight an exact duplicate.

### 6. Select validation and write deterministic shards

After decontamination and the 32--16,384-residue storage filter, the first
4,096 eligible proteins in SHA order from each source become validation data.
The union of those validation digests is excluded from every training arm.
Remaining records are written in ascending sequence-SHA order as Parquet with
three fields:

```text
sequence: string
sha256: string
length: int32
```

A new shard starts before crossing 268,435,456 uncompressed residues. Each
shard receipt binds its row and residue counts, compressed bytes, SHA-256, and
minimum/maximum sequence digest.

### 7. Verify, publish, and pin

An independent verifier rehashes every Parquet file and every sequence, checks
length and strict ordering, proves zero within- or cross-source training
duplicates, recomputes the exact/homology intersection, and proves global
train/validation separation. Only that verified manifest is published. The Hub
commit linked above is the resulting immutable release; training resolves it to
that exact commit before downloading any shard and records the selected files
and hashes in its local receipt.

## Source terms

Lumin Science's selection, arrangement, decontamination ledger, packing, and
release metadata are distributed under CC BY-SA 4.0. Upstream records retain
their own terms: UniRef90 is CC BY 4.0; the direct OMG distribution declares CC
BY-SA 4.0; MGnify remains under the EMBL-EBI Terms of Use plus applicable
original-owner rights. Tokenization and packing do not relicense underlying
sequences; the full notices and path-specific attribution accompany the public
manifest.

## Main corpus gap relative to ESMC

The main data-scale gap is the JGI-role corpus. The stage-aligned comparison
below places the **Clusters** column of Table S2 in the original
[ESMC paper](https://doi.org/10.64898/2026.06.03.729735) beside our
post-clustering **70%-identity representatives**. Both populations have already
been reduced independently within each source at 70% sequence identity.

| Source role | ESMC Table S2 clusters | Nano-ESMC 70%-identity representatives | Nano / ESMC scale |
|---|---:|---:|---:|
| UniRef | approximately 83 M | 92,230,941 | approximately 111% |
| MGnify | approximately 372 M | 348,135,082 | approximately 94% |
| **JGI role / OMG-IMG surrogate** | **approximately 2 B** | **324,923,979** | **approximately 16%** |
| **Total** | **approximately 2.455 B** | **765,290,002** | **approximately 31%** |

UniRef and MGnify are near the reported ESMC scale. The roughly 1.68-billion
representative shortfall in the JGI-role arm accounts for almost the entire
overall gap. Nano-ESMC preserves ESMC's Stage-1 source weighting despite this:
the reported 36:11:54 weights are normalized to 35.64:10.89:53.47 at runtime.

The paper's main text separately says that the source data contains 156 million
UniRef 2023_02 sequences, 621 million MGnify 2023_02 sequences, and 2.029
billion sequences from a JGI snapshot downloaded in July 2023. Those are
pre-clustering context, not the Table S2 pool used for the comparison above.
Because ESMC does not publish a source-by-source filtering ledger, we do not
equate those numbers with any specific Nano-ESMC pre-clustering column.

This comparison aligns processing stages; it does not establish identical processing. The public OMG/IMG arm is an open surrogate for the JGI role rather than the authors' July 2023 JGI snapshot. Nano-ESMC follows the reported source roles, source-wise 70%-identity reduction, and Stage-1 36:11:54 sampling weights. It samples one available representative per reconstructed cluster rather than the paper's cluster-then-member draw because the transferred cluster-membership tables do not contain member sequences. ESMC Stage 2 uses 63:6:31 weights and a 2,048-token context, which lies outside the current 171M AutoResearch task's fixed Stage-1 context.

The source publications are Suzek et al.
([UniRef](https://doi.org/10.1093/bioinformatics/btu739)), Richardson et al.
([MGnify](https://doi.org/10.1093/nar/gkac1080)), and Cornman et al.
([OMG](https://doi.org/10.1101/2024.08.14.607850)).

## Frozen contact evaluation data

The [P@L setup bundle](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC/blob/5eae416dbb415d2df206b9641dd5ddabe04371dc/evaluation/contact-evaluation-v2.tar.gz) contains 16 probe-fit chains, four probe-validation chains, all 20,775 evaluation chains and six frozen evaluator source files. Version 2 removes experiment-report links and machine paths from packaging metadata; chain payloads, splits and numerical source files retain their original hashes. Fresh setup downloads this immutable release. Existing verified research installations remain supported, and historical downloads remain available at their original revision.

Structures come from the **2024-02-28 RCSB Protein Data Bank snapshot**. Chain
selection and preprocessing remain unchanged: the benchmark is paper-faithful,
not claimed to be identical to the ESMC authors' unpublished chain selection.
The [evaluation contract](EVALUATION.md#contact-pl) defines probe fitting,
long-range contacts and confidence intervals.

PDB archive data are available under **CC0 1.0**, per the
[wwPDB usage policy](https://www.wwpdb.org/about/usage-policies).
Please acknowledge the PDB and original structure authors. Chain IDs remain in
`CONTACT_MANIFEST.jsonl`; PDB entry pages provide the associated publications.

> Berman, H. M. et al. The Protein Data Bank. *Nucleic Acids Research* 28,
> 235–242 (2000). [doi:10.1093/nar/28.1.235](https://doi.org/10.1093/nar/28.1.235).

The evaluator source is distributed under this repository's [MIT license](../LICENSE).
Its six source files are copied byte-for-byte from the recorded evaluator bundle;
`SOURCE_MANIFEST.json` retains their SHA-256 hashes. The contact manifest hash is
`c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3`.
The payload inventory hash is
`1b73f5f466420c8d0c74be452ebabe46af837482cee357674cad01d99e6f4b70`.

Setup verifies the archive checksum, source files, manifest, inventory and every chain payload before reporting success. Both `evaluation/contact/` and `evaluation/source/` live beneath `DATA_ROOT`.
