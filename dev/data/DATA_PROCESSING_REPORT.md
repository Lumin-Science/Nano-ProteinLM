# Full open-protein corpus: processing and release report

Status date: 2026-08-23  
Build host: `tmoss` (`moss`; 255 online logical CPUs, 1 TB RAM)  
Canonical builder: `dev/data/process_full_corpus.py`  
Python environment: repository `uv.lock`; no ad-hoc pip environment

## Executive result

The “400 GB dataset” was previously three different objects described as though
they were one:

| Object | Measured size | Proteins represented | Publish? |
|---|---:|---:|---|
| Raw upstream downloads | 1.55 TB compressed | all raw records | No; reproducible from pinned upstream objects |
| Step-9 directory including cluster-membership TSVs and MMseqs artifacts | 459 GiB | 765,290,002 representatives plus cluster metadata | No; the TSVs are audit/build material, not training samples |
| Three representative FASTAs | 241.60 GB decimal (225.01 GiB) | 765,290,002 | Input to the final screen and shard build |
| Final decontaminated Parquet release | measured after the Q9 screen and shard verification | measured after exclusions | Yes, after the explicit publication gate |

The release is therefore not a 400 GB monolithic file. It is a complete,
SHA-ordered Parquet reservoir split into roughly 256 Mi-residue immutable shards.
Training first downloads only the minimum whole-shard prefix needed for its
planned sample budget, plus all small validation shards. There is no per-row
network access in the training loop.

## 1. Source lineage and redistribution marking

The data repository uses `license: cc-by-sa-4.0` for Lumin Science's original
selection, arrangement, decontamination ledger, packing, and metadata. This is
an umbrella for rights Lumin Science holds, not an attempt to relicense every
third-party record.

| Arm | Exact input | Pin | Governing terms in the release |
|---|---|---|---|
| UniRef90 | official UniRef 2023_02 archive | 211,819,312,677 bytes; MD5 `353681f464572bb199fa032f714d4669` | [UniProt CC BY 4.0](https://www.uniprot.org/help/license) |
| MGnify | official Protein DB 2023_02 `mgy_clusters.fa.gz` | 83,473,342,442 bytes; MD5 `332d36d2a943bdb769237a03e050ed03` | [EMBL-EBI Terms of Use](https://www.ebi.ac.uk/about/terms-of-use/) plus applicable original-owner rights; not relicensed by Lumin Science |
| OMG/IMG | all 959 Parquet objects in `tattabio/OMG` | 1,253,813,127,320 bytes; every path, size, and LFS SHA-256 pinned in `omg_upstream_shards.tsv` | [CC BY-SA 4.0 as declared by OMG](https://huggingface.co/datasets/tattabio/OMG) |

EMBL-EBI states that it adds no restrictions beyond original owners and expects
attribution; it also warns that community contributors remain owners and that
third-party rights can apply. The MGnify path therefore carries those terms
verbatim rather than an invented Creative Commons license.

## 2. Cleaning, exact deduplication, and diversity clustering

Every sequence is processed in this fixed order:

1. remove whitespace and terminal `*`, uppercase, and map unsupported characters
   to `X`;
2. reject sequences shorter than 60 residues;
3. reject sequences with more than 20% non-canonical residues;
4. for OMG, keep numeric/JGI/IMG accessions and reject `ERZ`/MGnify-origin rows,
   preventing a second MGnify arm;
5. SHA-256 partition all accepted records into 256 buckets;
6. globally collapse identical normalized strings while retaining every source
   and original identifier in membership Parquet;
7. emit a source view for each source membership; and
8. independently run MMseqs2 Linclust per source at 70% identity and 80%
   shorter-sequence coverage (`--cov-mode 1 --cluster-mode 2`).

“Wrong source” is not a quality judgment. It means that an OMG record has an
MGnify-origin accession and would duplicate the dedicated MGnify source arm.

### Measured full-corpus accounting

| Source | Original records | `<60 aa` | `>20%` ambiguous | Wrong source | Quality eligible | Exact unique in source | 70% representatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| UniRef90 | 170,669,877 | 4,755,787 | 29,797 | 0 | 165,884,293 | 165,884,293 | 92,230,941 |
| MGnify | 729,215,663 | 117,372,608 | 26,021 | 0 | 611,817,034 | 611,788,129 | 348,135,082 |
| OMG/IMG | 3,280,269,924 | 145,672,826 | 0 | 1,048,850,563 | 2,085,746,535 | 963,673,186 | 324,923,979 |

Across source memberships there are 1,661,993,387 exact unique normalized
sequences before source-specific 70% clustering. The representative counts sum
to 765,290,002. Cluster membership TSVs do not carry member sequences and are not
part of the training download.

## 3. Evaluation decontamination

### Policy

Pretraining excludes homology to **all** benchmark splits, not just headline test
sets. That means probe-fit/train, validation, gallery/reference, test, and blocked
candidate splits are queries. This conservative choice prevents representation
pretraining from receiving sequences later used to fit a linear/contact probe,
choose probe hyperparameters, or score the final model.

The frozen MMseqs2 criterion is:

```text
--min-seq-id 0.30 -c 0.80 --cov-mode 0 --max-seqs 1000000 -s 7.5
```

`--cov-mode 0` requires the 80% coverage threshold on both query and target. The
finalizer rejects any emitted row below identity or either coverage threshold,
rejects unknown query IDs, and fails if any query reaches the one-million-hit cap.
Exact SHA-256 exclusions are also applied independently.

### Verified parent screen

The complete legacy screen queried 116,841 unique sequences against all
765,290,002 representatives. It covers P@L probe-fit, validation, and all 20,775
test chains plus every split of P-CORE v0.2. Its verified output excludes
62,229,127 unique representatives:

| Source target | Representatives matched |
|---|---:|
| UniRef90 | 14,947,338 |
| MGnify | 14,938,117 |
| OMG/IMG | 36,214,295 |

The global count is lower than the source sum because identical digest IDs can
retain membership in more than one source arm. The parent exclusion digest file
is 4,044,893,255 bytes with SHA-256
`3738a5bcab185f553613480a17d3ecbdd309323b9920ed66a7d6fb5700fecaab`.

### Q9 extension

P-CORE Q9 adds six task files. The builder reads the task files themselves—not
the smaller runnable index—so blocked CAFA sequences remain protected. It also
extracts both PRING partners and the mutant plus wild-type MegaScale sequences.

| Q9 task | Row sequence occurrences screened | Unique task sequences | Special handling |
|---|---:|---:|---|
| CATH44 retrieval | 11,180 | 11,177 | gallery and test protected |
| CAFA5 MF NK30 hard | 78,747 | 77,742 | blocked for scoring, still protected |
| CAID3 disorder | 666 | 666 | train, valid, test protected |
| FLIP2 shift | 104,946 | 68,888 | train, valid, test protected |
| MegaScale family30 ddG | 176,344 | 88,468 | mutant and wild type protected |
| PRING human C3-30 | 334,642 | 9,775 | both proteins in every pair protected |

After cross-task and legacy exact deduplication, the frozen union contains
317,000 sequences and 88,031,573 unique residues. Of these, 200,159 are new
relative to the parent screen. The union artifacts are:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `evaluation_all_splits.fasta` | 112,397,257 | `1b8f43a95719d37674b184ed42097388b891ea59ca2c559713fea48e05c2e7e1` |
| `evaluation_q9_delta.fasta` | 67,904,864 | `15c8449452335a5e99f83ab368aaf9ab46f60c0aaba8cac105ead75756ef97bd` |
| `sequence_memberships.jsonl` | 77,449,335 | `3d41b4bf311da1c7144c4d4e362f6316e9e2e11427c8f72b085fb43717025a71` |
| `evaluation_exact_sha256.txt` | 20,605,000 | `45b352a403b9b8456ad31d14e441a351acfa358bea9939e60690cc5c479f11e1` |

The Q9-only delta is searched against the same verified complete MMseqs target
databases used by the parent run. Its validated target digest union is then
externalized with the immutable parent exclusions. This is set-equivalent to
rerunning all 317,000 queries and avoids repeating the 116,841-query parent work.

Full delta screen: Slurm job `10096`; final measured hit/exclusion counts are
written here only after `HOMOLOGY_EXCLUSION_VERIFIED.json` exists.

## 4. Final train/validation selection and sharding

For each source's 70%-cluster representatives, the release builder:

1. recomputes the sequence SHA-256 and verifies the FASTA header;
2. externally sorts the deterministic-but-not-lexical MMseqs output through 256
   leading-byte digest buckets, then sorts each bounded bucket in memory;
3. removes every exact evaluation digest;
4. removes every parent-or-Q9 MMseqs target digest;
5. applies the training storage length range 32--16,384 residues;
6. takes 4,096 eligible SHA-ordered representatives per source for validation;
7. removes the union of all validation digests from every source's training arm;
8. writes the remaining complete reservoir in ascending SHA-256 order; and
9. starts a new Parquet shard before crossing 268,435,456 uncompressed residues.

The Parquet schema is deliberately minimal:

```text
sequence: string
sha256: string
length: int32
```

Each shard receipt records rows, residues, bytes, SHA-256, minimum sequence
digest, and maximum sequence digest. The independent verifier rehashes every
file and sequence, checks length and strict order, rejects duplicates within a
source split, recomputes the exact/homology intersection, and checks global
train-validation separation before promoting `manifest.generated.json` to the
downloadable `manifest.json`.

## 5. Partial download for a training budget

The supported command is:

```bash
uv sync --frozen
uv run --frozen python scripts/download_data.py \
  --repo-id LuminScience/LuminBench-Nano-ESMC \
  --revision <immutable-HF-commit> \
  --training-samples 5376000 \
  --cache-root data/cache/full-open-v2 \
  --output-root data/processed/run-prefix
```

The planner normalizes the configured 36:11:54 source weights, computes the
required unique records in each arm, and chooses the smallest prefix of each
arm whose manifest row counts meet that budget. It always downloads all
validation shards. Every downloaded file is checked against the release
manifest, then sequences are rehashed while producing the mmap layout consumed
by the existing trainer.

This differs from remote streaming: a run downloads and caches complete,
immutable Parquet shards before using them. Hugging Face also supports selective
snapshot downloads through `allow_patterns`, but the repository planner uses
the manifest so it can select by sample budget instead of requiring users to
guess filenames.

## 6. Reproduction commands on a 64-CPU host

The exact pinned OMG manifest is checked into this directory. Representative
commands are below; all outputs should be placed outside Git.

```bash
uv sync --frozen
PIPE=dev/data/process_full_corpus.py
ROOT=/absolute/path/to/protein-corpus

uv run --frozen python "$PIPE" download \
  --data-root "$ROOT" --omg-manifest dev/data/omg_upstream_shards.tsv

# Run UniRef and MGnify concurrently. Split the 959 OMG paths into several
# normalize invocations; each writes its own normalized directory.
uv run --frozen python "$PIPE" normalize --source uniref90 \
  --input "$ROOT/raw/uniref90_2023_02/uniref2023_02.tar.gz" \
  --output "$ROOT/normalized/uniref90"
uv run --frozen python "$PIPE" normalize --source mgnify \
  --input "$ROOT/raw/mgnify_2023_02/mgy_clusters.fa.gz" \
  --output "$ROOT/normalized/mgnify"

uv run --frozen python "$PIPE" deduplicate \
  --input "$ROOT/normalized/uniref90" \
  --input "$ROOT/normalized/mgnify" \
  --input "$ROOT/normalized/omg-batch-00" \
  --output "$ROOT/deduplicated"

for source in uniref90 mgnify omg_img; do
  uv run --frozen python "$PIPE" cluster \
    --dedup-root "$ROOT/deduplicated" \
    --source "$source" --output "$ROOT/clusters/$source" \
    --mmseqs /absolute/path/to/mmseqs --threads 64
done
```

The evaluation-union, delta-screen, finalization, sharding, and verification
commands are fully enumerated in `run_q9_delta_screen_tmoss.sbatch` and the CLI
help. Every output directory is create-once: the script refuses to overwrite an
existing result, making resume decisions explicit.

## 7. Hugging Face publication procedure

The checked release should be uploaded as a large folder, not committed through
Git LFS manually:

```bash
HF_XET_HIGH_PERFORMANCE=1 uv run --frozen python scripts/upload_data.py \
  --release-root /absolute/path/to/verified-release \
  --repo-id LuminScience/LuminBench-Nano-ESMC \
  --confirm-public-repo LuminScience/LuminBench-Nano-ESMC
```

Hugging Face recommends Parquet for large datasets, `upload_large_folder` for
prepared large directories, fewer than 10,000 files per folder, and files below
200 GB. This design is far below the per-file and per-folder limits. The final
upload remains a separate, explicit public-publication action after the manifest,
license/attribution card, dataset viewer, and downstream download smoke test pass.
The uploader refuses to run unless the independent receipt binds the final
manifest, every required metadata file is present, Xet high-performance mode is
enabled, and the confirmation value exactly matches the destination repo.

## 8. Fail-closed release gates

Publication is blocked unless all of the following are true:

- source pins and all 959 OMG object hashes match;
- cleaning accounting sums exactly to input counts;
- representative FASTA hashes and counts match Step-9 receipts;
- the 317,000-sequence evaluation ledger hashes match;
- all delta hit rows meet the frozen MMseqs thresholds and no query reaches the cap;
- the final exclusion file is the exact parent-plus-delta set union;
- all Parquet file and sequence hashes pass;
- exact/homology exclusion intersection is zero;
- train-validation intersection is zero across every pair of source arms; and
- the partial downloader passes the same training-side Q9-aware gate as the full release.
