---
pretty_name: LuminBench Nano ESMC Stage-1 300M Production v1
license: other
license_name: mixed-source-terms
license_link: https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC/blob/main/LICENSE_AND_ATTRIBUTION.md
size_categories:
  - 1M<n<10M
tags:
  - biology
  - protein
  - protein-language-model
  - esmc
  - decontaminated
  - mmap
---

# LuminBench Nano ESMC Stage-1 300M Production v1

This is the only supported Stage-1 training corpus for
[`Lumin-Science/LuminBench-Nano-ESMC`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC).
It contains deterministic, memory-mapped token stores for training a 300M-parameter
ESMC-like protein masked language model.

## License at a glance

This is a **mixed-terms dataset**, so the repository metadata deliberately uses
`license: other`. No single repository-level license replaces the upstream terms:

| Path | Direct source | Governing terms |
|---|---|---|
| `uniref90/**` | UniRef90 2023_02 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `mgnify/**` | MGnify Protein Database 2023_02 | [EMBL-EBI Terms of Use](https://www.ebi.ac.uk/about/terms-of-use/), including original-data-owner rights |
| `omg_img/**` | JGI/IMG rows from `tattabio/OMG` | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) as declared by the direct OMG distribution |
| release documentation and project-authored metadata | Lumin Science | CC BY 4.0, excluding embedded upstream material and factual identifiers |

Read [`LICENSE_AND_ATTRIBUTION.md`](LICENSE_AND_ATTRIBUTION.md) before using or
redistributing any data. The binary token stores are reversible representations of
protein sequences; tokenization does not remove upstream obligations.

## Contents

| Source arm | Train proteins | Validation proteins | Train residues | Validation residues |
|---|---:|---:|---:|---:|
| UniRef90 | 3,000,000 | 4,096 | 1,012,397,817 | 1,373,842 |
| MGnify | 3,000,000 | 4,096 | 565,676,089 | 753,905 |
| OMG/IMG | 3,000,000 | 4,096 | 764,769,134 | 1,024,819 |
| **Total** | **9,000,000** | **12,288** | **2,342,843,040** | **3,152,566** |

The payload contains 9,012,288 proteins, 2,345,995,606 residue tokens, and about
2.55 GiB of data before repository metadata. Each source is stocked with the same
number of representatives; training samples the three stores using the configured
mixture (the current Stage-1 default is 36:11:54 for UniRef90:MGnify:OMG/IMG,
normalized at runtime).

## File format

Each `<source>/<split>/` directory contains:

- `tokens.bin`: concatenated ESMC residue-token IDs as unsigned 8-bit integers;
- `index.npy`: a NumPy structured array with little-endian fields
  `offset: uint64`, `length: uint32`, and `digest: 32-byte SHA-256`.

Offsets and lengths point into `tokens.bin`. Digests identify the normalized amino-acid
sequence. The 33-token sequence vocabulary and the loader are implemented in
`nano_protein/tokenizer.py` and `nano_protein/data.py` in the GitHub repository.

`manifest.json` is the immutable preparation receipt. Its absolute filesystem paths are
historical provenance pointers from the build host, not paths required to load the
downloaded stores. `CORPUS_VERIFICATION.json` binds the payload hashes and records the
post-write decontamination checks. `RELEASE_MANIFEST.json` is the portable file ledger.

## Construction

The source reservoir was built as follows:

1. download and checksum UniRef90 2023_02, MGnify Protein DB 2023_02 cluster
   representatives, and all 959 pinned Parquet shards of `tattabio/OMG`;
2. retain only numeric-accession JGI/IMG rows from OMG and exclude its `ERZ...`
   MGnify-origin rows so MGnify is not counted twice;
3. normalize sequence spelling, reject short/high-ambiguity records, and globally
   exact-deduplicate while retaining source membership;
4. cluster each source arm independently at 70% identity and 80% coverage;
5. select deterministic representative prefixes and exclude every exact or homologous
   hit to the evaluation corpus;
6. apply the 32--16,384-residue prepared-corpus length gate and deterministic
   SHA-modulus train/validation assignment; and
7. independently verify file hashes, zero train/validation overlap, and zero intersection
   with the combined exact-plus-homology exclusion set.

The homology screen used MMseqs2 at at least 30% identity, 80% query coverage, and
80% target coverage against all evaluation fit, validation, and test splits. It screened
116,841 unique evaluation sequences and excluded 1,592,566 candidate representatives.
The final verifier reports zero excluded-sequence intersections in every released split.

See `SOURCE_PROVENANCE.json` for exact upstream URLs, versions, byte counts, checksums,
filter counts, and transformations. The release does not contain the raw archives or the
evaluation sequences.

## Download and use with uv

Pin the immutable Hugging Face commit recorded in the matching GitHub release rather than
depending on the mutable `main` revision:

```bash
uvx --from huggingface-hub hf download \
  LuminScience/LuminBench-Nano-ESMC \
  --repo-type dataset \
  --revision <HUGGING_FACE_COMMIT> \
  --local-dir data/processed/stage1-300m-production-v1

uv sync --frozen
uv run python scripts/train.py \
  --config configs/esmc_300m_stage1_4xa100_4h.yaml \
  --data-root data/processed/stage1-300m-production-v1
```

The training entrypoint fails closed if the manifest, source hashes, all-splits homology
contract, or corpus-verification receipt does not match.

## Limitations

- This is a deterministic subset of one representative per transferred 70%-identity
  cluster, not the full source reservoirs and not ESMC's undisclosed member-within-cluster
  sampling corpus.
- Source annotation and taxonomy are not included; the release is sequence-only.
- The fixed 3M representatives per arm are a production reservoir. A run samples from
  these stores according to its configured source mixture and may revisit examples.
- Protein databases can contain deposited errors, synthetic sequences, sensitive
  biodiversity information, patents, or other third-party rights. Users remain
  responsible for their intended use.
- The license review documents the direct-source terms available on 2026-08-23 and is
  not legal advice.

## Attribution

Please cite the original resources and this release. Full notices and the OMG paper
citation are in [`LICENSE_AND_ATTRIBUTION.md`](LICENSE_AND_ATTRIBUTION.md).
