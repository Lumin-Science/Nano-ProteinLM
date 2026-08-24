---
pretty_name: LuminBench Nano ESMC Full Open Reservoir v2
license: cc-by-sa-4.0
size_categories:
  - 100M<n<1B
tags:
  - biology
  - protein
  - protein-language-model
  - esmc
  - decontaminated
  - parquet
---

# LuminBench Nano ESMC Full Open Reservoir v2

This is the complete decontaminated 70%-identity representative reservoir for
[`Lumin-Science/LuminBench-Nano-ESMC`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC).
It is organized as immutable, SHA-ordered Parquet shards so each run can download
only the smallest deterministic prefix required by its training budget.

Do not use this template as a release receipt. Measured post-Q9 counts, bytes,
checksums, and the immutable Hub revision are inserted only after the full shard
verifier succeeds.

## License and source terms

Lumin Science's original database selection, arrangement, decontamination
ledger, packing, and metadata are offered under CC BY-SA 4.0. Third-party
sequence records retain their source terms:

| Path | Direct source | Governing terms |
|---|---|---|
| `train/uniref90/**`, `validation/uniref90/**` | UniRef90 2023_02 | [CC BY 4.0](https://www.uniprot.org/help/license) |
| `train/mgnify/**`, `validation/mgnify/**` | MGnify Protein DB 2023_02 | [EMBL-EBI Terms of Use](https://www.ebi.ac.uk/about/terms-of-use/) plus applicable original-owner rights; not relicensed by Lumin Science |
| `train/omg_img/**`, `validation/omg_img/**` | JGI/IMG records from `tattabio/OMG` | [CC BY-SA 4.0](https://huggingface.co/datasets/tattabio/OMG) |

Redistribution notes and modifications are preserved in
`LICENSE_AND_ATTRIBUTION.md` and `SOURCE_PROVENANCE.json`.

## Decontamination

Every probe-fit, validation, reference/gallery, test, and blocked-candidate
sequence from contact P@L, P-CORE v0.2, and P-CORE v0.5-alpha-q9 is protected.
The full representative reservoir is screened with MMseqs2 at 30% sequence
identity and 80% query plus target coverage. Exact SHA-256 matches are excluded
independently. The released validation union is excluded from every training
source arm.

## Download only what a run needs

```bash
git clone https://github.com/Lumin-Science/LuminBench-Nano-ESMC
cd LuminBench-Nano-ESMC
uv sync --frozen
uv run --frozen python scripts/download_data.py \
  --repo-id LuminScience/LuminBench-Nano-ESMC \
  --revision <immutable-release-commit> \
  --training-samples 5376000 \
  --cache-root data/cache/full-open-v2 \
  --output-root data/processed/run-prefix
```

The command fetches `manifest.json`, converts the requested total sample count
to per-source requirements using the 36:11:54 mixture, downloads the minimum
whole-shard prefix for each source plus every validation shard, verifies the
checksums, and materializes the existing mmap training layout. Training is local;
it does not make row-level network requests.

## Parquet schema

| Column | Type | Meaning |
|---|---|---|
| `sequence` | string | normalized amino-acid sequence |
| `sha256` | string | SHA-256 of the ASCII sequence |
| `length` | int32 | residue count |

See the GitHub processing report and `manifest.json` for full lineage, measured
counts, rejection accounting, shard hashes, and the complete build recipe.
