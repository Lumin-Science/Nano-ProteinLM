# LuminBench Nano ESMC

A compact, auditable, uv-managed stack for training protein embedding models
and evaluating them with frozen protein benchmarks.

`main` contains the supported training, data, and evaluation path. Active
research, reports, completed experiment receipts, and AutoResearch instructions
are isolated under [`dev/`](dev/).

## Quick start

`uv.lock` is the sole Python environment contract:

```bash
uv sync --frozen

uvx --from huggingface-hub hf download \
  LuminScience/LuminBench-Nano-ESMC \
  --repo-type dataset \
  --local-dir data/processed/stage1-300m-production-v1

DATA_ROOT=$PWD/data/processed/stage1-300m-production-v1 \
WALLTIME_SECONDS=60 \
  bash runs/qualify.sh
```

The canonical four-A100 training and full evaluation entrypoint is:

```bash
bash runs/stage1_300m_4xa100_4h.sh
```

Every training command fails closed unless the corpus has matching content
hashes, exact evaluation exclusion, all-splits MMseqs2 homology exclusion, zero
excluded-sequence intersections, and zero train-validation overlap.

## Why the dataset is 2.55 GiB

The Hugging Face release is the **local Stage-1 production subset**, not the
full source reservoir. It contains 3,000,000 training representatives and 4,096
validation representatives from each of UniRef90, MGnify, and OMG/IMG:
9,012,288 proteins and 2,345,995,606 residues in total.

Residues are stored as one-byte ESMC token IDs. The payload is therefore about
2.346 GB of tokens plus 396 MB of train indexes and small validation/metadata
files: 2.55 GiB on disk. This is enough diversity for the current four-A100
two-, four-, and sixteen-hour experiments. A future eight-H100 seven-day run
requires a larger sharded release or deterministic streaming from the full
70%-cluster reservoir; it must not be described as training on only this subset
without reporting repeated exposure.

The full controlled reservoir remains much larger: 92,230,941 UniRef90,
348,135,082 MGnify, and 324,923,979 OMG/IMG 70%-identity cluster
representatives. See [`docs/DATA.md`](docs/DATA.md).

There is no second GitHub repository for the binary corpus. Code, builders, and
portable manifests live here; the separately versioned data artifact belongs in
[`LuminScience/LuminBench-Nano-ESMC`](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC).

## Full-reservoir and budget-sized downloads

The full release is organized as immutable, SHA-ordered Parquet shards rather
than one large archive. A run asks for its estimated total sequence exposures;
the downloader selects the minimum per-source shard prefixes that cover that
budget under the configured source mixture, pins one Hugging Face commit,
downloads every validation shard, checks all file and sequence hashes, and
materializes the existing fast mmap layout:

```bash
uv run --frozen python scripts/download_data.py \
  --repo-id LuminScience/LuminBench-Nano-ESMC \
  --revision <immutable-release-commit> \
  --training-samples 5376000 \
  --cache-root data/cache/full-open-v2 \
  --output-root data/processed/run-prefix
```

This is shard-level on-demand download, like nanochat—not row-level network
streaming during training. The full raw-to-release builder, all 959 pinned OMG
object hashes, and the measured processing report live under
[`dev/data/`](dev/data/).

## Distribution license

The released database compilation—our selection, arrangement, decontamination
ledger, packing, and release metadata—is distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Source protein
records retain their upstream terms and are marked path-by-path: UniRef90 under
CC BY 4.0, OMG/IMG under CC BY-SA 4.0, and MGnify under the EMBL-EBI Terms of
Use plus applicable original-owner rights. The umbrella license grants only
rights Lumin Science holds; it does not relicense upstream records.

The code remains under the repository [`LICENSE`](LICENSE). Full data notices
are in the Hugging Face release template under
[`release/huggingface/stage1-300m-production-v1/`](release/huggingface/stage1-300m-production-v1/).

## Production evaluation performance

The accelerated evaluator is part of `main`, not `dev/`. It preserves the
frozen metrics while reducing wasted compute:

- packs windows from different proteins into residue-budgeted GPU batches;
- stores residue embeddings only for secondary-structure sequences instead of
  all P-CORE proteins;
- runs exact P-CORE probes concurrently as bounded, restartable subprocesses;
- shards the full 20,775-chain contact P@L evaluation across three GPUs while
  P-CORE uses the fourth; and
- merges P@L rows back into the global deterministic order before the unchanged
  5,000-replicate chain bootstrap.

The implementation is in [`nano_protein/evaluate.py`](nano_protein/evaluate.py),
[`runs/evaluate_full_parallel.sh`](runs/evaluate_full_parallel.sh), and
[`scripts/merge_full_evaluation.py`](scripts/merge_full_evaluation.py). Contract
tests cover packed embedding parity and deterministic P@L merging.

## Repository map

```text
configs/          supported production training configuration
nano_protein/     tokenizer, ESMC model, mmap data, trainer, evaluation
runs/             production preparation, training, and evaluation entrypoints
scripts/          uv-invoked command-line interfaces and receipt verification
tests/            fast contract, parity, and determinism checks
docs/             stable architecture, data, evaluation, and release contracts
release/          portable release cards and immutable-manifest templates
dev/              AutoResearch, proposed work, reports, plans, and run receipts
```

Source and issues: [`Lumin-Science/LuminBench-Nano-ESMC`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC).
