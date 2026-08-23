# LuminBench Nano ESMC

A small, auditable, uv-managed stack for training protein embedding models under
fixed compute and evaluating them with frozen protein benchmarks.

This `auto-research` branch optimizes frozen long-range contact precision at L
(P@L) on four GPUs with a two-hour training-loop budget. The research contract
is in [`program.md`](program.md).

## Mandatory production data

Every training command fails closed unless the selected corpus contains:

- exact SHA-256 exclusion against P-CORE and P@L;
- verified MMseqs2 homology exclusion against **all** evaluation splits;
- a matching homology-exclusion receipt and corpus-verification receipt; and
- zero excluded train/validation sequences and zero train-validation overlap.

The canonical corpus is `data/processed/stage1-300m-production-v1`. Its compact,
tracked receipts are in [`results/stage1-300m-production-v1/`](results/stage1-300m-production-v1/).
The mmap payload remains in controlled storage and is ignored by Git.

Prepare or independently verify it with:

```bash
DATA_ROOT=$PWD/data/processed/stage1-300m-production-v1 \
  bash runs/prepare_stage1_300m_corpus.sh

uv run --frozen python scripts/verify_prepared_corpus.py \
  --data-root "$PWD/data/processed/stage1-300m-production-v1"
```

The preparer requires both the MMseqs2 exclusion digest set and its verified
receipt. There is no exact-only preparation mode.

## Environment and qualification

`uv.lock` is the sole Python environment contract. Use `uv sync --frozen` and
`uv run --frozen`; ambient environments are not supported.

```bash
uv sync --frozen
DATA_ROOT=$PWD/data/processed/stage1-300m-production-v1 \
WALLTIME_SECONDS=60 \
  bash runs/qualify.sh
```

Qualification uses the production-gated 300M Stage-1 configuration and the
same mandatory corpus verifier as every full run.

## Frozen production baselines

The completed four- and sixteen-hour ESMC-300M Stage-1 baselines are retained
as decontaminated reference points:

```bash
bash runs/stage1_300m_4xa100_4h.sh
bash runs/stage1_300m_4xa100_16h.sh
```

Both default to `stage1-300m-production-v1`, run all Python through uv, and use
the full parallel evaluator. Results and immutable receipts live under
[`results/`](results/).

## Repository map

```text
configs/          production-gated training and proposed evaluation contracts
nano_protein/     tokenizer, ESMC model, mmap data, trainer, evaluation
runs/             production preparation, training, and evaluation entrypoints
scripts/          uv-invoked command-line interfaces and receipt verification
tests/            fast contract and determinism checks
docs/             scientific and operational decisions
report/           NeurIPS-style experiment report
results/          compact completed-run and data receipts
```

## Publication layout

- Source and issues: [`Lumin-Science/LuminBench-Nano-ESMC`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC)
- Production training dataset: [`LuminScience/LuminBench-Nano-ESMC`](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC)
- Approved checkpoints: [`LuminScience`](https://huggingface.co/LuminScience)
- Large raw corpora remain in controlled storage and are not republished without
  a source-license audit.

No GitHub or Hugging Face remote is created by a training command.
