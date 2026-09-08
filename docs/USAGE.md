# Usage

## Requirements

Use Linux, a compatible NVIDIA driver and `uv >=0.11.31,<0.12`. Python and
project dependencies are pinned in the repository. Research uses four L40S GPUs;
the recommended 100k-step training example uses four H100s and BF16/FA3.

## Setup

Run commands from the repository root. Copy the environment example and edit
only the two root paths; no cache, model, seed or budget settings belong in `.env`.

```bash
cp .env.example .env
# Edit DATA_ROOT and OUTPUT_ROOT in .env for this machine.
set -a
source .env
set +a
uv sync --frozen
```

The standard layout is:

```text
$DATA_ROOT/
  training/             # Verified prepared corpus, including MLM validation data
  evaluation/contact/   # Frozen contact dataset
  evaluation/source/    # Pinned evaluator source containing autoresearch_esm
  cache/                # Downloaded corpus shards, managed by data preparation
$OUTPUT_ROOT/
  <experiment>/         # Checkpoints, effective configs, logs and evaluation records
```

Prepare the fixed corpus used by the benchmark:

```bash
uv run --frozen python scripts/download_data.py \
  --repo-id LuminScience/LuminBench-Nano-ESMC \
  --revision bd38448d50d8f426d7b9bd4410b53159ea001259 \
  --training-samples 5376000 \
  --cache-root "$DATA_ROOT/cache" --output-root "$DATA_ROOT/training"
```

Install the frozen contact payload and evaluator source at the locations above
before contact evaluation; existing verified directories can be linked there.
See [evaluation provenance](EVALUATION.md#dataset-provenance-and-split-contract).
Contact data/source packaging for a fresh public installation remains part of
the [release cleanup](CONFIG_CLEANUP_PLAN.md); these are not downloaded by the
training-data command. Training and MLM evaluation do not require them.

## Training

Use the [README's direct training command](../README.md#training-a-170m-model). It selects
[Setting 3](../configs/default.yaml), with 100,000 steps and a 16-hour guard.
The recipe contains model/optimizer settings; command-line options select the
execution budget and can override seed, attention, warmup and batch layout.
Every run records the source config hash and writes its effective `config.yaml`.

To inspect a resolved recipe without GPUs or training:

```bash
uv run --frozen python -m nano_protein.train --config configs/default.yaml \
  --seed 42 --max-steps 100000 --walltime-seconds 57600 --print-config
```

Budget arguments accept `none` to clear an inherited step/token limit explicitly.
Batch-layout overrides require a single-stage recipe. Use fresh output directories;
see [checkpoint continuation](checkpoint-resume.md) for resuming saved optimizer state.

## AutoResearch

[program.md](../program.md) directs an agent to the selected
[task definition](../task/171m-validation-loss.md). Run one complete research
measurement with the transparent shell example:

```bash
bash task/171m-validation-loss_ar.sh configs/default.yaml experiment-001
```

It loads `.env`, checks the frozen inputs and four L40S GPUs, runs seeds 42 and 43
through the ordinary training/evaluation APIs, and reports mean loss and sample
SD in `$OUTPUT_ROOT/experiment-001/summary.json`. It records full contact P@L as a
diagnostic. It does not search, decide acceptance or launch Test of Progress.

## Evaluation

To evaluate one saved checkpoint directly:

```bash
uv run --frozen python -m nano_protein.evaluate \
  --checkpoint "$OUTPUT_ROOT/setting3-100k/checkpoint-final.pt" \
  --data-root "$DATA_ROOT/training" --output-root "$OUTPUT_ROOT/setting3-100k/evaluation" \
  --validation-batches 1024 --validation-batch-size 4 --validation-context 512 \
  --run-contact --contact-chains 20775 --contact-bootstrap 5000 \
  --contact-root "$DATA_ROOT/evaluation/contact" --external-src "$DATA_ROOT/evaluation/source"
```

Test of Progress is a separate, owner-run experiment using the
[manual commands](EVALUATION.md#manual-test-of-progress). There is no verification
launcher. Optional parallel evaluation tools and metric details are documented
in [EVALUATION.md](EVALUATION.md).

## Existing launch helpers

`runs/speedrun.sh` remains an optional data-preparation/training convenience. It
now reads the same two roots, defaults to Setting 3 for 100k steps on four H100s,
qualifies the configured attention backend, and uses normal UV cache defaults.
Its `RUN_NAME` selects a child directory of `OUTPUT_ROOT`; it no longer treats
`OUTPUT_ROOT` as the individual run directory or `DATA_ROOT` as prepared data.
Other files in `runs/` preserve historical interfaces until the remaining cleanup.
The direct Python APIs remain available independently of these helpers.
