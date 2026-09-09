# Usage

## Setup

The [README](../README.md#setting-up-data--environments) contains the supported setup and training
quickstart. Run `bash runs/setup.sh` to prepare the environment and
training data and both MLM/P@L evaluation assets, or `bash runs/speedrun.sh` to perform setup and train the default
recipe in one call. The only local settings are `DATA_ROOT` and `OUTPUT_ROOT`
in an optional `.env`; defaults are the repository's `data/` and `outputs/`.

```text
$DATA_ROOT/
  training/             # Verified training subset and MLM validation data
  cache/                # Downloaded corpus shards and contact archive
  evaluation/contact/   # Frozen 20-chain probe + 20,775-chain P@L dataset
  evaluation/source/    # Pinned source containing autoresearch_esm
$OUTPUT_ROOT/
  <run-name>/           # Checkpoints, effective config, logs and evaluation records
```

The default pins the release revision and downloads **30 of 565 training Parquet
shards**: 13 UniRef90, 3 MGnify and 14 OMG/IMG, containing 29,979,351 training
proteins. Downloads including MLM validation occupy 5.62 GB; prepared token stores
add 9.38 GB. Allow 20 GB for the complete data setup, excluding the environment
and checkpoints. All three MLM validation shards (12,288 proteins) and the frozen
[CONTACT_DATA.md](CONTACT_DATA.md) bundle are always prepared; no P-CORE data
are downloaded. At 100k steps and batch 1,024, the default corpus is sampled for
about 3.4 passes. See [DATA.md](DATA.md#sizing-a-training-download) for larger selections.

For a different training corpus size, choose a fresh `DATA_ROOT` in `.env` and run:

```bash
bash runs/setup.sh --training-shards 105
```

The range is **3–565 total training shards**, with at least one per source.
Selection extends the source with the least coverage of the 36:11:54 sampling
mixture; all selections are deterministic source prefixes. `565` selects the
entire training release. Setup without this option reuses the stored shard count
on later calls, including calls from speedrun. An explicit different count refuses
to overwrite existing prepared data; use another root for that experiment.
The autoresearch task and historical leaderboard retain the original 7-shard
selection. In a separate `DATA_ROOT`, prepare it explicitly with
`bash runs/setup.sh --training-shards 7`; changing the general setup default does
not change the frozen benchmark corpus.

For full control, the ordinary data API accepts either `--training-shards` or
`--training-samples`; it always includes all MLM validation shards. Inspect a
selection without downloading training shards:

```bash
uv run --frozen python -m nanoprotein.sharded_data \
  --revision bd38448d50d8f426d7b9bd4410b53159ea001259 --training-shards 30 \
  --cache-root data/cache --output-root data/training --plan-only
```

The contact installer can also use an already downloaded bundle offline:

```bash
uv run --frozen python -m nanoprotein.setup_evaluation \
  --data-root data --archive /path/to/contact-evaluation-v1.tar.gz
```

## Training

The default 171M model targets small-budget experiments and follows the paper's
170M scaling backbone ([Table S4](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1.full.pdf#page=29)).
For larger models, select a [300M or 600M reference config](../configs/reference/README.md);
their local training assumptions are documented alongside the presets.

The speedrun is a readable shell script that calls the ordinary Python API:

```bash
# Same best recipe, fresh output directory, different seed.
bash runs/speedrun.sh configs/default.yaml setting3-seed42 --seed 42

# Original AdamW recipe with its original one-hour budget and no step cap.
bash runs/speedrun.sh configs/esmc-171m-original.yaml adamw-1h \
  --max-steps none --walltime-seconds 3600
```

It uses four GPUs. Arguments after the recipe and run name pass through to
`nanoprotein.train`, overriding the default 100k-step/16-hour limits. For
example, `--attention-backend flash` selects FA2 on L40S; a long run on slower
hardware may also need a larger `--walltime-seconds` guard. `.env` contains paths,
not these execution settings. Each output directory must be fresh.

For other GPU counts or full control, call the training API directly:

```bash
set -a
if [ -f .env ]; then source .env; fi
source .env.example
set +a
uv run --frozen python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m nanoprotein.train --config configs/default.yaml \
  --max-steps 100000 --walltime-seconds 57600 \
  --data-root "$DATA_ROOT/training" --output-root "$OUTPUT_ROOT/setting3-direct"
```

The recipe owns model and optimizer settings. CLI options select the execution
budget and can override seed, attention, warmup and batch layout. Every run
records the source config hash and saves its effective `config.yaml`. To inspect
the resolved recipe without GPUs or training:

```bash
uv run --frozen python -m nanoprotein.train --config configs/default.yaml \
  --seed 42 --max-steps 100000 --walltime-seconds 57600 --print-config
```

Budget arguments accept `none` to clear inherited step/token limits. Batch-layout
overrides require a single-stage recipe. Full final checkpoints include optimizer
state; see [continuation](checkpoint-resume.md) for resuming on another GPU count.

## AutoResearch

[program.md](../program.md) directs an agent to the selected
[task definition](../tasks/171m-validation-loss.md). Run one research measurement:

```bash
bash tasks/171m-validation-loss_ar.sh configs/default.yaml experiment-001
```

The task script loads `.env`, checks frozen inputs and four L40S GPUs, runs seeds
42 and 43 through the standard training/evaluation APIs, and reports mean loss
and sample SD in `$OUTPUT_ROOT/experiment-001/summary.json`. It records full
contact P@L as a diagnostic. Search and acceptance decisions belong to the agent.

## Evaluation

Setup installs all MLM validation data plus the frozen contact payload and
evaluator under `$DATA_ROOT/evaluation/{contact,source}`. The installer checks
all frozen hashes before reporting success and verifies existing installations
on reuse. See [contact data provenance](CONTACT_DATA.md) and
[evaluation provenance](EVALUATION.md#dataset-provenance-and-split-contract).

With the two roots loaded in your shell, evaluate a saved checkpoint:

```bash
uv run --frozen python -m nanoprotein.evaluate \
  --checkpoint "$OUTPUT_ROOT/setting3-100k/checkpoint-final.pt" \
  --data-root "$DATA_ROOT/training" --output-root "$OUTPUT_ROOT/setting3-100k/evaluation" \
  --validation-batches 1024 --validation-batch-size 4 --validation-context 512 \
  --run-contact --contact-chains 20775 --contact-bootstrap 5000 \
  --contact-root "$DATA_ROOT/evaluation/contact" --external-src "$DATA_ROOT/evaluation/source"
```

Omit `--run-contact` and the contact arguments for MLM alone. Parallel evaluation
helpers live in `src/`; [EVALUATION.md](EVALUATION.md) documents their interfaces.
Test of Progress is owner-run using the [manual commands](EVALUATION.md#manual-test-of-progress).
There is no verification launcher.

## Repository layout

See [AGENTS.md](../AGENTS.md) for concise layout and modification guidance.

```text
src/nanoprotein/   # Training, models, data, evaluation and runtime CLI modules
src/*.sh          # Optional parallel evaluation launchers
runs/             # Public setup and speedrun scripts
tasks/            # Autoresearch definition and measurement command
.dev/scripts/     # Plotting, release preparation and historical analysis tools
.dev/tests/       # Developer regression tests
.dev/reports/     # Published experiment records and figures
```

The package uses a standard src layout. Run setup after updating an existing
checkout to refresh the installed package. Direct commands now use
`python -m nanoprotein.train` and `python -m nanoprotein.evaluate`.
The old `nano_protein` namespace and thin scripts/train.py, scripts/evaluate.py
and scripts/download_data.py wrappers have been retired. Existing tensor/state-dict
checkpoints remain loadable; their saved recipe values and model names are unchanged.

After setup, run the developer tests from the repository root:

```bash
.venv/bin/python -m unittest discover -s .dev/tests -q
```
