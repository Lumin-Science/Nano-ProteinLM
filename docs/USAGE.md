# Usage

## Setup

The [README](../README.md#setup) contains the supported setup and training
quickstart. Run `bash runs/setup_env_and_data.sh` to prepare the environment and
training data, or `bash runs/speedrun.sh` to perform setup and train the default
recipe in one call. The only local settings are `DATA_ROOT` and `OUTPUT_ROOT`
in an optional `.env`; defaults are the repository's `data/` and `outputs/`.

```text
$DATA_ROOT/
  training/             # Verified training subset and MLM validation data
  cache/                # Downloaded corpus shards
  evaluation/contact/   # Frozen contact dataset, installed separately
  evaluation/source/    # Pinned source containing autoresearch_esm
$OUTPUT_ROOT/
  <run-name>/           # Checkpoints, effective config, logs and evaluation records
```

The setup script pins the release revision and downloads the same whole-shard
prefix used by the benchmark: 7,109,469 training proteins, plus all 12,288
validation proteins. Training repeatedly samples this subset; it does not
consume the full 666.0M release. To prepare a different corpus size for independent
research, call `python -m nanoprotein.sharded_data` directly with `--training-samples`,
`--revision`, `--cache-root` and `--output-root`.

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
[task definition](../task/171m-validation-loss.md). Run one research measurement:

```bash
bash task/171m-validation-loss_ar.sh configs/default.yaml experiment-001
```

The task script loads `.env`, checks frozen inputs and four L40S GPUs, runs seeds
42 and 43 through the standard training/evaluation APIs, and reports mean loss
and sample SD in `$OUTPUT_ROOT/experiment-001/summary.json`. It records full
contact P@L as a diagnostic. Search and acceptance decisions belong to the agent.

## Evaluation

Training and MLM validation are ready after setup. For contact P@L, install the
frozen contact payload under `$DATA_ROOT/evaluation/contact/` and the pinned
evaluator source under `$DATA_ROOT/evaluation/source/`; existing verified
directories can be linked there. These are not downloaded by the setup script.
Public packaging remains [release work](CONFIG_CLEANUP_PLAN.md); see
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

```text
src/nanoprotein/   # Training, models, data, evaluation and runtime CLI modules
src/*.sh          # Optional parallel evaluation launchers
runs/             # Public setup and speedrun scripts
task/             # Autoresearch definition and measurement command
.dev/scripts/     # Plotting, release preparation and historical analysis tools
.dev/reports/     # Published experiment records and figures
```

The package uses a standard src layout. Run setup after updating an existing
checkout to refresh the installed package. Direct commands now use
`python -m nanoprotein.train` and `python -m nanoprotein.evaluate`.
The old `nano_protein` namespace and thin scripts/train.py, scripts/evaluate.py
and scripts/download_data.py wrappers have been retired. Existing tensor/state-dict
checkpoints remain loadable; their saved recipe values and model names are unchanged.
