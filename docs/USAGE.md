# Usage

## Setup

The [README](../README.md#setting-up-data--environments) contains the supported setup and training
quickstart. Run `bash scripts/setup.sh` to prepare the environment and
training data and both MLM/P@L evaluation assets, or `bash scripts/speedrun.sh` to perform setup and train the default
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

The default pins the release revision and downloads **30 of 565 training Parquet shards**: 13 UniRef90, 3 MGnify and 14 OMG/IMG, containing 29,979,351 training proteins. Downloads including MLM validation occupy 5.62 GB; prepared token stores add 9.38 GB. Allow 20 GB for the complete data setup, excluding the environment and checkpoints. All three MLM validation shards (12,288 proteins) and the frozen [contact evaluation](DATA.md#frozen-contact-evaluation-data) bundle are always prepared; no P-CORE data are downloaded. A 100k-step run at batch 1,024 needs a larger download: the default trainer prevents source resampling. See [DATA.md](DATA.md#sizing-a-training-download) for sample budgets and source coverage.

For a different training corpus size, choose a fresh `DATA_ROOT` in `.env` and run:

```bash
# 100k steps × batch 1,024, with 1% sampling headroom for the default mixture.
bash scripts/setup.sh --training-samples 103424000
```

With `--training-shards N`, the range is **3–565 total training shards**, with at least one per source. Selection extends the source with the least coverage of the 36:11:54 sampling mixture; all selections are deterministic source prefixes. `565` selects the entire training release. Setup without a selection argument reuses the stored shard count on later calls, including calls from speedrun. An explicit different count refuses to overwrite existing prepared data; use another root for that experiment.

Historical campaigns used seven shards; reproduce those records with `bash scripts/setup.sh --training-shards 7` in a separate `DATA_ROOT`. The current protocol permits data selection and source-mixture changes within the provided corpus. Prepare enough data for the selected recipe and retain its manifest.

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
  --data-root data --archive /path/to/contact-evaluation-v2.tar.gz
```

## Training

The default 171M model targets small-budget experiments and follows the paper's
170M scaling backbone ([Table S4](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1.full.pdf#page=29)).
[configs/README.md](../configs/README.md) lists the search-setting and final-evaluation recipes.

The speedrun is a readable shell script that calls the ordinary Python API:

```bash
# Same best recipe, fresh output directory, different seed.
bash scripts/speedrun.sh configs/test-100k/nanop-best-171m-round2.yaml round2-seed42 --seed 42

# Plain ESMC reference with the same 100k-step budget.
bash scripts/speedrun.sh configs/test-100k/esmc-171m.yaml esmc-171m-100k
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
  -m nanoprotein.train --config configs/test-100k/nanop-best-171m-round2.yaml \
  --max-steps 100000 --walltime-seconds 57600 \
  --data-root "$DATA_ROOT/training" --output-root "$OUTPUT_ROOT/default-direct"
```

The recipe owns model and optimizer settings. CLI options select the execution
budget and can override seed, attention, warmup and batch layout. Every run
records the source config hash and saves its effective `config.yaml`. To inspect
the resolved recipe without GPUs or training:

```bash
uv run --frozen python -m nanoprotein.train --config configs/test-100k/nanop-best-171m-round2.yaml \
  --seed 42 --max-steps 100000 --walltime-seconds 57600 --print-config
```

Budget arguments accept `none` to clear inherited step/token limits. Batch-layout overrides require a single-stage recipe. Full final checkpoints include optimizer state. Pass `--resume /path/to/checkpoint-final.pt` to the training API with the original recipe and a fresh output directory to continue training; preserve the global batch size when changing the GPU count.

## AutoResearch

[171m-validation-loss.md](../tasks/171m-validation-loss.md) and [171m-p-at-l.md](../tasks/171m-p-at-l.md) each contain the complete scientific task definition for any AutoResearch method, and [AUTORESEARCH.md](AUTORESEARCH.md) defines the shared search budget and final evaluation. Our sequential-search method comes in two programs: [karpathy_ar_reward_gate.md](../autoresearch/karpathy_ar_reward_gate.md) keeps a candidate by a fixed two-seed rule, and [karpathy_ar_agent_gate.md](../autoresearch/karpathy_ar_agent_gate.md) lets the agent decide. To start it, select the task and program and give your agent the following instruction.

> Read `autoresearch/karpathy_ar_reward_gate.md` and start autoresearch for `tasks/171m-validation-loss.md`.

Use `tasks/171m-p-at-l.md` in that instruction to optimize contact P@L instead, or `autoresearch/karpathy_ar_agent_gate.md` to let the agent decide what to keep.

After setup and GPU allocation, run one research measurement:

```bash
bash tasks/171m-validation-loss_ar.sh configs/autoresearch/esmc-171m.yaml experiment-001 42
# Or use P@L as the reward with the same measurements:
bash tasks/171m-p-at-l_ar.sh configs/autoresearch/esmc-171m.yaml experiment-p-at-l-001 42
```

The third argument supplies the training seed. Each task script loads `.env`, qualifies the declared GPU model and attention backend, saves the recipe and performs one training run through the standard APIs. Its `evaluation/EVALUATION.json` reports loss on all 12,288 validation proteins and P@L over all 20,775 contact chains. The validation-loss task scores `validation_mlm.sequence_mean_nll` (lower is better); the P@L task scores `contact.precision_at_l` (higher is better). Replication, aggregation and acceptance belong to the caller; [our sequential method](AUTORESEARCH_BASELINE.md#running-the-example-loop) documents those choices and commands. The agent reviews task boundaries and run completion.

## Evaluation

After setup, `bash scripts/speedrun.sh --evaluate default-100k` loads your local paths and evaluates that run’s final checkpoint on all 12,288 MLM validation proteins and parallel contact P@L over all 20,775 chains. Replace `default-100k` with another run name; evaluation CLI options can follow it. This command performs evaluation only.

Setup installs all MLM validation data plus the frozen contact payload and evaluator under `$DATA_ROOT/evaluation/{contact,source}`. The installer checks all frozen hashes before reporting success and verifies existing installations on reuse. See [contact data provenance](DATA.md#frozen-contact-evaluation-data) and [evaluation provenance](EVALUATION.md#dataset-provenance-and-split-contract).

With the two roots loaded in your shell, evaluate a saved checkpoint. Contact P@L uses parallel workers across the visible GPUs by default (32 workers on four GPUs), with one shared probe and the full 20,775-chain population:

```bash
uv run --frozen python -m nanoprotein.evaluate \
  --checkpoint "$OUTPUT_ROOT/default-100k/checkpoint-final.pt" \
  --data-root "$DATA_ROOT/training" --output-root "$OUTPUT_ROOT/default-100k/evaluation" \
  --validation-context 512 \
  --run-contact --contact-chains 20775 --contact-bootstrap 5000 \
  --contact-root "$DATA_ROOT/evaluation/contact" --external-src "$DATA_ROOT/evaluation/source"
```

Omit `--run-contact` and the contact arguments for MLM alone. Use `--contact-mode serial` for one-process contact evaluation, or `--contact-gpus` and `--contact-workers` to select devices and concurrency. Both research tasks use the standard parallel evaluator automatically. [EVALUATION.md](EVALUATION.md#evaluation-execution) documents caching, resuming and the compatibility launcher. Final evaluation is owner-run; [AUTORESEARCH.md](AUTORESEARCH.md#final-evaluation) gives its command.

## Repository layout

See [AGENTS.md](../AGENTS.md) for concise layout and modification guidance.

```text
src/nanoprotein/   # Training, models, data, evaluation and runtime CLI modules
src/*.sh          # Optional parallel evaluation launchers
configs/          # Search-setting (autoresearch/) and final-evaluation (test-100k/) recipes
scripts/          # Setup and speedrun scripts
tasks/            # Autoresearch definition and measurement command
autoresearch/     # Agent research-loop guidance
.dev/             # Development log, TODO list and technical report
```

The package uses a standard src layout. Run setup after updating an existing
checkout to refresh the installed package. Direct commands now use
`python -m nanoprotein.train` and `python -m nanoprotein.evaluate`.
Existing checkpoints remain loadable; saved recipe values and model names are unchanged.
