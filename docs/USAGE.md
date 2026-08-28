# Usage

## Requirements

The supported training path requires `uv >=0.11.31,<0.12`, a compatible NVIDIA
driver, and four visible BF16-capable GPUs. Python and project dependencies are
defined by `.python-version`, `pyproject.toml`, and `uv.lock`.

## Train

From a fresh clone:

```bash
git clone https://github.com/Lumin-Science/LuminBench-Nano-ESMC.git
cd LuminBench-Nano-ESMC
bash runs/speedrun.sh
```

`runs/speedrun.sh` creates the locked environment, downloads and verifies the
required data shards, materializes on-disk training data, qualifies the CUDA
path, trains ESMC-300M, and verifies the final artifacts. Generated state
defaults to `.exps/`.

Copy the example environment file to customize the artifact root, data
revision, training samples, config, GPU count, wall time, or run name:

```bash
cp .env.example .env
```

The default is the original four-GPU, four-hour configuration. For a short
end-to-end smoke run:

```bash
TRAINING_SAMPLES=1 \
WALLTIME_SECONDS=60 \
RUN_NAME=smoke \
  bash runs/speedrun.sh
```

Each successful run records its resolved configuration, immutable data
revision, environment and corpus receipts, metrics, completion receipt, final
checkpoint, and checkpoint hash.

## Evaluate

The production training-and-evaluation entry point is:

```bash
bash runs/stage1_300m_4xa100_4h.sh
```

Full evaluation requires the separately prepared representation-probe and
contact datasets.
See [`EVALUATION.md`](EVALUATION.md) for dataset lineage, split roles, metrics,
and restartable execution details.
