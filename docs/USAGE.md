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

The measured one-hour AutoResearch incumbent is available but deliberately
off by default. Opt in explicitly:

```bash
CONFIG="$PWD/configs/autoresearch_300m_4xa100_1h.yaml" \
WALLTIME_SECONDS=3600 \
RUN_NAME=autoresearch-incumbent-1h \
  bash runs/speedrun.sh
```

`configs/esmc-300m-current-best.yaml` is the stable public alias for this exact
incumbent. Both retain the winning model/training recipe; neither changes the
default original-ESMC speedrun.

The validated ESMC-171M R02 preset is also available:

```bash
CONFIG="$PWD/configs/autoresearch_171m_4xl40s_1h.yaml" \
NUM_GPUS=4 \
WALLTIME_SECONDS=3600 \
RUN_NAME=autoresearch-171m-r02-1h \
  bash runs/speedrun.sh
```

Its published results use four L40S GPUs. The preset uses Muon, RoPE base
20,000, and warmup followed by constant LR. Set `WALLTIME_SECONDS` explicitly:
the speedrun's four-hour default overrides the duration in the YAML.

Each successful run records its resolved configuration, immutable data
revision, environment and corpus receipts, metrics, completion receipt, final
checkpoint, and checkpoint hash.

### H100 FlashAttention-3 baseline

`configs/esmc-171m-original-h100-fa3-12h.yaml` keeps the original ESMC-171M
recipe and selects `attention_backend: flash3` with a 43,200-second budget.
It requires Hopper GPUs and Linux/CUDA 12.6 or 13.0 PyTorch. The dependency
lock uses CUDA 12.6; CUDA 13.0 measurements must identify their runtime explicitly.
The first use downloads the matching pinned FA3 build from
`kernels-community/flash-attn3` at revision
`e29f138fc363b396e5d2706c8a5f6fa7d36f41e0`; packaged files are checked
against SHA-256 hashes before importing the kernel.

Qualify forward/backward numerics and verify FA3 profiler events before training:

```bash
uv run --frozen python scripts/check_environment.py \
  --require-gpus 4 --attention-backend flash3 --output /path/to/ENVIRONMENT.json
```

Pass the H100 config and `WALLTIME_SECONDS=43200` to the training entry point.
Training receipts identify the actual attention backend and pinned kernel build.

## Evaluate

The production training-and-evaluation entry point is:

```bash
bash runs/stage1_300m_4xa100_4h.sh
```

Full evaluation requires the separately prepared representation-probe and
contact datasets. Contact P@L uses the one-probe, sparse-scoring fast path by
default; a receipt-bound static scoring cache can be enabled with
`CONTACT_SCORING_CACHE_ROOT`.
See [`EVALUATION.md`](EVALUATION.md) for dataset lineage, split roles, metrics,
and restartable execution details.

### ESMC-171M default at global batch 1,024

`configs/esmc-171m-default-h100-fa3-b1024-stage1-100k.yaml` prepares the
Stage-1 default with peak LR `5e-4`, 1,000 warmup steps, and 100,000 optimizer
steps. On four H100s it uses microbatch 64 with accumulation 4 and FA3/BF16.
The 16-hour wall-time guard allows the projected roughly 13-hour step budget
to finish; overriding it to 43,200 seconds can stop before 100k steps.
See [the three-way recipe comparison](171M_RECIPES.md) for the retained R02
settings and the paper reference. This preset does not change an active run.

The matched R02 comparison preset is
`configs/esmc-171m-r02-h100-fa3-b1024-stage1-100k.yaml`. It shares the default's
base LR/WD, warmup, batch, seed, context, mixture, and 100k-step budget, while
retaining the R02 architecture and Muon group multipliers. Use the same data
root and evaluation arguments for both; see the linked comparison contract.
The original one-hour R02 preset remains the historical record.
