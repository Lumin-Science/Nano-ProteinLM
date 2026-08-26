# Reproduction and speedrun

## Environment contract

The project-level prerequisite is `uv >=0.11.31,<0.12`. A training host also
needs a supported NVIDIA driver and four visible BF16-capable GPUs. Python,
Torch, CUDA user-space libraries, and every Python dependency are resolved from
`.python-version`, `pyproject.toml`, and `uv.lock`; no Conda environment or
manual `pip` installation is part of the supported path.

From a fresh clone:

```bash
git clone https://github.com/Lumin-Science/LuminBench-Nano-ESMC.git
cd LuminBench-Nano-ESMC
bash speedrun.sh
```

`speedrun.sh` performs four fail-closed phases:

1. create the locked uv environment;
2. download a deterministic shard prefix from one immutable Hugging Face
   revision, verify every selected file and sequence, and materialize mmap data;
3. qualify Python, Torch, CUDA, GPU count, BF16, variable-length FlashAttention,
   and a packed forward/backward pass; and
4. train ESMC-300M and verify the required completion artifacts.

All generated state defaults to `.exps/` in the clone. This includes the uv and
Hugging Face caches, virtual environment, prepared data, environment receipt,
metrics, run contract, completion receipt, and checkpoint. Existing run roots
are never overwritten.

## Configuration

Copy the tracked example to create a local configuration:

```bash
cp .env.example .env
```

The main controls are `ARTIFACT_ROOT`, `DATA_REVISION`, `TRAINING_SAMPLES`,
`CONFIG`, `NUM_GPUS`, `WALLTIME_SECONDS`, `RUN_NAME`, `DATA_ROOT`, and
`OUTPUT_ROOT`. Exported values override the defaults preserved by the example
file. The default data revision is the verified public commit
`bd38448d50d8f426d7b9bd4410b53159ea001259`.

The original four-GPU, four-hour configuration is the default. A short
full-path smoke run is:

```bash
TRAINING_SAMPLES=1 \
WALLTIME_SECONDS=60 \
RUN_NAME=smoke \
  bash speedrun.sh
```

Whole-shard verification means this minimum request still selects one train
shard from each source plus all three validation shards: 3,339,505 records and
577,305,913 compressed bytes.

## Fresh-environment pilot

The smoke command was executed from an empty source copy on Killarney node
`kn082` using only a standalone `uv 0.11.31` bootstrap and its existing
four-L40S allocation. No virtual environment, dependency cache, data, or output was
copied into the test directory.

| Check | Observed result |
|---|---|
| Environment | Python 3.11.4; Torch 2.13.0+cu126; CUDA 12.6 |
| GPUs | Four NVIDIA L40S; BF16 and packed FlashAttention forward/backward passed |
| Data | Six published files; every file and sequence hash verified; zero declared train/evaluation and train/validation intersections |
| Model | 332,997,184 parameters |
| Training | 97 optimizer steps; 60.3805 seconds; 5,875,454 model tokens |
| Peak allocated CUDA memory | 35,691,617,280 bytes |
| Final checkpoint SHA-256 | `c7aae89f2c059d17ecf057a58fb4b304f1119b5c9c3e62bab86255805d502e7f` |

Torch emitted non-fatal allocator-probe warnings while searching for large
blocks on the L40S devices. No rank failed, training continued through the
wall-time stop, all required receipts were written, and an independent final
checkpoint hash matched the completion receipt.

## Required acceptance artifacts

A completed run is reproducible only when all of the following agree:

- the Git commit and dirty-state record;
- `uv.lock` and its SHA-256;
- immutable Hugging Face revision and download-plan digest;
- prepared-corpus manifest and `CORPUS_VERIFICATION.json`;
- config path and digest;
- `ENVIRONMENT.json` and `run_contract.json`;
- `metrics.jsonl` and `TRAINING_COMPLETE.json`; and
- `checkpoint-final.pt` and its recorded SHA-256.

The speedrun validates existence and non-emptiness; the data loader and trainer
validate the content-bound contracts before training. Release acceptance also
requires a fresh controlled rebuild and repeated evaluation as specified in
[`TASK.md`](TASK.md) and [`RELEASE.md`](RELEASE.md).

## Full evaluation

The training-only speedrun stops after checkpoint verification. Full evaluation
requires the separately reconstructed P-CORE and contact payloads:

```bash
bash runs/stage1_300m_4xa100_4h.sh
```

Dataset sources, publications, split roles, and access expectations are in
[`EVALUATION_DATASETS.md`](EVALUATION_DATASETS.md). Metric definitions and
restartable execution are in [`EVALUATION.md`](EVALUATION.md).
