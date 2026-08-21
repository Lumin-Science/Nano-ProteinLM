# nano-protein-embedding

A small, auditable training stack for one question: **how good a protein
embedding model can we train with a fixed amount of compute?**

The organization follows the useful part of nanochat's philosophy: one package,
one canonical speedrun, explicit immutable inputs, JSON receipts, and a short
path from raw data to a measured checkpoint. The default target is the released
ESMC 300M tier (332,997,184 parameters); ESMC 600M (575,036,992) is a first-class
configuration rather than a differently shaped approximation.

## The canonical run

On tmoss, from the existing four-A100 interactive allocation:

```bash
cd ~/workspace/nano-protein-embedding
uv sync --frozen
bash runs/speedrun.sh
```

`uv.lock` is the sole Python environment contract. The runner verifies the
locked Python/Torch/CUDA versions and executes every Python command through
`uv run --frozen`; `PYTHON_BIN` and ambient virtual environments are ignored.

`runs/speedrun.sh` prepares a reusable mmap corpus when needed, trains for 1,800
measured GPU-compute seconds, saves the stage boundary and final checkpoints,
and evaluates both. The default `EVAL_PROFILE=full` runs held-out MLM, a frozen
20-chain paper-aligned P@L diagnostic, and the current full P-CORE v0.2 suite.
Use `EVAL_PROFILE=standard` to omit P-CORE during kernel qualification.
Evaluation and checkpoint I/O are outside the 30-minute training clock.
Per-step compute time is max-reduced across DDP ranks, making both context-stage
and stopping decisions identical on every worker.

For a one-minute kernel/memory qualification:

```bash
DATA_ROOT=$PWD/data/processed/pilot-v1 WALLTIME_SECONDS=60 bash runs/qualify.sh
```

## What is exact, and what is a speedrun approximation?

Exact model contracts:

- 300M: 30 layers, width 960, 15 heads, 64-dimensional heads, 332,997,184 parameters.
- 600M: 36 layers, width 1,152, 18 heads, 64-dimensional heads, 575,036,992 parameters.
- Bias-free projections, full-width Q/K LayerNorm, non-interleaved RoPE, rounded
  8/3 SwiGLU, final LayerNorm, and the two-layer GELU language-model head.
- 15% mask-all MLM corruption, per-sequence mean cross-entropy, AdamW
  (0.9, 0.95), selective decay, and gradient clipping at 1.0.
- Stage mixtures: 36/11/54% then 63/6/31% for UniRef/MGnify/JGI (OMG/IMG here).
- Stage contexts: 512 then 2,048 tokens, with WSD decay throughout Stage 2.

Speedrun approximations:

- The transferred payload contains one sequence per 70%-identity cluster, not
  all member sequences. Sampling is therefore uniform over transferred cluster
  representatives; random member-within-cluster sampling cannot be recovered.
- The 30-minute run compresses the original 1M + 500k step schedule into a 2:1
  measured-time split.
- The paper describes the µP transfer rule but does not disclose its calibrated
  proxy LR/decay values. Configured values apply the disclosed rule to the
  explicit hypothesis `6e-4 / 0.01` at width 512, depth 16.
- Exact sequence hashes from P-CORE and P@L are excluded. Homology-level
  benchmark decontamination is still a release blocker.
- Prefix-padded batches are compacted to PyTorch's in-tree variable-length
  FlashAttention kernel. Short proteins neither pay quadratic padding compute
  nor require biased full-length-only sampling.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DATA.md](docs/DATA.md),
[docs/EVALUATION.md](docs/EVALUATION.md), [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md),
and [docs/ROADMAP.md](docs/ROADMAP.md).

## Repository map

```text
configs/          frozen 300M and 600M run contracts
nano_protein/     tokenizer, ESMC model, mmap data, trainer, evaluation
runs/             canonical speedrun and short qualification entrypoints
scripts/          thin command-line wrappers
tests/            fast contract and determinism checks
docs/             scientific and operational decisions
report/           NeurIPS-style experiment report
```

## Publication layout

- Source and issues: `github.com/Lumin-Science/nano-protein-embedding`
- Data manifests and released checkpoints: `huggingface.co/LuminScience`
- Large raw corpora stay in controlled storage; they are not committed to Git or
  republished without a source-license audit.

This repository is developed locally first. No GitHub or Hugging Face remote is
created by the speedrun.
