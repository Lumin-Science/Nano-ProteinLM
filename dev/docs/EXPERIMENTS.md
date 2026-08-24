# Completed production experiments

These historical training results used the then-verified
`stage1-300m-production-v1` corpus. Their data manifest, corpus-verification, uv
lock, config, code revision, checkpoint, and evaluation digests are recorded in
the result receipts. This corpus predates the Q9 screen and is archived for
provenance only; new runs must use an immutable `full-open-v2` shard prefix.

| Run | GPUs | Target steps | Completed steps | Warmup | P@L | Status |
|---|---:|---:|---:|---:|---:|---|
| Stage-1 300M 4h | 4×A100 80GB | 21,000 | 20,971 | 2,100 | 0.101252 | wall-time stop |
| Stage-1 300M 16h | 4×A100 80GB | 84,000 | 84,000 | 8,400 | 0.163566 | step complete |

The sixteen-hour checkpoint improves full 20,775-chain P@L by 0.062315 over
the four-hour checkpoint. This is a paired evaluation-unit comparison between
two single-seed runs; it does not estimate training-seed uncertainty.

Detailed receipts:

- [`results/stage1-300m-4xa100-4h/`](../results/stage1-300m-4xa100-4h/)
- [`results/stage1-300m-4xa100-16h/`](../results/stage1-300m-4xa100-16h/)

## Active AutoResearch contract

Architecture and optimizer experiments on the `auto-research` branch use four
GPUs and exactly 7,200 seconds of synchronized training-loop wall time. Before
each run, a steady-state smoke test estimates the step budget. Frozen P@L is the
only model-selection score. See [`program.md`](../program.md).

The production corpus and contact evaluator are immutable during this loop.
Every candidate still passes the trainer's mandatory homology-decontamination
gate; a configuration cannot opt out.
