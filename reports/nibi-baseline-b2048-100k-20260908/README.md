# Nibi: ESMC-like AdamW baseline, batch 2,048

Status: **cancelled at the user's request** after approximately 500 steps on
September 8, 2026, to add evaluation every 10,000 steps and restart from scratch.
Only production step **12162637.7** was cancelled; the parent allocation remains
running. The [replacement run](../nibi-baseline-b2048-100k-eval10k-20260908/README.md)
uses the same recipe with periodic evaluation. Early throughput of this initial
run was 0.437 seconds/step, with finite losses and gradients.

The frozen training commit is `b653f7a54e286a756150b54eb6111cb236653bbc`.
The exact 83 Fir runtime package versions were reproduced on Nibi, including
PyTorch 2.13.0/CUDA 13.0. Kernel hashes, BF16, packed FA3 forward/backward, and
agreement with the FA2 numerical reference passed. The hourly monitor covers
this run and the still-running Fir tied-embedding experiment.

The user selected the **ESMC-like AdamW baseline**, not cumulative Setting 2. This
run starts from scratch and doubles the previous baseline's global batch while
retaining its model, learning rate, weight decay, warmup, data, and 100,000-step
Stage-1 endpoint. It is the project's ESMC-like recipe, not an exact reproduction
of an undisclosed paper optimizer calibration.

| Setting | Value |
|---|---|
| Parameters | 170,671,168 |
| GPUs | 8 × H100 80 GB, Nibi g27, allocation 12162637 |
| Global batch | 2,048 = 64/GPU × 8 GPUs × 4 accumulation steps |
| Steps / sequences | 100,000 / 204,800,000 |
| Context | 512; packed nonpadding tokens |
| Optimizer | AdamW, betas (0.9, 0.95), gradient clip 1.0 |
| Learning rate / weight decay | 5e-4 / 1e-2 |
| Warmup | 1,000 steps, followed by constant Stage-1 LR |
| Attention / precision | FlashAttention-3 / BF16 autocast, FP32 master parameters |
| Architecture | Original LayerNorm, RoPE base 10,000, FFN 2,048 |
| Loss / batching | Sequence-mean MLM, no cross-rank batch balancing |
| Corpus | Same public, homology-filtered corpus as the Fir comparison |
| Checkpoints | Rolling latest every 10,000 steps; final retained on project storage |

Configuration: [`esmc-171m-default-nibi-fa3-b2048-stage1-100k.yaml`](../../configs/archive/esmc-171m-default-nibi-fa3-b2048-stage1-100k.yaml).
Continuation instructions: [`checkpoint-resume.md`](../../docs/checkpoint-resume.md).

Scratch root: `/scratch/muchenli/Nano-Protein-LM-nibi-b2048-100k-20260908`.
Frozen source checkout: the scratch root with `-run` appended.
Persistent final checkpoint directory:
`/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-baseline-b2048-100k-20260908`.

Qualification **passed** before production launch:

| Test | Steps | Result |
|---|---:|---|
| Eight GPUs, full model and batch | 0 → 200 | Finite weights/gradients and all 248 AdamW parameter states; 0.434 s/step |
| Eight-GPU checkpoint resume | 100 → 200 | Model, optimizer, counters, sampler and RNG restored; data stream reproduced exactly |
| Four-GPU continuation, accumulation 8 | 200 → 220 | Batch 2,048 preserved; all AdamW states reached step 220; 0.846 s/step |
| Direct GPU restoration audit | Saved step 200 | Every model and AdamW tensor loaded exactly |
| Reloaded four-GPU checkpoint MLM | 32 sequences | Finite loss 2.75440; technical smoke check only |

The shortened trials used 50 warmup steps; production retains 1,000. Subsequent
training is not guaranteed bitwise deterministic: after 100 more updates, the
same-layout resumed weights differed from uninterrupted training by 2.61% in
relative L2 norm (maximum absolute difference 0.01166), despite exact state loading
and identical sampling/RNG endpoints. Final logged losses were 2.77599 and
2.77711 respectively. These are restoration/stability checks, not production
quality results or a training-seed uncertainty estimate.

Receipts are in [`qualification/`](qualification/) and
[`RUNTIME_REPRODUCED.json`](RUNTIME_REPRODUCED.json). The production configuration,
source contract and initial metrics are in [`full/`](full/). Checkpoint binaries
stay on Nibi. This cancelled attempt stopped before its first production
checkpoint; it has no final production checkpoint. The replacement launch retains
the final-checkpoint preservation requirement.

Final evaluation uses the same frozen 4,096-sequence validation MLM and full
20,775-chain contact P@L protocol as the Fir comparison. The 95% interval is a
5,000-replicate chain bootstrap, not training-seed variability.
