# Nibi: ESMC-like AdamW baseline, batch 2,048

Status: preparing the authorized run; GPU qualification and full launch are pending.

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

Configuration: [`esmc-171m-default-nibi-fa3-b2048-stage1-100k.yaml`](../../configs/esmc-171m-default-nibi-fa3-b2048-stage1-100k.yaml).
Continuation instructions: [`checkpoint-resume.md`](../../docs/checkpoint-resume.md).

Scratch root: `/scratch/muchenli/Nano-Protein-LM-nibi-b2048-100k-20260908`.
Frozen source checkout: the scratch root with `-run` appended.
Persistent final checkpoint directory:
`/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-baseline-b2048-100k-20260908`.

Qualification will exercise a 200-step eight-GPU run with a rolling checkpoint at
step 100, same-layout continuation from 100 to 200, and four-GPU continuation from
200 to 220 with accumulation doubled. The production run begins only after the
checkpoint, optimizer state, finite metrics, FA3, and global-batch checks pass.
The shortened trial uses 50 warmup steps; production retains 1,000.

Final evaluation uses the same frozen 4,096-sequence validation MLM and full
20,775-chain contact P@L protocol as the Fir comparison. The 95% interval is a
5,000-replicate chain bootstrap, not training-seed variability.
