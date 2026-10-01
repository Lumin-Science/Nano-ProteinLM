# nanop-best-171m-round2

Round 2 adds separate Q/K/V Muon updates to [round 1](nanop-best-171m-round1.md). It retains the same model architecture and training-loss definition.

Configs: [search](../../configs/autoresearch/nanop-best-171m-round2.yaml) · [scale-up](../../configs/test-100k/nanop-best-171m-round2.yaml). The [leaderboard](../LEADERBOARD.md) reports the three-seed search comparison and five-attempt scale-up metrics defined by the [evaluation contract](../EVALUATION.md).

## What differs from plain ESMC

| Component | Plain ESMC reference | nanop-best-171m-round2 | Added in |
|---|---|---|---|
| Transformer matrix optimizer | AdamW | Muon for attention and FFN matrices | Round 1 |
| Muon group LR / WD | — | Attention 0.9× and FFN 0.75× base LR; WD 0.75× base WD | Round 1 |
| Q/K/V Muon update | — | Separate Q, K and V updates of the fused QKV matrix | Round 2 |
| Embedding / MLM-head optimizer | AdamW | AdamW retained | — |
| Transformer normalization | Affine LayerNorm | Parameter-free RMSNorm | Round 1 |
| Stream entering each block | Current hidden stream | Learned mixture of current stream and original token embeddings | Round 1 |
| Attention-output / FFN-down initialization | Normal, standard deviation 0.02 | Normal, standard deviation `0.02 / sqrt(2 × 24)` ≈ 0.002887 | Round 1 |
| Cross-GPU batch assignment | Each rank keeps its sampled examples | Redistribute already-masked examples to balance token counts | Round 1 |
| Training loss | Equal mean weight per protein | Protein mean weighted by sqrt(masked-target count) | Round 1 |
| Validation loss | Equal mean weight per protein | Same evaluation | — |
| Trainable parameters | 170,671,168 | 170,559,856 | Round 1 |

Both recipes keep the tokenizer, context 512, 24 layers of width 768 with 12 heads, SwiGLU FFN width 2,048, RoPE base 10,000 and untied embeddings. The round-1 page explains each inherited component with formulas and worked examples: [Muon groups](nanop-best-171m-round1.md#hybrid-optimizer-and-its-actual-lrwd-settings), [RMSNorm, routing and initialization](nanop-best-171m-round1.md#rmsnorm-routing-and-initialization), [batch balance](nanop-best-171m-round1.md#batch-balance-equalize-work-across-gpus) and [sqrt loss](nanop-best-171m-round1.md#sqrt-weighted-training-loss).

## Separate Q/K/V Muon updates

With `muon_split_qkv: true`, the fused QKV weight keeps its shape and checkpoint parameter name, while Muon receives three square, storage-sharing views. Momentum, orthogonalization and matrix-shape scaling are applied separately to Q, K and V. The forward pass, parameter count and DDP synchronization still use the fused tensor, so the change affects only the optimizer update. [Implementation](../../src/nanoprotein/split_qkv_muon.py).
