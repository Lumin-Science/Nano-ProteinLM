<div class="ai">

# nanop-best-171m-round2

</div>

<div class="ai">

Round 2 of our [sequential AutoResearch](../AUTORESEARCH_BASELINE.md#round-2-contact-pl) optimized contact P@L, starting from [nanop-best-171m-round1](nanop-best-171m-round1.md). It accepted two additions; the owner kept separate Q/K/V Muon updates and dropped query centering with RMS restoration. This recipe therefore contains every change from plain ESMC in the table below and is our current best recipe, found by GPT-6 with human effort across both rounds.

</div>

<div class="ai">

Configs: [search setting](../../configs/autoresearch/nanop-best-171m-round2.yaml) · [final evaluation](../../configs/test-100k/nanop-best-171m-round2.yaml). [LEADERBOARD.md](../LEADERBOARD.md) holds results under the current protocol.

</div>

<div class="ai">

## What differs from plain ESMC

</div>

<div class="ai">

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

</div>

<div class="ai">

Both recipes keep the tokenizer, context 512, 24 layers of width 768 with 12 heads, SwiGLU FFN width 2,048, RoPE base 10,000 and untied embeddings. The round-1 page explains each inherited component with formulas, worked examples and its H100 scale-up evidence: [Muon groups](nanop-best-171m-round1.md#hybrid-optimizer-and-its-actual-lrwd-settings), [RMSNorm, routing and initialization](nanop-best-171m-round1.md#rmsnorm-routing-and-initialization), [batch balance](nanop-best-171m-round1.md#3-batch-balance-equalize-work-across-gpus) and [sqrt loss](nanop-best-171m-round1.md#4-sqrt-loss-change-protein-weighting-not-the-validation-metric).

</div>

<div class="ai">

## Separate Q/K/V Muon updates

</div>

<div class="ai">

With `muon_split_qkv: true`, the fused QKV weight keeps its shape and checkpoint parameter name, while Muon receives three square, storage-sharing views. Momentum, orthogonalization and matrix-shape scaling are applied separately to Q, K and V. The forward pass, parameter count and DDP synchronization still use the fused tensor, so the change affects only the optimizer update. [Implementation](../../src/nanoprotein/split_qkv_muon.py) · [Promotion decision](../../.dev/reports/cck-contact-ablations-100k-20260919/DEFAULT_PROMOTION.md).

</div>

<div class="ai">

## 100k-step component study under the previous protocol

</div>

<div class="ai">

Each arm trained from scratch for **100,000 updates on four L40S GPUs** with FA2, seed 42, global batch 1,024, context 512, base LR 5e-4 and 1,000 warmup steps followed by constant LR. Each consumed 102,400,000 sequences and 24,196,983,520 non-padding model tokens from a 111-shard corpus, with no repeated source epochs. Evaluation used 4,096 MLM validation sequences and all 20,775 contact chains with 5,000 chain-bootstrap replicates. These settings differ from the current final evaluation, so the numbers explain the promotion decision rather than serve as leaderboard entries.

</div>

<div class="ai">

| Recipe | Separate Q/K/V Muon | Query centering + RMS restoration | Validation loss ↓ | P@L ↑ | P@L chain-bootstrap 95% CI | Training hours on 4 L40S |
|---|---|---|---:|---:|---:|---:|
| **nanop-best-171m-round2** | On | Off | **2.410035** | **33.449770%** | 33.214952–33.679334% | 35.4727 |
| Full round-2 search recipe | On | On | 2.413881 | 33.485122% | 33.251392–33.716149% | 37.7056 |
| No separate Q/K/V | Off | On | 2.416150 | 32.767462% | 32.535210–32.998056% | 37.2700 |

</div>

<div class="ai">

With query centering and RMS restoration enabled, separate Q/K/V updates improved P@L by **0.717660 percentage points**. Adding centering and RMS restoration to the split-Q/K/V recipe changed P@L by **+0.035352 points** and validation loss by **+0.003846**, so the owner kept the simpler recipe with the lower validation loss. There is no arm with both additions disabled, so the study cannot isolate the standalone Q/K/V gain over round 1. Each arm used one training seed; the intervals measure variation across contact chains, not training seeds. See the [comparison receipt](../../.dev/reports/cck-contact-ablations-100k-20260919/COMPARISON.json) and [study report](../../.dev/reports/cck-contact-ablations-100k-20260919/STATUS.md).

</div>
