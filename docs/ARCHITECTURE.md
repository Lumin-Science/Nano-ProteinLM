# Architecture contract

The implementation is frozen against Biohub `esm` commit
`9b9078cadf4e08bd02e9715ba99313fc5127379a` and the June 2026 ESMC preprint.

| Tier | Layers | Width | Heads | FFN width | Parameters |
|---|---:|---:|---:|---:|---:|
| ESMC 300M | 30 | 960 | 15 | 2,560 | 332,997,184 |
| ESMC 600M | 36 | 1,152 | 18 | 3,072 | 575,036,992 |

Every head is 64 dimensions. Q and K are normalized across the full projected
width before reshaping into heads. RoPE uses the GPT-NeoX/non-interleaved half
rotation. Attention uses the uv-locked Torch 2.13 CUDA 12.6 build. Dense batches
use scaled-dot-product FlashAttention; prefix-padded batches are compacted to
`aten::_flash_attention_forward` with cumulative sequence lengths. This is the
same in-tree varlen kernel behind jagged SDPA. Requesting `flash` fails rather
than silently falling back, and qualification exercises forward and backward.
The optimized CUDA path packs once before the transformer and scatters once
after the language-model head, so LayerNorm and SwiGLU also skip pad rows.

The frozen production ESMC-300M baseline uses 64 sequences/GPU at context 512
without accumulation. `ESMCConfig` defaults remain the original released
architecture: learned residual routing is off and transformer normalization is
LayerNorm. `configs/esmc_300m_stage1_4xa100_4h.yaml` preserves that setting.

The separately named `configs/esmc_300m_stage1_4xa100_4h_best.yaml` is the
current opt-in P@L-selected setting. AutoResearch round 2 added one learned
residual-stream scalar and one learned input-embedding scalar per layer; round
6 replaced transformer attention, Q/K, FFN, and final norms with parameter-free
RMSNorm while retaining the LayerNorm MLM head. Their two-hour frozen full-chain
P@L scores were 0.0973519991 and 0.0987442911 respectively, versus 0.0958063581
for the original baseline. The combined 300M variant has 332,823,484 parameters.
Rejected or not-yet-qualified variants remain isolated under `dev/` or on the
`auto-research` branch.

## Recorded source discrepancy

The paper text says residual updates are divided by `sqrt(n_layers)`. Biohub's
released implementation divides by `sqrt(n_layers / 36)`. This repository uses
the released source convention to preserve checkpoint geometry and records the
choice in every production config.
