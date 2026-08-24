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
without accumulation. Experimental architecture and optimizer variants are
isolated under `dev/` or on the `auto-research` branch.

## Recorded source discrepancy

The paper text says residual updates are divided by `sqrt(n_layers)`. Biohub's
released implementation divides by `sqrt(n_layers / 36)`. This repository uses
the released source convention to preserve checkpoint geometry and records the
choice in every production config.
