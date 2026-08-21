# Operational qualification

All trials used four A100 80GB PCIe GPUs, the same seed/data manifest, exact
ESMC-300M geometry, BF16, and 60 measured training seconds. These are systems
checks, not model-quality comparisons.

| Trial | Stage 1 / 2 microbatch | Model tokens | Sequences | Outcome |
|---|---:|---:|---:|---|
| Per-layer attention packing | 8 / 1 | 4,259,840 shaped; 1,662,828 residues | not recorded | rejected: pad rows still crossed every MLP |
| Whole-model packing baseline | 8 / 1 | 2,096,285 | 8,128 | passed |
| Larger microbatch | 32 / 8 | 2,821,267 | 11,072 | passed, +35% tokens |
| Selected | 64 / 16 | 4,521,293 | 17,408 | passed, 2.16x baseline tokens |
| `torch.compile(dynamic=True)` | 32 / 8 | 0 | 0 | rejected before step 1 by Torch AOT metadata failure |

The selected point reached up to 101,752 real tokens/s and 16.3% 6N MFU in
Stage 1; Stage 2 reached up to 46,097 tokens/s and 7.4% 6N MFU. MFU uses the
dense 312 BF16 TFLOP/s peak per A100 and therefore excludes attention's
quadratic FLOPs; it is an explicit, conservative operational proxy.

`torch.compile` failed inside AOTAutograd with `AttributeError: 'float' object
has no attribute 'meta'` on the dynamic packed graph. Eager execution is frozen
for the canonical pilot. The successful experiment configs remain under
`configs/experiments/`; the known-failing compile config is intentionally not
shipped as a runnable contract.

The exact 575,036,992-parameter ESMC-600M contract was separately qualified
with its conservative 4 / 1 microbatches and gradient accumulation of 8. It
completed both context stages in 61.3 measured seconds, processing 794,061
model tokens across 3,040 sequences without an out-of-memory error. This is a
runnability check only; the 300M tier remains the canonical speedrun target.
