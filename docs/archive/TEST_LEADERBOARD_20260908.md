# Archived six-recipe Test Leaderboard — September 8, 2026

Preserved from the [README at `0bba9f3`](https://github.com/Lumin-Science/Nano-Protein-LM/blob/0bba9f30efdadc9c53ef549d6236226e38c03b7c/README.md#scale-up-leaderboard) before its reorganization.
The original table, numbers, discussion and numbering are retained below; relative links are adjusted for this archive directory. This is a historical snapshot, not live training status.

In the current [Test Leaderboard](../../README.md#test-leaderboard), tied embeddings are **Setting 5**, corresponding to the historical **Setting 4** below. Current change 4 denotes FFN narrowing and is excluded from the fixed-size tests. The previous RoPE20k R02 row remains a historical reference.

## Original leaderboard

All six recipes completed **100,000 Stage-1 steps on four H100s**, with batch
**1,024**, **102.4M sampled sequences**, and the same full evaluations. Settings
1–4 add changes cumulatively; every run starts from scratch.

| Recipe | Validation loss ↓ | P@L ↑ | P@L 95% CI | Training time |
|---|---:|---:|---:|---:|
| ESMC-like baseline — AdamW | 2.47436 | 26.505% | 26.295–26.719% | 12h 01m |
| Previous R02 — RoPE20k | 2.43698 | 30.310% | 30.079–30.547% | 12h 57m |
| 1: R02 — RoPE10k | 2.43781 | 30.165% | 29.936–30.394% | 12h 58m |
| 2: + batch balance | 2.43872 | 30.715% | 30.487–30.948% | 12h 34m |
| **3: + sqrt loss** | **2.41872** | **32.682%** | **32.447–32.920%** | **12h 35m** |
| 4: + tied embeddings | 2.42304 | 31.884% | 31.651–32.123% | 12h 33m |

**Setting 3 is best on both metrics:** validation loss is **2.25% lower** and
P@L is **6.18 percentage points higher** than the AdamW baseline, with **4.71%
longer training**. It combines hybrid Muon/AdamW, parameter-free transformer
RMSNorm, learned residual/input routing, depth-scaled initialization, rank
balancing, and square-root masked-target weighting. RoPE remains 10k, FFN
width remains 2048, and its embeddings are untied.

The next scale-up comparison uses **eight H100s, batch 2,048, 100,000 steps,
and full evaluation every 10,000 steps** on Nibi. The [AdamW baseline
learning curve](../../.dev/reports/nibi-baseline-b2048-100k-eval10k-20260908/README.md)
is followed by [Setting 3](../../.dev/reports/nibi-setting3-b2048-100k-eval10k-20260908/README.md).
Both preserve their full final optimizer checkpoints for continuation. These
runs have twice the sequence exposure of the four-GPU table above.

All use base LR 5e-4, base WD 0.01, and a 1,000-step warmup; Muon retains R02's
per-group LR/WD multipliers. Validation uses the same 4,096 held-out sequences
and P@L uses the same 20,775 chains. Intervals are 5,000-resample chain-bootstrap
95% CIs, not training-seed uncertainty; each recipe has one training seed.
Training times exclude evaluation and are approximate to the minute.

See **[best recipe versus baseline: differences, figures, and worked examples](../BEST_RECIPE_VS_BASELINE.md)**,
the [complete results and adjacent comparisons](../../.dev/reports/fir-r02-rope10k-100k-20260906/README.md),
and the [historical baseline/R02 records](../../.dev/reports/fir-171m-100k-20260906/README.md).
The narrower-FFN change remains deferred in [TODO](../../TODO.md).

