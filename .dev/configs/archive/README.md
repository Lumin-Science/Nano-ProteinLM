# Historical training configurations

These 27 YAMLs and the scale-up manifest preserve completed and retired
experiments. They are retained for provenance and checkpoint compatibility;
new runs start with the [maintained recipes](../../../configs/README.md). Files were
moved without changing their contents. Paths inside receipts refer to their
original locations before the September 8, 2026 cleanup.

## 171M one-hour research: 13 presets

These use four L40S GPUs, global batch 256, context 512, and a 554-step warmup
followed by constant learning rates. Each paired recipe has two files that
differ only in training seed.

| Recipe | Configs | Parameters | Role |
|---|---|---:|---|
| Original AdamW | [original](../../../configs/esmc/esmc-171m.yaml) | 170,671,168 | Scientific baseline; seed 20260824 |
| Paired AdamW baseline | [42](program2/baseline_seed42.yaml), [43](program2/baseline_seed43.yaml) | 170,671,168 | Baseline for the published research history |
| R01 Muon | [42](program2/r01_muon_seed42.yaml), [43](program2/r01_muon_seed43.yaml) | 170,671,168 | Historical recipe; within the size bound |
| R04 + rank balance | [42](program2/r04_batchbalance_seed42.yaml), [43](program2/r04_batchbalance_seed43.yaml) | 170,671,168 | Historical recipe; within the size bound |
| R10 + square-root loss weights | [42](program2/r10_sqrtloss_seed42.yaml), [43](program2/r10_sqrtloss_seed43.yaml) | 170,671,168 | Historical recipe; within the size bound |
| R22 + FFN width 1536 | [42](program2/r22_ffn1536_seed42.yaml), [43](program2/r22_ffn1536_seed43.yaml) | 142,359,616 | Historical; below the current size bound |
| R29 + tied embeddings | [42](program2/r29_tied_seed42.yaml), [43](program2/r29_tied_seed43.yaml) | 142,310,464 | Historical; below the current size bound |

## H100 100k-step comparisons: 6 presets

All use four H100 GPUs, FA3, global batch 1,024, context 512, base LR 5e-4,
base WD 0.01, a 1,000-step warmup, and a 16-hour guard. Each run starts from
scratch. The four cumulative presets retain FFN width 2048 and about 171M
parameters; their R29 differs from the narrower one-hour R29 above.

| Recipe | Config | Role |
|---|---|---|
| AdamW | [default 100k](esmc-171m-default-h100-fa3-b1024-stage1-100k.yaml) | Scale-up baseline |
| R02, RoPE 20k | [R02 100k](esmc-171m-r02-h100-fa3-b1024-stage1-100k.yaml) | Completed reference recipe |
| R02, RoPE 10k | [setting 1](program2_h100_100k/r02_rope10k.yaml) | Starting recipe for the cumulative comparison |
| + rank balance | [setting 2](program2_h100_100k/r04_batchbalance.yaml) | Cumulative change |
| + square-root loss weights | [setting 3](program2_h100_100k/r10_sqrtloss.yaml) | Cumulative change |
| + tied embeddings | [setting 4](program2_h100_100k/r29_tied.yaml) | Cumulative change |

The [manifest](program2_h100_100k/manifest.json) records settings and config
hashes. See the [recipe differences](../../../docs/AUTORESEARCH_SCALEUP.md) and
[results](../../reports/fir-r02-rope10k-100k-20260906/README.md) for the comparison.

## Nibi eight-H100 comparisons: 2 presets

The [batch-2,048 AdamW baseline](esmc-171m-default-nibi-fa3-b2048-stage1-100k.yaml)
keeps the four-GPU baseline's 100,000 steps, LR 5e-4, WD 0.01, and warmup 1,000.
It uses 64 sequences/GPU and four accumulation steps on eight H100s, with a
24-hour training guard and checkpoint evaluation every 10,000 steps (full
validation MLM and contact P@L, with evaluation time recorded separately). See the
[launch record](../../reports/nibi-baseline-b2048-100k-eval10k-20260908/README.md) and
[four-GPU continuation instructions](../../../docs/checkpoint-resume.md).

The [batch-2,048 Setting 3](esmc-171m-setting3-nibi-fa3-b2048-stage1-100k.yaml)
uses the same budget, batch, base LR/WD, warmup and evaluation cadence. It retains
the winning Fir recipe's Muon group multipliers, RMSNorm, residual routing and
initialization, batch balance and sqrt loss. Its [queue and qualification
record](../../.dev/reports/nibi-setting3-b2048-100k-eval10k-20260908/README.md) places it
after the Nibi baseline and preserves the full final Muon/AdamW checkpoint.

## Older presets and reproduction references: 7 presets

| Config | Purpose |
|---|---|
| [171M R02, one hour](autoresearch_171m_4xl40s_1h.yaml) | Earlier contact-selected campaign; Muon, retained architecture, RoPE 20k |
| [171M AdamW, H100 12 hours](esmc-171m-original-h100-fa3-12h.yaml) | Earlier batch-256 FA3 preset with a wall-time budget |
| [171M AdamW, H100 10k steps](esmc-171m-default-h100-fa3-b1024-stage1-10k.yaml) | Cancelled pilot, superseded by the 100k comparison |
| [171M R02, H100 10k steps](esmc-171m-r02-h100-fa3-b1024-stage1-10k.yaml) | Cancelled pilot, superseded by the 100k comparison |
| [300M original](esmc-300m-original.yaml) | Original reproduction recipe; retained by the historical Stage-1 launcher |
| [300M one-hour autoresearch](autoresearch_300m_4xa100_1h.yaml) | Earlier contact-selected recipe with a final-20% cooldown |
| [300M current-best alias](esmc-300m-current-best.yaml) | Same configuration values as the preceding preset; compatibility alias |
