<div class="ai">

# AutoResearch results

</div>

<div class="ai">

Recorded measurements are grouped by training budget and hardware. Each entry links to its recipe and experiment records. The [protocol](autoresearch.md) defines the search and final evaluation budgets, and [AUTORESEARCH_BASELINE.md](AUTORESEARCH_BASELINE.md) describes our sequential-search method and its historical comparisons.

</div>

<div class="ai">

## Current best: 4,096-protein reward re-evaluation

</div>

<div class="ai">

These are fresh MLM evaluations of the six existing best-recipe checkpoints, completed September 23, 2026. Each profile reports mean ± sample SD over seeds 42, 43 and 44, using 1,024 batches of four at context 512. Training was not repeated. The original 32-protein baseline/reference results remain in the sections below.

</div>

<div class="ai">

| Recipe and original training profile | MLM validation loss ↓, mean ± SD |
|---|---:|
| [Current best, Fir: 4 H100 × 20 minutes](leaderboard/CURRENT_DEFAULT_20260921.md#reward-re-evaluation-on-4096-proteins) | **2.667656 ± 0.000482** |
| [Current best, CCK: 4 L40S × 1 hour](leaderboard/CURRENT_DEFAULT_20260921.md#reward-re-evaluation-on-4096-proteins) | **2.660151 ± 0.001316** |

</div>

<div class="ai">

Only the MLM reward was recomputed. The [re-evaluation report](../.dev/reports/best-recipe-reward-4096-20260923/README.md) retains the existing full-contact scores with explicit reuse provenance, per-seed losses and timings. Checkpoint identities and sample settings were independently verified across both clusters.

</div>

<div class="ai">

## One-hour L40S reference

</div>

<div class="ai">

The recorded current-default reference uses one hour on four L40S GPUs per seed, FlashAttention-2, 32 MLM validation sequences and all 20,775 contact chains. This study repeats the recipe over seeds 42, 43 and 44. Values below are the mean and sample SD across the study's three seeds. P@L is shown as a percentage, with its SD in percentage points. Training GPU-hours exclude setup and evaluation.

</div>

<div class="ai">

| Recipe | Seeds | MLM validation loss ↓, mean ± SD | P@L ↑, mean ± SD |
|---|---|---:|---:|
| [human+ai baseline-09-26: current default](leaderboard/CURRENT_DEFAULT_20260921.md#search-budget-measurements) | 42, 43, 44 | **2.589492 ± 0.009195** | **11.771370% ± 0.256263 pp** |

</div>

<div class="ai">

## H100 reference measurement

</div>

<div class="ai">

The separately recorded H100 reference uses 20 minutes on four H100 GPUs per seed, FlashAttention-3, seeds 42, 43 and 44, 32 MLM validation sequences and all 20,775 contact chains. Its three-seed training allowance is 4 H100 GPU-hours. Keep its hardware, attention backend and duration attached to its scores when comparing it with the L40S reference. All three evaluations completed on September 22, 2026. Values are the mean and sample SD across training seeds; P@L SD is in percentage points. Actual training consumed 4.0010 H100 GPU-hours; final evaluation used another 0.8209 allocated GPU-hours, with setup excluded. The linked report includes per-seed values and independent verification.

</div>

<div class="ai">

| Recipe | Seeds | MLM validation loss ↓, mean ± SD | P@L ↑, mean ± SD |
|---|---|---:|---:|
| [human+ai baseline-09-26: current default](leaderboard/CURRENT_DEFAULT_20260921.md#search-budget-measurements) | 42, 43, 44 | **2.601592 ± 0.012375** | **11.207152% ± 0.253026 pp** |

</div>

<div class="ai">

<aitofix resolved>remove the 100k-step comparison 100k scaled up results of the esmc baseline our current best all the table Recipe just refer to the leaderboard folder corresponding remove Training GPU-hours	and Evidence, maybe. Fixed: Replaced the three-arm ablation table with the matched H100 baseline-versus-best results, linked each recipe through the leaderboard folder and removed the GPU-hours and Evidence columns from all overview tables. The current Q/K/V default's separate L40S result remains on its recipe page.</aitofix>

</div>

<div class="ai">

## 100k-step H100 scale-up

</div>

<div class="ai">

This matched comparison trained the ESMC-like AdamW baseline and our best recipe in that study for 100,000 updates on four H100s, with FA3, batch 1,024 and seed 20260824. Each run processed **24,200,224,761 model tokens**. Final evaluation used **4,096 MLM validation sequences** and **all 20,775 contact chains**. Each recipe has one training seed, so across-seed SD is unavailable.

</div>

<div class="ai">

| Recipe | MLM validation loss ↓ | P@L ↑ |
|---|---:|---:|
| [ESMC-like AdamW baseline](leaderboard/CURRENT_DEFAULT_20260921.md#1-the-complete-comparison) | 2.474360 | 26.504938% |
| [Our best recipe in this matched H100 study, before separate Q/K/V updates](leaderboard/CURRENT_DEFAULT_20260921.md#2-exactly-what-differs-from-the-baseline) | **2.418720** | **32.682480%** |

</div>

<div class="ai">

The [current default](leaderboard/CURRENT_DEFAULT_20260921.md#completed-100k-step-result) adds separate Q/K/V Muon updates and has a completed 100k-step L40S result. The H100 table above retains the earlier recipe actually measured against AdamW. Recipe pages contain configuration details, training costs, confidence intervals and source records. The scale-up losses use 4,096 validation sequences; the historical search-budget references above use 32. New task measurements use 4,096, so re-evaluate historical checkpoints on that sample before comparing their losses with new results.

</div>
