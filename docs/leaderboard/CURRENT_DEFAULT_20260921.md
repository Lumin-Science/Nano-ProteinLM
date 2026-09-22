<div class="ai">

# Current default: separate Q/K/V Muon updates

</div>

<div class="ai">

The default selected on September 21, 2026 adds separate Q/K/V Muon updates to the earlier improved recipe. Query centering and RMS restoration are disabled. The remaining architecture, normalization, residual routing, initialization, batch balancing, sqrt-mask-count loss and optimizer-group settings are unchanged. [Current preset](../../configs/default.yaml) · [Promotion decision and component evidence](../../.dev/reports/cck-contact-ablations-100k-20260919/DEFAULT_PROMOTION.md).

</div>

<div class="ai">

## Search-budget measurements

</div>

<div class="ai">

Both studies repeated the same frozen recipe from source revision `43f2996729e6b41a998de92674340217afacfd5b` across seeds 42, 43 and 44. They used the original seven-shard corpus, global batch 1,024, context 512 and 554 warmup steps followed by constant learning rate. Final evaluation used 32 MLM validation sequences and all 20,775 contact chains. Values are mean ± sample SD across training seeds; P@L SD is in percentage points.

</div>

<div class="ai">

| Hardware and training time per seed | MLM validation loss ↓ | P@L ↑ | Training GPU-hours | Run records |
|---|---:|---:|---:|---|
| 4 L40S, 1 hour, FA2 | 2.589492 ± 0.009195 | 11.771370% ± 0.256263 pp | 12.001692 | [CCK study](../../.dev/reports/cck-human-ai-baseline-09-26-20260921/README.md) |
| 4 H100, 20 minutes, FA3 | 2.601592 ± 0.012375 | 11.207152% ± 0.253026 pp | 4.000974 | [Fir study](../../.dev/reports/fir-human-ai-baseline-09-26-20260921/README.md) |

</div>

<div class="ai">

Training GPU-hours exclude setup and evaluation. The studies' final evaluations used another 1.350473 allocated L40S GPU-hours and 0.820892 allocated H100 GPU-hours, respectively. The linked reports retain per-seed metrics, configuration details and completion checks.

</div>

<div class="ai">

## Completed 100k-step result

</div>

<div class="ai">

The current recipe completed 100,000 updates on four L40S GPUs with FA2, seed 42 and global batch 1,024, processing 24,196,983,520 non-padding model tokens without source resampling. Its final checkpoint achieved **MLM validation loss 2.410035** and **contact P@L 33.449770%**, using 4,096 validation sequences and all 20,775 contact chains. Training consumed 141.8907 L40S GPU-hours. This study has one training seed, so across-seed SD is unavailable.

</div>

<div class="ai">

The [component comparison](../AUTORESEARCH_BASELINE.md#l40s-100k-step-component-ablations) retains all three recipes and their uncertainty estimates. The matched [H100 baseline-versus-best comparison](BEST_RECIPE_22_09_26.md#1-the-complete-comparison) predates the Q/K/V update. Its H100 results belong to the earlier recipe; this L40S result does not provide a matched H100 comparison against AdamW.

</div>
