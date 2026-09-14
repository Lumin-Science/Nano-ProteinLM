<div class="ai">

# README comparison figures — September 13, 2026

</div>

<div class="ai">

The opening figure summarizes saved training results. The final comparison also includes a new frozen contact evaluation of ESM-2 150M; no model was pretrained for this README update. The [data and source hashes](comparison-data.json) and [figure generator](../../scripts/build_readme_figures.py) make the plots reproducible. PNGs are used in the README; SVGs are available for export.

</div>

<div class="ai">

## Matched 171M recipe comparison

</div>

<div class="ai">

The opening figure compares our local ESMC-like 171M AdamW reproduction (orange, “ESMC 171M · AdamW”) with the improved 171M recipe (green, “Auto Research Best · Sep 13, 2026”). The date identifies the documented best recipe; the matched 100k runs themselves finished earlier and are preserved under their original run dates. Neither label denotes a released Biohub 171M checkpoint.

</div>

<div class="ai">

The left panel puts all four evaluation curves on one shared step axis: two contact P@L curves on the left vertical axis and two MLM validation-loss curves on the right. It uses all ten 10k-spaced endpoints per model from the [matched Nibi experiment](../nibi-paired-unique-b2048-100k-20260909/learning-curve.json): batch 2,048, 100k updates, four H100s per model, 204.8M distinct records and approximately 48.39B model tokens. Every endpoint was independently audited. P@L bands are 5,000-replicate 95% chain-bootstrap intervals over 20,775 chains; validation loss uses 4,096 held-out sequences. One training seed per recipe means these are chain intervals, not seed intervals.

</div>

<div class="ai">

The right panel reads the original compressed [AdamW logs](../nibi-paired-unique-b2048-100k-20260909/full/baseline/metrics.jsonl.gz) and [Auto Research logs](../nibi-paired-unique-b2048-100k-20260909/full/setting3/metrics.jsonl.gz): 10,001 records each, at step 1 and every tenth update through 100k. Thin translucent traces show raw `loss`; thick lines show a trailing average of 100 records, spanning 1,000 updates after warmup. The main training panel shows steps 1,000–100,000; an inset preserves the full loss range for steps 1–1,000. [Extracted summaries](training-curve-summary.json) record endpoints and post-warmup ranges.

</div>

<div class="ai">

The common `loss` field is rank 0’s sequence-mean masked cross entropy, averaged across local accumulation microbatches. It is a training diagnostic, not a globally reduced held-out metric. The improved recipe also logs `objective_loss`, which uses square-root mask-count weights; that different objective is deliberately excluded from this comparison. Training noise and the warmup transient remain visible. Earlier repeated-data runs and the later 400k/700k continuation use different budgets and remain in their original reports.

</div>

<div class="ai">

## Final model reference comparison

</div>

<div class="ai">

The final reference block includes ESMC-300M, ESMC-600M, ESM-2 150M and our completed 171M model. All P@L values use our full 20,775-chain split. The ESMC means are 0.5386739700782069 and 0.5803125316548938; the [recovery receipt](released-local-split.json) records their full-report hashes, model revisions and source archive hashes. They come specifically from `source.full_point_estimate` in archived diagnostic receipts and agree with the rounded full-population values in [EVALUATION.md](../../../docs/EVALUATION.md#released-esmc-checkpoint-pl). The 1,024-chain diagnostic means and intervals are not used. Full-population confidence intervals were not recovered for the ESMC references, so their table cells are blank and their plot markers have no error bars. The final 171M point and its 95% interval come from the [verified Stage 2 record](../nibi-setting3-stage2-b2048-300k-20260911/FINAL_VERIFIED.json).

</div>

<div class="ai">

ESM-2 150M has a new full-split evaluation with the unchanged fitted-probe and scoring APIs, including a 5,000-resample chain confidence interval. The [verified evaluation receipt](../released-150m-contact-20260913/esm2/RESULT_VERIFIED.json) documents its checkpoint identity and result. The figure shows a single horizontal P@L chart with available 95% confidence intervals for all four models.

</div>

<div class="ai">

The [ESMC paper](https://doi.org/10.64898/2026.06.03.729735) specifies 1M Stage 1 updates with batch 8,192 and context 512, then 500k Stage 2 updates with batch 2,048 and context 2,048 (pages 28–29). These imply nominal token budgets of 4.194304T and 2.097152T. Our 400k/300k update schedule with batch 2,048 implies 0.4194304T and 1.2582912T. These are maximum-context token budgets; the paper does not supply separately measured non-padding totals for the two stages. Token exposures are not counts of unique dataset proteins.

</div>

<div class="ai">

FLOPs are computed separately for each stage with the paper’s page-30 formula: `3 × (2 × parameters + 4 × layers × context × width) × nominal tokens`, then summed. ESMC-300M and ESMC-600M parameter counts are the paper’s Table S4 estimates, 333.0M and 575.0M; architecture dimensions come from Table S1. Our checkpoint has 170,559,856 parameters, 24 layers and width 768. These table values are nominal-budget estimates, not measured hardware operations or a transcription of the paper’s Stage 1 scaling-curve FLOPs.

</div>

<div class="ai">

Our logs separately record 193,501,303,973 non-padding model tokens in Stage 1 and 181,403,984,859 in Stage 2, including BOS/EOS: 374,905,288,832 total. The logger’s `6 × parameters × model_tokens` estimate is 3.8366275246e20 FLOPs. This excludes the explicit attention term and uses actual non-padding tokens, so it should not replace the nominal estimate in a cross-model compute ratio.

</div>

<div class="ai">

## Config relocation

</div>

<div class="ai">

The four Nibi presets moved from `configs/` to [`.dev/configs/nibi/`](../../configs/nibi/) without changing file contents. [The relocation receipt](config-moves.json) records their SHA-256 hashes. Current documentation and local recipe readers use the new paths. Historical commit-qualified paths, run-contract fields and archived remote launch receipts retain their original meaning; the two launch scripts accept both the relocated preset and its original location in a frozen checkout.

</div>

<div class="ai">

## Rebuild

</div>

<div class="ai">

Run with Python, NumPy and Matplotlib 3.10.8 in a plotting environment; training dependencies do not need to change.

</div>

<div class="ai">

```bash
python .dev/scripts/build_readme_figures.py
```

</div>
