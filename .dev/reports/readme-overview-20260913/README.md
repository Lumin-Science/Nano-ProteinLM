<div class="ai">

# README comparison figures — September 13, 2026

</div>

<div class="ai">

These figures summarize saved results; no model was trained or re-evaluated for this README update. The [data and source hashes](comparison-data.json) and [figure generator](../../scripts/build_readme_figures.py) make the plots reproducible. PNGs are used in the README; SVGs are available for export.

</div>

<div class="ai">

## Final model reference comparison

</div>

<div class="ai">

The ESMC-300M, ESMC-600M and ESMC-6B P@L values and 95% confidence intervals come from [Appendix A.1.4.1 of the ESMC paper](https://doi.org/10.64898/2026.06.03.729735), page 30. Our final 700k result comes from the [verified Stage 2 record](../nibi-setting3-stage2-b2048-300k-20260911/FINAL_VERIFIED.json). The paper and this repository follow the same published contact protocol, but their exact chain populations cannot be proven identical; this figure is a model reference comparison, not a controlled recipe comparison. Our local released-model point estimates remain documented in [EVALUATION.md](../../../docs/EVALUATION.md#released-esmc-checkpoint-pl).

</div>

<div class="ai">

The paper specifies 1M Stage 1 updates with batch 8,192 and context 512, then 500k Stage 2 updates with batch 2,048 and context 2,048 (pages 28–29). These imply nominal token budgets of 4.194304T and 2.097152T. Our 400k/300k update schedule with batch 2,048 implies 0.4194304T and 1.2582912T. These are maximum-context token budgets; the ESMC paper does not supply separately measured non-padding totals for the two stages.

</div>

<div class="ai">

FLOPs are computed separately for each stage with the paper’s page-30 formula: `3 × (2 × parameters + 4 × layers × context × width) × nominal tokens`, then summed. ESMC parameter counts are the paper’s Table S4 estimates: 333.0M, 575.0M and 6.35B; architecture dimensions come from Table S1. Our checkpoint has 170,559,856 parameters, 24 layers and width 768. The plotted values are consistent nominal-budget estimates, not measured hardware operations or a transcription of the paper’s Stage 1 scaling-curve FLOPs.

</div>

<div class="ai">

Our logs separately record 193,501,303,973 non-padding model tokens in Stage 1 and 181,403,984,859 in Stage 2, including BOS/EOS: 374,905,288,832 total. The logger’s `6 × parameters × model_tokens` estimate is 3.8366275246e20 FLOPs. This excludes the explicit attention term and uses actual non-padding tokens, so it should not be substituted for the nominal estimate in a cross-model compute ratio.

</div>

<div class="ai">

## Matched 100k-step verification

</div>

<div class="ai">

The two curves use all ten 10k-spaced endpoints per model from the [matched Nibi experiment](../nibi-paired-unique-b2048-100k-20260909/learning-curve.json): batch 2,048, 100k updates, four H100s per model, 204.8M distinct records and approximately 48.39B model tokens. Each endpoint was independently audited. The P@L bands are 5,000-replicate 95% chain-bootstrap intervals over 20,775 chains; validation loss uses 4,096 held-out sequences. There is one training seed per recipe, so no training-seed interval is implied. Earlier repeated-data runs and the later 400k/700k continuation have different budgets and are preserved in their original reports.

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
