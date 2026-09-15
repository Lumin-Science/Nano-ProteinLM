<div class="ai">

# Static README figures — September 14, 2026

</div>

<div class="ai">

These figures use Matplotlib 3.10.8 and NumPy 2.4.6, with standard blue/orange colors on white backgrounds. Data plots contain titles, axes and legends; experimental context stays in the repository README. The data-preparation and AutoResearch diagrams are static Matplotlib figures. PNGs are embedded in the README, with SVGs available for export. The [generator](../../scripts/build_readme_figures.py) rebuilds all five figure pairs in this directory using a separate plotting environment; training dependencies are unchanged.

</div>

<div class="ai">

The matched 171M comparison uses all ten evaluation endpoints from the [paired 100k experiment](../nibi-paired-unique-b2048-100k-20260909/learning-curve.json) and 10,001 logged training-loss records per recipe from its compressed training logs. The baseline is blue and the improved recipe orange. MLM training loss is on the left. On the right, solid circles show contact P@L against the left axis and dashed squares show MLM validation loss against the right axis. Contact bands retain the 5,000-replicate 95% chain-bootstrap intervals over 20,775 chains. Validation uses 4,096 sequences. The training panel shows faint raw losses and the same 100-record trailing mean, spanning 1,000 updates after warmup; the inset shows raw steps 1–1,000. Both recipes have one training seed. [Extracted training statistics](training-curve-summary.json) match the previous figure's statistics exactly.

</div>

<div class="ai">

AutoResearch progress comes from the unchanged [78-run log through R38](../program2/runs-through-r38.tsv). The [history reader](../../scripts/plot_autoresearch_history.py) checks all 39 method means, sample SDs, incumbent transitions and historical keep/discard decisions against the two seed results. Orange points show means with sample-SD bars; the blue step curve follows the retained recipe. Labels 1–5 identify the accepted changes at rounds 1, 4, 10, 22 and 29, listed in the [numbered improvement table](../program2/README.md#numbered-improvements). This historical campaign accepted a candidate when its mean validation-loss reduction exceeded its own sample SD. Its final two accepted changes used approximately 142M parameters, before the current ±5% size rule. The plot does not reinterpret those results using the current example program's confidence-interval rule.

</div>

<div class="ai">

The released-model comparison uses the unchanged [comparison data](../readme-overview-20260913/comparison-data.json). Bars show P@L on our frozen 20,775-chain split. Whiskers show the available 95% chain-bootstrap intervals for ESM-2 150M and our final 171M model; intervals were not recovered for ESMC-300M or ESMC-600M. The original [source and compute notes](../readme-overview-20260913/README.md) are retained, with the additional ESM-2 estimate documented below; the previous figures and measurement records remain in their original directories.

</div>

<div class="ai">

The data diagram summarizes [data preparation](../../../docs/DATA.md). The loop diagram summarizes the example in [program.md](../../../autoresearch/program.md), with the task's scale-up test after search. Diagram labels describe stages; they are not measured results.

</div>

<div class="ai">

From the repository root, with NumPy and Matplotlib installed in a plotting environment:

</div>

<div class="ai">

```bash
python .dev/scripts/build_readme_figures.py
```

</div>

<div class="ai">

## Training compute estimates

</div>

<div class="ai">

The README uses `3 × (2 × P + 4 × layers × context × width) × tokens`, summed over stages where applicable. This includes an attention term in addition to the usual `6 × P × tokens` approximation. ESMC-300M, ESMC-600M and our 171M estimates retain the [previous stage budgets and calculations](../readme-overview-20260913/README.md), using nominal tokens from batch size × maximum context × updates. The estimates are not measured hardware operations.

</div>

<div class="ai">

For ESM-2 150M, [Zeming Lin’s thesis, §A.3.2.4](https://cs.nyu.edu/media/publications/ZemingLin-phd.pdf#page=148) (printed page 114) reports 500,000 updates at two million tokens per update and random crops of 1,024 tokens for long proteins. The [official checkpoint configuration](https://huggingface.co/facebook/esm2_t30_150M_UR50D/blob/main/config.json) specifies 30 layers and width 640; our [checkpoint audit](../released-150m-contact-20260913/esm2/RESULT_VERIFIED.json) counts 148,140,154 parameters. Substituting these values gives `3 × (2 × 148,140,154 + 4 × 30 × 1,024 × 640) × 10¹² = 1.124770524 × 10²¹ FLOPs`, rounded to **1.125 × 10²¹** in the README. [Calculation inputs and source hash](esm2-compute-estimate.json).

</div>

<div class="ai">

This ESM-2 estimate assumes the maximum crop length throughout; the original run’s sequence-length and padding distributions are unavailable here. Its token budget comes from reported tokens per update, while the other rows use maximum-context budgets. All rows use the same FLOP formula, but that difference in token accounting, along with different corpora and training schedules, limits cross-model compute comparisons. No estimate establishes equal-compute performance.

</div>
