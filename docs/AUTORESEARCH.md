# AutoResearch records

The [171M validation-loss task](../tasks/171m-validation-loss.md) and [171M contact P@L task](../tasks/171m-p-at-l.md) are designed for small-budget training experiments. They share the same protocol and differ only in the research reward: minimize MLM validation loss or maximize contact P@L. Their baseline backbone follows the paper's 170M scaling model ([Appendix A.1.4.1, Table S4](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1.full.pdf#page=29)); [300M/600M references](../configs/esmc/README.md) are separate from these tasks. Both use the same [measurement script](../tasks/171m-validation-loss_ar.sh), which calls the standard training/evaluation APIs directly. [autoresearch/program.md](../autoresearch/program.md) defines baseline measurement, iteration, comparison and recording rules; each task owns scientific settings, and the agent reviews compliance. Test of Progress is manual. This page is a results index.

The historical [38-round research campaign](../.dev/reports/program2/README.md)
predates the current ±5% parameter bound. Its 142M FFN/tied endpoints retain
their original protocol, preserved in [the archived instructions](archive/program2.md).
The [completed Test Leaderboard](../README.md#test-leaderboard)
used 100k steps, batch 1,024 and one training seed (20260824); every run consumed
24,200,224,761 model tokens. Those records are not two-seed token-stopped tests.

The cumulative Muon, batch-balance, sqrt-loss and tied-embedding rows share
RMSNorm, residual routing and depth-scaled initialization. The first row adds
that full recipe, so it is not a Muon-only ablation. See [executed configurations](AUTORESEARCH_SCALEUP.md) and
[best versus baseline](BEST_RECIPE_VS_BASELINE.md) for details.

## Commands and results

- [Recommended training command](../README.md#train-the-esmc-style-protein-language-model) for the 100k-step default run.
- Paired-seed research and progress commands: [validation-loss reward](../tasks/171m-validation-loss.md) or [P@L reward](../tasks/171m-p-at-l.md).
- [38-round curve](../README.md#autoresearch), including all means and sample SDs.
- [Test Leaderboard](../README.md#test-leaderboard).
- [All 78 runs through R38](../.dev/reports/program2/runs-through-r38.tsv).
- [Import details and the original R29 audit](../.dev/reports/program2/README.md).
- [Per-method statistics through R29](../.dev/reports/program2/methods.tsv).
- [Evaluation setup and execution](EVALUATION.md).
- Batch-2,048 / eight-H100 scale-up records: [AdamW baseline](../.dev/reports/nibi-baseline-b2048-100k-eval10k-20260908/README.md)
  and [improved default](../.dev/reports/nibi-setting3-b2048-100k-eval10k-20260908/README.md),
  both configured for 100k steps, evaluation every 10k steps and full final
  optimizer checkpoints for continuation.

To regenerate the curve without changing the training environment:

```bash
python3 -m venv /tmp/nano-esmc-plot
/tmp/nano-esmc-plot/bin/python -m pip install 'matplotlib==3.11.1'
/tmp/nano-esmc-plot/bin/python .dev/scripts/plot_autoresearch_history.py
```

Run these commands from the repository root. The script verifies seed means,
sample SDs, and historical keep/discard decisions before writing PNG and SVG files to
`.dev/reports/program2/`. It reads `.dev/reports/program2/runs-through-r38.tsv` by default;
use `--input path/to/results.tsv` for another per-run or per-method log.

The figure shows every tested recipe as a faint point with a sample-SD bar and
the retained recipe as a teal step line. Numbers 1–5 identify the cumulative
changes in the table below it. Amber markers 4–5 use roughly 142M parameters.
The plot omits long provenance text; the source TSV and full protocol remain in
the linked campaign report.
