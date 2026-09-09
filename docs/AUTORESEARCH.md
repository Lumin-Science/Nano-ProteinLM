# AutoResearch records

The [171M validation-loss task](../tasks/171m-validation-loss.md) is designed for
small-budget training experiments. Its baseline backbone follows the paper's
170M scaling model ([Appendix A.1.4.1, Table S4](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1.full.pdf#page=29));
[300M/600M references](../configs/reference/README.md) are separate from this task.
Its [research shell script](../tasks/171m-validation-loss_ar.sh) calls the standard
training/evaluation APIs directly. [program.md](../program.md) asks an agent to
optimize the task; Test of Progress is manual. This page is a results index.

The historical [38-round research campaign](../.dev/reports/program2/README.md)
predates the current ±5% parameter bound. Its 142M FFN/tied endpoints retain
their original protocol, preserved in [the archived instructions](archive/program2.md).
The [completed Test Leaderboard](../README.md#test-leaderboard)
used 100k steps, batch 1,024 and one training seed (20260824); every run consumed
24,200,224,761 model tokens. Those records are not two-seed token-stopped tests.

The displayed Settings 1 → 2 → 3 → 5 use the full R02 base; Setting 1 is not a
Muon-only ablation. See [executed configurations](PROGRAM2_SCALEUP.md) and
[best versus baseline](BEST_RECIPE_VS_BASELINE.md) for details.

## Commands and results

- [Recommended training command](../README.md#train-a-171m-model) for the 100k-step Setting 3 run.
- [Research and progress commands](../tasks/171m-validation-loss.md) for paired-seed comparisons.
- [38-round curve](../README.md#autoresearch), including all means and sample SDs.
- [Test Leaderboard](../README.md#test-leaderboard).
- [All 78 runs through R38](../.dev/reports/program2/runs-through-r38.tsv).
- [Import details and the original R29 audit](../.dev/reports/program2/README.md).
- [Per-method statistics through R29](../.dev/reports/program2/methods.tsv).
- [Evaluation setup and execution](EVALUATION.md).
- Batch-2,048 / eight-H100 scale-up records: [AdamW baseline](../.dev/reports/nibi-baseline-b2048-100k-eval10k-20260908/README.md)
  and [Setting 3](../.dev/reports/nibi-setting3-b2048-100k-eval10k-20260908/README.md),
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
