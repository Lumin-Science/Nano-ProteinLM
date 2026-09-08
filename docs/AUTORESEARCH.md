# AutoResearch records

The single task definition is [program.md](../program.md): **Background**,
**Autoresearch** (protocol and boundaries), and **Test of Progress**. It contains
the executable command examples and all task rules. This page is a results index.

The historical [38-round research campaign](../reports/program2/README.md)
predates the current ±5% parameter bound. Its 142M FFN/tied endpoints retain
their original protocol. The [completed Test Leaderboard](../README.md#test-leaderboard)
used 100k steps, batch 1,024 and one training seed (20260824); every run consumed
24,200,224,761 model tokens. Those records are not two-seed token-stopped tests.

The displayed Settings 1 → 2 → 3 → 5 use the full R02 base; Setting 1 is not a
Muon-only ablation. See [executed configurations](PROGRAM2_SCALEUP.md) and
[best versus baseline](BEST_RECIPE_VS_BASELINE.md) for details.

## Commands and results

- [Baseline commands](../README.md#baselines) for research and scale-up training.
- [38-round curve](../README.md#autoresearch), including all means and sample SDs.
- [Test Leaderboard](../README.md#test-leaderboard).
- [All 78 runs through R38](../reports/program2/runs-through-r38.tsv).
- [Import details and the original R29 audit](../reports/program2/README.md).
- [Per-method statistics through R29](../reports/program2/methods.tsv).
- [Evaluation setup and execution](EVALUATION.md).

To regenerate the curve without changing the training environment:

```bash
python3 -m venv /tmp/nano-esmc-plot
/tmp/nano-esmc-plot/bin/python -m pip install 'matplotlib==3.11.1'
/tmp/nano-esmc-plot/bin/python scripts/plot_autoresearch_history.py
```

Run these commands from the repository root. The script verifies seed means,
sample SDs, and keep/discard decisions before writing PNG and SVG files to
`reports/program2/`. It reads `reports/program2/runs-through-r38.tsv` by default;
use `--input path/to/results.tsv` for another per-run or per-method log.
