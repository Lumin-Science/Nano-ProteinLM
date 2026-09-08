# AutoResearch

An AutoResearch task has five parts: a research question, an established
codebase, a repeatable search protocol, explicit experiment boundaries, and a
separate test protocol. Use **search score** for the metric selecting candidates
and **test metrics** for the final evidence of transfer.

The [portable task standard](AUTORESEARCH_TASK_STANDARD.md) defines this format.
The protein task is specified in [program.md](../program.md) and
[tasks/protein-embedding.yaml](../tasks/protein-embedding.yaml), with a
[JSON Schema](../tasks/task.schema.json) and [OpenMM examples](../tasks/examples/).
The structured contracts are specifications; existing runners do not load them.

## Protein task

Find better training recipes for useful ESMC-family protein representations at
approximately fixed model size and fixed data. Search on held-out MLM loss;
confirm transfer using both MLM loss and long-range contact P@L. An attention
contact probe alone does not establish broad downstream-embedding quality.

| Property | Research | New v1 test |
|---|---|---|
| Budget per training seed | 3,600 synchronized training-loop seconds | 24,200,224,761 non-padding model tokens, including BOS/EOS |
| Hardware | 4 L40S, BF16 | 4 H100, BF16, FA3 |
| Repeats | N=2, seeds 42 and 43 | N=2, seeds 42 and 43 |
| Evaluation | 32 held-out MLM sequences; full P@L diagnostic | 4,096 MLM sequences and all 20,775 contact chains |
| Decision | Mean loss improvement over incumbent exceeds candidate sample SD | Both mean loss and mean P@L improve relative to the named comparator |
| LR schedule | 554-step warmup, then constant | 1,000-step warmup, then constant; base LR 5e-4, WD 0.01 |
| Runner status | Existing wall-time runner | Token stopping adapter required before launch |

N counts independent from-scratch training seeds, not GPU ranks or bootstrap
samples. Both decisions require all repeats and validity checks. The search
noise-margin rule and the test direction rule are not significance tests.
Report mean ± sample SD across seeds and separate per-seed P@L chain-bootstrap
95% intervals. Test assets overlap research evaluation; this checks transfer
across training budgets rather than an untouched holdout.

The [file/key allowlist](../program.md#4-experiment-boundaries) permits model,
optimizer, backward loss and execution experiments while freezing data,
tokenizer, evaluation, timing/accounting semantics and environment. Actual
trainable parameters must remain within **162,137,610–179,204,726**, or ±5% of
the original 170,671,168-parameter baseline. Unlisted files are protected during
candidate research; the task owner can revise the contract between campaigns.

## Historical results

The [38-round research history](../reports/program2/README.md) used its original
one-hour protocol. Its 142M FFN/tied endpoints predate the ±5% bound and remain
historical results, not v1-compliant candidates.

The [completed Test Leaderboard](../README.md#test-leaderboard) uses 100,000
steps, global batch 1,024 and one training seed, 20260824. Its matched streams
consumed 24,200,224,761 model tokens. These runs stopped on steps and are not
retroactively token-stopped, two-seed tests. All five displayed recipes are
complete; Setting 3 is best on both metrics in that comparison.

Settings 1 → 2 → 3 → 5 retain the full R02 base, adding batch balance, sqrt loss
and tied embeddings. Setting 1 is not a pure Muon ablation. The narrower-FFN
change is excluded from this comparison. See [exact executed configurations](PROGRAM2_SCALEUP.md)
and [best versus baseline](BEST_RECIPE_VS_BASELINE.md); their original run IDs
and historical numbering are retained.

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
