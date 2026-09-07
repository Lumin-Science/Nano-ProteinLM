# AutoResearch

The default benchmark uses ESMC-171M and selects training changes by held-out
MLM validation loss. The full experiment instructions are in
[program2.md](../program2.md), from the
[`autoresearch-171m-val-loss` branch](https://github.com/Lumin-Science/LuminBench-Nano-ESMC/tree/autoresearch-171m-val-loss).
Earlier contact-selected campaigns remain documented in
[BASELINES.md](BASELINES.md).

## Research setting

Start from the original AdamW baseline: 24 layers, width 768, 12 heads, and
170,671,168 trainable parameters. Each run starts from scratch on four L40S
GPUs, with a synchronized 3,600-second training budget and a 171M parameter cap.

Candidates may change model architecture, optimizer, training loss, batching,
and training implementation. The corpus, mixture, tokenizer, dependency lock,
hardware, training budget, and evaluation remain fixed. Every learning-rate
group warms linearly for 554 steps, then stays at its configured peak.

The reward is `sequence_mean_nll` in `eval-validation/VALIDATION_MLM.json`,
computed over 32 fixed held-out sequences at context 512 with evaluation seed
20260821. Training loss and full 20,775-chain contact P@L are required
diagnostics and do not affect selection.

## Repeats and acceptance

Run the baseline and every candidate on at least N independent training seeds,
with **N = 2 by default**. Choose the seeds before running, compare methods on
the same seed set, and keep evaluation seeds fixed. Every repeat receives the
full training budget and a fresh output directory.

Compute arithmetic mean loss and sample standard deviation (`ddof=1`) from all
completed repeats. Keep a candidate only if:

```text
candidate_mean_val_loss < current_best_mean_val_loss - candidate_val_loss_std
```

The threshold uses the candidate's standard deviation. A tie fails; a single
run cannot qualify. The current best is the last accepted recipe, even when a
discarded candidate has a lower mean. This rule measures improvement relative
to observed seed variation; it is not a formal significance test.

## Scale-up setting

Every kept research change needs a longer run to establish whether its gain
transfers. The current scale-up uses the 171M model family, four H100 GPUs,
100,000 optimizer steps, global batch 1,024, and a 1,000-step warmup followed by
constant learning rates. Evaluate 4,096 held-out MLM sequences and all 20,775
contact chains. Compare each recipe with the baseline and preceding recipe
under these matched settings.

The completed AdamW/R02 comparison has one training seed per recipe. The
validation-loss campaign's scale-up results remain pending in the published
[launch record](../reports/fir-r02-rope10k-100k-20260906/README.md). That plan
adds rank balance, square-root loss weights, and tied embeddings to R02 with
RoPE 10k, retaining FFN width 2048. The FFN-narrowing check is deferred in
[TODO](../TODO.md). See [PROGRAM2_SCALEUP.md](PROGRAM2_SCALEUP.md) for the exact
recipes and evaluation contract.

## Commands and results

- [Baseline commands](../README.md#baselines) for research and scale-up training.
- [29-round curve](../README.md#autoresearch), including all means and sample SDs.
- [Scale-up leaderboard](../README.md#scale-up-leaderboard).
- [Published research history and audit](../reports/program2/README.md).
- [Per-method statistics](../reports/program2/methods.tsv) and
  [all 60 runs](../reports/program2/runs.tsv).
- [Evaluation setup and execution](EVALUATION.md).

To regenerate the curve without changing the training environment:

```bash
python3 -m venv /tmp/nano-esmc-plot
/tmp/nano-esmc-plot/bin/python -m pip install 'matplotlib==3.11.1'
/tmp/nano-esmc-plot/bin/python scripts/plot_autoresearch_history.py
```

Run these commands from the repository root. The script verifies seed means,
sample SDs, and keep/discard decisions before writing PNG and SVG files to
`reports/program2/`.
