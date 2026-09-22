# 171M validation loss

This file is the existing two-seed, one-hour/four-L40S example task. The [agent benchmark protocol](../docs/autoresearch.md) defines the new search tracks and three-setting scale-up submission; its allowances are not automatically implemented by this task script.

## Background

Improve protein-embedding training recipes under a small compute budget.
The 171M baseline follows the paper's 170M scaling model: 24 layers, width 768,
and 170.7M parameters ([Appendix A.1.4.1, Table S4, p. 29](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1.full.pdf#page=29)).
Runtime code lives in [src/nanoprotein](../src/nanoprotein), with public
decontaminated data and frozen evaluations.

[default.yaml](../configs/default.yaml) is the starting recipe;
[esmc-171m.yaml](../configs/esmc/esmc-171m.yaml) is the reference.
The larger presets in [configs/esmc/README.md](../configs/esmc/README.md) support
separate experiments and are outside this task's size bound.

## Autoresearch protocol

Compare mean validation loss after time-limited training of approximately
fixed-size models on the same data and hardware.

- **Score:** mean sequence-mean MLM validation loss across training seeds
  42 and 43; lower is better. Report per-seed values and sample SD. Contact P@L is
  recorded as a diagnostic and does not determine the research score.
- **Compute:** one hour of training on four L40S GPUs per seed, starting from
  scratch. The clock excludes setup, final checkpoint saving and evaluation.
- **Boundaries:** keep data, tokenizer, evaluation, hardware and budget accounting
  fixed; keep actual trainable parameters within ±5% of the original 171M model.
  No held-out training, pretrained weights, dummy parameters or altered scores.
  Preserve Stage-1 context and source mixture; use the script's linear warmup
  followed by constant peak LR, with no post-warmup decay.
  Recipe and training implementation changes are permitted within these limits.
  The task scripts, evaluation implementation, dependencies and input receipts
  are protected. The agent reviews these boundaries and the completed run
  records; a successful command alone does not establish compliance.

Configure the two local roots using [.env.example](../.env.example) and the
[README.md](../README.md#setting-up-data--environments). In a dedicated `DATA_ROOT`, run
`bash runs/setup.sh --training-shards 7` to prepare the frozen benchmark corpus
and all MLM validation and contact P@L assets. The general setup default is now
30 shards; it does not change this task's data contract. Then run:

```bash
bash tasks/171m-validation-loss_ar.sh configs/default.yaml experiment-001
```

The [171m-validation-loss_ar.sh](171m-validation-loss_ar.sh) script loads local paths, snapshots the recipe, trains/evaluates both seeds through the standard APIs and summarizes the scores in `$OUTPUT_ROOT/experiment-001/summary.json`. The standard evaluator runs accelerated P@L by default: it fits one shared probe, scores all 20,775 chains with 32 workers across the four allocated GPUs, then performs the same 5,000-replicate global chain bootstrap. MLM validation retains its 32 sequences and existing masking protocol. Evaluation remains outside the one-hour training clock.

Use a fresh experiment name for each candidate. Setup and GPU allocation happen before this command. Incomplete runs cannot supply a benchmark score. Loop policy lives in [autoresearch/program.md](../autoresearch/program.md).

## Test of Progress

The benchmark owner manually verifies a selected recipe with a fixed 24.20B-token
training budget on four H100s per seed, then compares full validation loss and
P@L against the reference. This is separate from the agent's research loop.

Use the manual training/evaluation commands in [EVALUATION.md](../docs/EVALUATION.md#manual-test-of-progress) and preserve the full final optimizer checkpoint for continuation. Historical results retain their original protocols; see the [benchmark protocol and historical results](../docs/autoresearch.md).
