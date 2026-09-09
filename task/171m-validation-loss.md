# 171M validation loss

## Background

Improve protein-embedding training recipes under a small compute budget.
The 171M baseline follows the paper's 170M scaling model: 24 layers, width 768,
and 170.7M parameters ([Appendix A.1.4.1, Table S4, p. 29](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1.full.pdf#page=29)).
Runtime code lives in [src/nanoprotein](../src/nanoprotein/), with public\ndecontaminated data and frozen evaluations.

[Setting 3](../configs/default.yaml) is the starting recipe;
[171M AdamW](../configs/esmc-171m-original.yaml) is the reference.
The [300M/600M reference configs](../configs/reference/README.md) support separate,
larger-model experiments and are outside this task's size bound.

## Autoresearch protocol

Compare mean validation loss after time-limited training of approximately
fixed-size models on the same data and hardware.

- **Score:** mean sequence-mean MLM validation loss across two fixed training
  seeds; lower is better. Report per-seed values and sample SD. Contact P@L is
  recorded as a diagnostic and does not determine the research score.
- **Compute:** one hour of training on four L40S GPUs per seed, starting from
  scratch. The clock excludes setup, final checkpoint saving and evaluation.
- **Boundaries:** keep data, tokenizer, evaluation, hardware and budget accounting
  fixed; keep actual trainable parameters within ±5% of the original 171M model.
  No held-out training, pretrained weights, dummy parameters or altered scores.
  Recipe and training implementation changes are permitted within these limits.
  The task scripts, evaluation implementation, dependencies and input receipts
  are protected. Search strategy and acceptance decisions belong to the agent.

Configure the two local roots in [`.env`](../.env.example) using the
[setup instructions](../README.md#setup). Then run:

```bash
bash task/171m-validation-loss_ar.sh configs/default.yaml experiment-001
```

The [research script](171m-validation-loss_ar.sh) contains the direct
`nanoprotein.train` and `nanoprotein.evaluate` calls, their fixed arguments,
and the seed loop. Change the recipe argument and use a fresh experiment name
for each candidate. Results are written to `$OUTPUT_ROOT/experiment-001/summary.json`;
incomplete runs cannot supply a benchmark score.

## Test of Progress

The benchmark owner manually verifies a selected recipe with a fixed 24.20B-token
training budget on four H100s per seed, then compares full validation loss and
P@L against the reference. This is separate from the agent's research loop.

Use the [manual training/evaluation commands](../docs/EVALUATION.md#manual-test-of-progress)
and preserve the full final optimizer checkpoint for continuation.
Historical results retain their original protocols; see the
[experiment records](../docs/AUTORESEARCH.md).
