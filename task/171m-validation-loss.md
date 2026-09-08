# 171M validation loss

## Background

Improve training recipes for useful protein representations using the existing
ESMC-like training stack, public decontaminated corpus and frozen evaluations.
[current_best](../configs/default.yaml) is the default starting recipe;
[original 171M AdamW](../configs/esmc-171m-original.yaml) is the reference.

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
`nano_protein.train` and `nano_protein.evaluate` calls, their fixed arguments,
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
