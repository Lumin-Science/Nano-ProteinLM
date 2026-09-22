# 171M contact P@L

This file is the existing two-seed, one-hour/four-L40S example task. The [agent benchmark protocol](../docs/autoresearch.md) defines the new search tracks and three-setting scale-up submission; its allowances are not automatically implemented by this task script.

## Background

Improve protein-embedding training recipes under a small compute budget. The 171M baseline follows the paper's 170M scaling model: 24 layers, width 768, and 170.7M parameters ([Appendix A.1.4.1, Table S4, p. 29](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1.full.pdf#page=29)). Runtime code lives in [src/nanoprotein](../src/nanoprotein), with public decontaminated data and frozen evaluations.

[default.yaml](../configs/default.yaml) is the starting recipe; [esmc-171m.yaml](../configs/esmc/esmc-171m.yaml) is the reference. The larger presets in [configs/esmc/README.md](../configs/esmc/README.md) support separate experiments and are outside this task's size bound.

This task uses the same protocol as [171m-validation-loss.md](171m-validation-loss.md), with contact P@L as the research reward instead of MLM validation loss.

## Autoresearch protocol

Compare mean contact P@L after time-limited training of approximately fixed-size models on the same data and hardware.

- **Score:** mean contact P@L across training seeds 42 and 43; higher is better. Each seed's P@L is precision among the top L predicted long-range contacts, where L is chain length, averaged over all 20,775 frozen evaluation chains. Report per-seed values and sample SD across seeds. MLM validation loss is recorded as a diagnostic and does not determine the research score. Chain-bootstrap confidence intervals are separate from the across-seed sample SD used by the research loop.
- **Compute:** one hour of training on four L40S GPUs per seed, starting from scratch. The clock excludes setup, final checkpoint saving and evaluation.
- **Boundaries:** keep data, tokenizer, evaluation, hardware and budget accounting fixed; keep actual trainable parameters within ±5% of the original 171M model. No held-out training, pretrained weights, dummy parameters or altered scores. Preserve Stage-1 context and source mixture; use the script's linear warmup followed by constant peak LR, with no post-warmup decay. Recipe and training implementation changes are permitted within these limits. The task scripts, evaluation implementation, dependencies and input receipts are protected. The agent reviews these boundaries and the completed run records; a successful command alone does not establish compliance.

Configure the two local roots using [.env.example](../.env.example) and the [README.md](../README.md#setting-up-data--environments). In a dedicated `DATA_ROOT`, run `bash runs/setup.sh --training-shards 7` to prepare the frozen benchmark corpus and all MLM validation and contact P@L assets. The general setup default is now 30 shards; it does not change this task's data contract. Then run:

```bash
bash tasks/171m-p-at-l_ar.sh configs/default.yaml experiment-p-at-l-001
```

The [171m-p-at-l_ar.sh](171m-p-at-l_ar.sh) script delegates to the same [measurement command](171m-validation-loss_ar.sh) as the validation-loss task. It loads local paths, snapshots the recipe, trains/evaluates both seeds through the standard APIs and summarizes both metrics in `$OUTPUT_ROOT/experiment-p-at-l-001/summary.json`. Use `metrics.p_at_l.mean` as the reward, `metrics.p_at_l.sample_sd` as its sample SD and each entry's `p_at_l` in `runs` as the per-seed score. P@L is stored as a fraction from 0 to 1; use the same units for all score comparisons.

The standard evaluator runs accelerated P@L by default: one shared probe, 32 workers across the four allocated GPUs, all 20,775 frozen chains and the same 5,000-replicate global chain bootstrap. MLM validation retains its 32 sequences and existing masking protocol. Evaluation remains outside the one-hour training clock; no separate fast-evaluation command is needed.

Use a fresh experiment name for each candidate. Setup and GPU allocation happen before this command. Incomplete runs cannot supply a benchmark score. Loop policy lives in [autoresearch/program.md](../autoresearch/program.md); apply its higher-is-better comparison rule to P@L.

## Test of Progress

The benchmark owner manually verifies a selected recipe with a fixed 24.20B-token training budget on four H100s per seed, then compares full validation loss and P@L against the reference. This is separate from the agent's research loop.

Use the manual training/evaluation commands in [EVALUATION.md](../docs/EVALUATION.md#manual-test-of-progress) and preserve the full final optimizer checkpoint for continuation. Historical results retain their original protocols; see the [benchmark protocol and historical results](../docs/autoresearch.md).
