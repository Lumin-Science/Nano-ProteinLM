# Baselines and comparisons

Use [esmc-171m.yaml](../configs/esmc/esmc-171m.yaml) as the ESMC-style AdamW
reference and [default.yaml](../configs/default.yaml) as the improved 171M recipe.
They share the small-budget backbone but differ in optimizer, normalization,
routing, initialization, batch balancing and loss weighting. Their preset LR/WD,
warmup and batch layout also differ; align those explicitly for a matched run.

[esmc-300m.yaml](../configs/esmc/esmc-300m.yaml) and
[esmc-600m.yaml](../configs/esmc/esmc-600m.yaml) provide the original-size
architectures. Their local training assumptions and paper references are in
[configs/esmc/README.md](../configs/esmc/README.md).

- [USAGE.md](USAGE.md): ordinary training and evaluation commands.
- [171m-validation-loss.md](../tasks/171m-validation-loss.md): research score, budget and boundaries.
- [EVALUATION.md](EVALUATION.md#manual-test-of-progress): matched owner-run verification.
- [BEST_RECIPE_VS_BASELINE.md](BEST_RECIPE_VS_BASELINE.md): completed 100k-step comparison and worked method examples.
- [AUTORESEARCH.md](AUTORESEARCH.md): experiment records and provenance.

Released ESMC checkpoints are capability references, with results in
[EVALUATION.md](EVALUATION.md#released-esmc-checkpoint-pl); they used substantially
more pretraining compute. Earlier local pilot workflows and their measurements
remain in [baseline-workflows.md](../.dev/reports/archive/baseline-workflows.md).
