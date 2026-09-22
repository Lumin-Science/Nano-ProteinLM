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
- <span class="ai">[171m-validation-loss.md](../tasks/171m-validation-loss.md): historical L40S measurement profile for the sequential method.</span>
- [EVALUATION.md](EVALUATION.md#manual-test-of-progress): matched owner-run verification.
- <span class="ai">[Best recipe versus baseline](leaderboard/BEST_RECIPE_22_09_26.md): completed 100k-step comparison and worked method examples.</span>
- <span class="ai">[autoresearch.md](autoresearch.md): benchmark search and evaluation budgets; [LEADERBOARD.md](LEADERBOARD.md): recorded measurements; [sequential search](AUTORESEARCH_BASELINE.md): example loop and experiment provenance.</span>

Released ESMC checkpoints are capability references, with results in
[EVALUATION.md](EVALUATION.md#released-esmc-checkpoint-pl); they used substantially
more pretraining compute. Earlier local pilot workflows and their measurements
remain in [baseline-workflows.md](../.dev/reports/archive/baseline-workflows.md).
