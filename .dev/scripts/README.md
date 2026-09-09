# Development tools

These utilities support report rendering, release preparation and historical
analysis. Training and autoresearch runtime commands live in
[src/nanoprotein](../../src/nanoprotein/).

- [Autoresearch history](plot_autoresearch_history.py): validates paired-seed
  statistics and renders the [research curve](../reports/program2/validation-loss.png).
- [Best-recipe illustrations](plot_best_recipe_explainer.py): renders measured
  comparisons and worked batch-balance/sqrt-loss examples.
- [Nibi baseline curve](plot_nibi_baseline_learning_curve.py): renders the ten
  audited checkpoints from the eight-H100 baseline.
- [Evaluation bundle](build_evaluation_bundle.py): prepares frozen split ledgers
  for release construction.
- [Historical round summary](summarize_autoresearch_round.py): reads the older
  campaign's output layout.

Run plotting tools from a separate Matplotlib environment, as documented in
[AutoResearch](../../docs/AUTORESEARCH.md). The training dependency lock is unchanged.
