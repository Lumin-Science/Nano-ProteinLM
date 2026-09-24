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
- [Contact setup bundle](package_contact_evaluation.py): packages the frozen P@L
  payload and evaluator with checksums for installation.
- [Historical round summary](summarize_autoresearch_round.py): reads the older
  campaign's output layout.
- <span class="ai">[Clean starter](build_clean_starter.py): builds and audits the history-free AutoResearch starter from the plain reference. Maintainers only; it is never shipped to agents.</span>

Run plotting tools from a separate Matplotlib environment, as documented in [sequential-search plot guide](../../docs/AUTORESEARCH_BASELINE.md#evidence-and-plot-regeneration). The training dependency lock is unchanged.

<div class="ai">

## Publishing an AutoResearch starter release

</div>

<div class="ai">

Each release is a single root commit published as a tag; participants clone it with the command in [AUTORESEARCH.md](../../docs/AUTORESEARCH.md#preparation). Commit the source changes first, set `NAME` in the builder to the new tag, then build, audit and publish from the repository root:

</div>

<div class="ai">

```bash
release=autoresearch-v0
.venv/bin/python .dev/scripts/build_clean_starter.py --output /tmp/nano-starter
git -C "/tmp/nano-starter/$release" init -q
git -C "/tmp/nano-starter/$release" add -A
git -C "/tmp/nano-starter/$release" commit -q -m "Release plain ESMC AutoResearch starter $release"
git fetch /tmp/nano-starter/$release HEAD
git tag -a "$release" FETCH_HEAD -m "AutoResearch starter $release"
git push origin "$release"
```

</div>

<div class="ai">

Before pushing, clone the tag with the documented command from a local `file://` URL and run the starter's `.dev/tests` in the clone. Publish a new tag for every change instead of moving an existing one.

</div>
