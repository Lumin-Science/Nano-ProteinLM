# Release cleanup status

## Implemented

- Two recommended 171M recipes: [current-best Setting 3](../configs/default.yaml) and
  [original 171M AdamW](../configs/esmc-171m-original.yaml). All 27 other YAMLs
  and the scale-up manifest are in the [config archive](../configs/archive/README.md),
  preserving their bytes and recorded hashes. This replaces the earlier plan
  to maintain four public presets for now.
- [300M/600M reference presets](../configs/reference/README.md) expose the paper's
  original-size architectures separately, with documented local training assumptions.
- Two entry points in `runs/`: [setup](../runs/setup_env_and_data.sh) installs the
  locked environment and prepares/reuses the fixed training subset;
  [speedrun](../runs/speedrun.sh) calls setup and trains Setting 3 by default.
  It accepts a recipe, run name and normal training CLI options.
- Old training launchers are [archived](archive/runs/README.md). Parallel
  evaluation helpers moved to `scripts/` without changing evaluation behavior.
- `.env` contains only `DATA_ROOT` and `OUTPUT_ROOT`. Setup and training are
  documented in the [README](../README.md#usage); the direct APIs remain
  available for independent research.
- [program.md](../program.md) selects a [task definition](../task/171m-validation-loss.md).
  The task's shell script declares research measurements. Test of Progress stays
  manual, using the [documented commands](EVALUATION.md#manual-test-of-progress).
  The obsolete root `program2.md` is [archived](archive/program2.md).

## Remaining work

- Package the frozen contact dataset and evaluator source for a fresh public
  installation. Until then, training setup prepares only training/MLM data;
  contact evaluation requires the existing verified assets.
- Rename the historical `program2` report family around a meaningful campaign
  name if useful, preserving result payloads, provenance and external links.
- Audit old defaults in the optional evaluation launchers. Moving them has not
  changed their existing environment-variable interface or scoring settings.
- Review unused code separately: `_OptimizerBundle.set_param_group_value` has no
  known callers; `muon_initial_momentum` may be a checkpoint compatibility alias.
  Check saved configurations before removing either.

## Compatibility to preserve

Keep model support for tied embeddings, configurable FFN widths, original AdamW
and LayerNorm, Muon and RMSNorm, routing/init, batch balance and sqrt loss. Existing
checkpoints require these even when an option is disabled in the default recipe.
Retain optimizer-state portability, periodic evaluation, token-budget stopping,
FA2/FA3, compilation and gradient checkpointing. Do not infer that a setting is
unused merely because the current recipe disables it.

Frozen data, evaluation implementations, probes and scoring caches remain unchanged.
Reports retain their measured results and immutable audit payloads; only links to
relocated recipes are updated. Existing remote training checkouts are unaffected.
