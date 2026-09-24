# Experiment TODO

- [x] **Exclude change 4: R22 narrower FFN (2048 → 1536) from the fixed-size Test Leaderboard.** Confirmed at the user's request on September 8, 2026, because it changes model size (170.67M → 142.36M). The test sequence is Baseline → 1 (Muon/R02) → 2 (batch balance) → 3 (sqrt loss) → 5 (tied embeddings), with FFN width 2048 throughout. No narrower-FFN test is planned. Its accepted research result and historical implementation remain available. See the [historical scale-up table](../docs/leaderboard/nanop-best-171m-round1.md#1-the-complete-comparison) and archived six-recipe comparison.

- [x] **Support the final-evaluation token budget.** Training now accepts
  `max_model_tokens`, stops at a synchronized optimizer boundary, records the
  target/actual tokens and overrun, and can resume to a larger token endpoint.
  The complete train-and-evaluate command is in [docs/AUTORESEARCH.md](../docs/AUTORESEARCH.md#final-evaluation).
  Historical time/step-based runs keep their existing behavior.

- [x] **Consolidate maintained recipes and launchers.** Keep the search-setting
  recipes in `configs/autoresearch/` and final-evaluation recipes in
  `configs/test-100k/`, with setup and speedrun scripts. Historical presets live in `.dev/configs/archive/`; old plans live
  in `.dev/reports/archive/`. See [configs/README.md](../configs/README.md).

- [x] **Publish evaluation data and remove campaign names from current usage.**
  Setup installs the frozen contact data/evaluator. Current commands use descriptive
  recipe names; historical reports preserve their recorded identities.

- [ ] **Complete the unused-code audit.** Review unused optimizer helpers without
  removing compatibility aliases needed by saved recipes and checkpoints.
