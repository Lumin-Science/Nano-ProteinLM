# Experiment TODO

- [x] **Exclude change 4: R22 narrower FFN (2048 → 1536) from the fixed-size Test Leaderboard.** Confirmed at the user's request on September 8, 2026, because it changes model size (170.67M → 142.36M). The test sequence is Baseline → 1 (Muon/R02) → 2 (batch balance) → 3 (sqrt loss) → 5 (tied embeddings), with FFN width 2048 throughout. No narrower-FFN test is planned. Its accepted research result and [historical implementation](.dev/configs/archive/program2/r22_ffn1536_seed42.yaml) remain available. See the [historical scale-up table](docs/autoresearch-sequential-search.md#detailed-scale-up-results) and [archived six-recipe comparison](docs/archive/TEST_LEADERBOARD_20260908.md).

- [x] **Support the Test of Progress token budget.** Training now accepts
  `max_model_tokens`, stops at a synchronized optimizer boundary, records the
  target/actual tokens and overrun, and can resume to a larger token endpoint.
  The complete train-and-evaluate command is in [tasks/171m-validation-loss.md](tasks/171m-validation-loss.md#test-of-progress).
  Historical time/step-based runs keep their existing behavior.

- [x] **Consolidate maintained recipes and launchers.** Keep `configs/default.yaml`
  and the 171M/300M/600M references in `configs/esmc/`, with setup and speedrun
  scripts. Historical presets live in `.dev/configs/archive/`; old plans live
  in `.dev/reports/archive/`. See [configs/README.md](configs/README.md).

- [x] **Publish evaluation data and remove campaign names from current usage.**
  Setup installs the frozen contact data/evaluator. Current commands use descriptive
  recipe names; historical reports preserve their recorded identities.

- [ ] **Complete the unused-code audit.** Review unused optimizer helpers without
  removing compatibility aliases needed by saved recipes and checkpoints.
