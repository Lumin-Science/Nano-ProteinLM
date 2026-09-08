# Experiment TODO

- [x] **Exclude change 4: R22 narrower FFN (2048 → 1536) from the fixed-size
  Test Leaderboard.** Confirmed at the user's request on September 8, 2026,
  because it changes model size (170.67M → 142.36M). The test sequence is
  Baseline → 1 (Muon/R02) → 2 (batch balance) → 3 (sqrt loss) → 5 (tied
  embeddings), with FFN width 2048 throughout. No narrower-FFN test is planned.
  Its accepted research result and [historical implementation](configs/program2/r22_ffn1536_seed42.yaml)
  remain available. See the [current Test Leaderboard](README.md#test-leaderboard)
  and [archived six-recipe comparison](docs/archive/TEST_LEADERBOARD_20260908.md).
