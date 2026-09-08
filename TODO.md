# Experiment TODO

- [x] **Exclude change 4: R22 narrower FFN (2048 → 1536) from the fixed-size
  Test Leaderboard.** Confirmed at the user's request on September 8, 2026,
  because it changes model size (170.67M → 142.36M). The test sequence is
  Baseline → 1 (Muon/R02) → 2 (batch balance) → 3 (sqrt loss) → 5 (tied
  embeddings), with FFN width 2048 throughout. No narrower-FFN test is planned.
  Its accepted research result and [historical implementation](configs/program2/r22_ffn1536_seed42.yaml)
  remain available. See the [current Test Leaderboard](README.md#test-leaderboard)
  and [archived six-recipe comparison](docs/archive/TEST_LEADERBOARD_20260908.md).

- [ ] **Implement the v1 fixed-token test adapter.** The
  [task contract](tasks/protein-embedding.yaml) specifies 24,200,224,761 global
  non-padding tokens, terminal-update overrun receipts and N=2 matched seeds.
  The trainer currently implements time/step stopping only. Verify synchronized
  counting, exact-boundary behavior, terminal-step overrun, incomplete wall-time
  stops and checkpoint/resume semantics before any new v1 test launch.
  Historical 100k-step results keep their original protocol.
