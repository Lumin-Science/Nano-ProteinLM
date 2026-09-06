# Experiment TODO

- [ ] **Deferred: R22 narrower FFN (2048 → 1536).** Temporarily excluded from
  the active four-setting 100k comparison at the user's request on September 6,
  2026. All active configs retain FFN width 2048. After the R02-RoPE10k → batch
  balance → sqrt loss → tied-embedding comparison, reconsider FFN narrowing as
  a separate capacity/compute ablation with the same LR, WD, batch and update
  budget. Its accepted implementation and historical
  [Program 2 recipe](configs/program2/r22_ffn1536_seed42.yaml) remain available;
  no narrower-FFN run is scheduled by the current plan. See
  [the current four settings](docs/PROGRAM2_SCALEUP.md#exact-cumulative-changes).
