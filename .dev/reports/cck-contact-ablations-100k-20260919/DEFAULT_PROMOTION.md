# Default recipe decision: separate Q/K/V Muon

On September 21, 2026, the owner selected the previous improved default plus `muon_split_qkv: true` as the new default. Query centering and RMS restoration are not included. The backbone, parameter count, normalization, routing, initialization, batch balance, loss weighting, optimizer-group multipliers and ordinary preset defaults are otherwise unchanged. This is an owner decision informed by the completed study, not a new application of the one-hour search's statistical keep rule.

The optimizer implementation was imported from the accepted contact-search branch at commit `e330dcd737f8f679a00584645bc378acabaf7269`; the separate-Q/K/V method was kept in trial 031. `SplitQKVMuon` presents three storage-sharing square views of each fused QKV matrix to ordinary PyTorch Muon, so momentum, orthogonalization and matrix-shape scaling are applied separately. The model retains the fused weight tensor, unchanged parameter names and unchanged parameter count. DDP synchronization and gradient clipping continue to act on the original model parameters. Other matrices retain their existing optimizer assignment.

## Matched 100k-step evidence

| Recipe | Separate Q/K/V Muon | Query centering + RMS restoration | P@L ↑ | Validation MLM NLL ↓ |
|---|---|---|---:|---:|
| Selected default addition | On | Off | 33.449770% | 2.410035 |
| Full search incumbent | On | On | 33.485122% | 2.413881 |
| No separate Q/K/V | Off | On | 32.767462% | 2.416150 |

All three runs started from scratch with seed 42, 100,000 updates, global batch 1,024, context 512, four L40S GPUs per run, FA2, 170,559,856 parameters and the same frozen corpus. Each consumed exactly 102,400,000 sequences and 24,196,983,520 model tokens without resampling. P@L used all 20,775 frozen chains with 5,000 bootstrap replicates; validation loss used the same 4,096 held-out sequences. The final checkpoint also had the best observed P@L in every run. See [the comparison receipt](COMPARISON.json), [completion receipt](COMPLETION_SUMMARY.json), [study contract](STUDY.json) and [full report](STATUS.md).

With centering/RMS restoration enabled, separate Q/K/V updates improved P@L by 0.717660 percentage points. With separate Q/K/V enabled, adding centering/RMS restoration changed P@L by only +0.035352 points and validation NLL by +0.003846. This three-arm study does not include a matched fourth arm with both additions disabled, so it does not directly estimate separate-Q/K/V's standalone gain over the previous default. These are one-training-seed point estimates; chain-bootstrap intervals do not estimate training-seed variability, and paired significance has not been calculated. The new default favors the simpler of the two nearly tied P@L recipes and the one with lower observed validation loss.

The study's FA2/L40S/seed-42 execution settings are evidence-specific. The general default retains its existing FA3/four-H100 execution preset and seed; those differences must be overridden explicitly when reproducing this study. Its 100k result is not a new owner-run H100 Test of Progress or a replacement for the older six-run H100 comparison.

## Compatibility and validation

Missing or false `muon_split_qkv` retains the original fused-matrix optimizer, so archived configurations and their optimizer state remain supported. A run cannot switch optimizer partitioning during resume: its original resolved recipe must be used. Research checkpoints containing the experimental centered-query model fields remain paired with their preserved research source and recorded recipes; this promotion imports only the separate-Q/K/V implementation.

The imported tests compare parameter values and momentum buffers bit-for-bit with independent ordinary Muon updates, exercise missing gradients, accumulation, clipping, zeroing and closures, verify storage ownership without added parameters, and check exact optimizer save/load continuation. The default-config regression permits precisely the new Q/K/V flag relative to the previous default after removing historical stopping budgets.

Validation completed successfully: all 21 selected regression tests passed on CCK, including the CUDA Muon reference test, and the promoted default completed a 20-step smoke run on four L40S GPUs with 170,559,856 parameters and 20,480 sequences. Lint and formatting checks passed. The isolated snapshot initially omitted two historical YAML test fixtures; after restoring them, the affected preset suite passed. See [the validation receipt](DEFAULT_VALIDATION.json).
