# CCK contact-P@L component ablations

The accepted contact recipe is built directly on `configs/default.yaml`, the improved Muon/RMSNorm/batch-balance/square-root-loss recipe. Its only scientific additions are query centering in the final eight layers with RMS restoration, and independent Q/K/V Muon updates. At the study's launch, the search worktree's default config had no differences from its `main` branch and was byte-for-byte identical to the local default (SHA-256 `61050e750013c42b71c91eb38750013cf603646cee1c60a464f158b4901f684b`). This experiment exports committed source at `e330dcd` and excludes the uncommitted trial-040 key-variance changes. The owner subsequently selected separate Q/K/V alone for the general default; see [the September 21 promotion record](DEFAULT_PROMOTION.md).

The search ledger contains 38 audited candidates through trial 039, with two accepted additions. The fresh default baseline scored 10.977610% mean contact P@L; trial 011 scored 11.447751%, and accepted trial 031 scored 11.826297%. Both trial-040 workers completed successfully, scoring 11.559086% and 11.961092% (mean 11.760089%), but its campaign audit and ledger entry remain unfinished. Trial 031 is still the accepted incumbent. These measurements used one hour per seed and do not establish 100k-step performance.

All three runs completed successfully from step zero at 100,000 updates, global batch 1,024, with 102,400,000 sequence draws per run and no repeated source epochs. Training and evaluation workers exited with code 0. All ten scheduled P@L evaluations at 10k–100k passed, and every run's best checkpoint was its 100k endpoint. Final and best full checkpoints remain saved. Completion times in Toronto were September 20 at 11:47 p.m. for no-query-centering on kn086, September 21 at 1:42 a.m. for no-split-QKV on kn087, and September 21 at 2:06 a.m. for the full recipe on kn088. The Slurm allocations remain reserved through September 26. `COMPLETION_SUMMARY.json`, `COMPARISON.json` and each run's `VERIFIED.json` record the completed study.

## Final results

| Arm | Contact P@L | Difference from full, percentage points | Best step |
|---|---:|---:|---:|
| Full accepted recipe | 33.4851% | 0.0000 | 100,000 |
| No query centering / RMS restoration | 33.4498% | −0.0354 | 100,000 |
| No separate Q/K/V Muon | 32.7675% | −0.7177 | 100,000 |

The full recipe had the highest observed 100k P@L. Removing query centering and RMS restoration changed P@L by only −0.0354 percentage points, whereas removing separate Q/K/V Muon updates changed it by −0.7177 points. These are one-seed point estimates; paired significance and variability across training seeds have not been evaluated. Each final evaluation used all 20,775 contact chains and 5,000 chain-bootstrap replicates.

## Matched design

| Arm | Query centering + RMS restoration | Separate Q/K/V Muon | Placement |
|---|---|---|---|
| Full accepted recipe | On, final eight layers | On | kn088 |
| No query centering | Off | On | kn086 |
| No split QKV | On, final eight layers | Off | kn087 |

These are two component ablations with one shared full-recipe control. All three train from scratch with seed 42, 100,000 optimizer updates, global batch 1,024 (64 × 4 accumulation × 4 GPUs), context 512, peak base LR 0.0005, 1,000 warmup steps followed by constant LR, and the unchanged 36:11:54 source mixture. The long-run warmup matches the default long-run recipe; the one-hour search used 554. Each model has 170,559,856 parameters. This is a one-seed study and cannot estimate training-seed uncertainty.

P@L evaluation runs at updates 10,000, 20,000, …, 100,000 using 4,096 MLM-validation sequences, all 20,775 frozen contact chains, and 5,000 chain-bootstrap replicates. Training pauses during each intermediate evaluation and resumes afterward; evaluation time does not consume its training-time safety limit. The latest full recovery checkpoint is refreshed every 1,000 updates. Every strict improvement in P@L preserves a byte-verified full checkpoint, including optimizer and sampler state, under `best-checkpoints/`; `BEST_P_AT_L.json` records its score and step, and `checkpoint-best-p-at-l.pt` points to the current best. The final 100k checkpoint is retained independently and considered for best selection. The main matched comparison remains the final 100k endpoint, with best-over-training reported separately. Final verification requires all ten evaluations, correct best selection, full training, and no repeated source epoch. Production uses the ordinary training and evaluation APIs; the scientific training recipe and historical search results are unchanged.

## Launch verification and timing

The full-size periodic-evaluation preflight restored the 5,000-step checkpoint, evaluated at step 5,001, saved and byte-verified the best full checkpoint, resumed training, and completed step 5,002. It scored all 20,775 contact chains with 5,000 bootstrap replicates and took 334.8 seconds for evaluation. The retained checkpoint contains the model, composite Muon/AdamW optimizer and all four ranks' sampler/RNG states. The first preflight exposed the parallel evaluator's requirement for an empty output directory; the wrapper now gives it a fresh `results/` subdirectory. The failed preflight is preserved as `periodic-preflight-attempt-1`. Selection checks verified that lower/tied scores preserve the best and corrupted checkpoint bytes are rejected. Frozen scientific source was unchanged.

Measured training time was 35.47 hours for no-query-centering, 37.27 hours for no-split-QKV and 37.71 hours for full. Including setup, periodic and final evaluations, all results were ready within about 38 hours 48 minutes of the parallel launch. The final forecast predicted early Monday morning and all three completed by 2:06 a.m. Toronto time on September 21. Historical launch and ETA snapshots remain preserved for provenance.

The original two ablations stopped at 08:42:13 UTC on September 19 after both allocations encountered `NODE_FAIL`; they had reached updates 8,820 and 8,500, with last checkpoints at 5,000. Slurm restarted the allocations at 14:23:51 UTC. The subsequent recovery runs were explicitly stopped for this fresh launch, at logged updates 5,570 and 5,550. Their complete outputs remain under `attempts/<arm>/stopped-for-fresh-20260919T151818Z`; earlier failed-node attempts and periodic-evaluation preflights remain preserved. Previous launch metadata is under `launch-history/before-fresh-20260919T151818Z`. None of those checkpoints or sampler states is loaded by the current runs.

## Data

Each run requires 102.4 million draws. The original seven-shard corpus contains only 7,109,469 records and is insufficient. The expanded corpus is pinned to `LuminScience/LuminBench-Nano-ESMC` revision `bd38448d50d8f426d7b9bd4410b53159ea001259`, preserving its exact-duplicate and evaluation-homology exclusions.

| Source | Expected draws | Required with 5% headroom | Verified records | Shards |
|---|---:|---:|---:|---:|
| UniRef90 | 36,499,009.90 | 38,323,964 | 38,887,175 | 48 |
| MGnify | 11,152,475.25 | 11,710,100 | 12,934,081 | 9 |
| OMG/IMG | 54,748,514.85 | 57,485,944 | 58,339,702 | 54 |

The prepared corpus totals 110,160,958 unique training records across 111 shards, about 20.77 GB compressed. Required counts above include rounding for four ranks. Downloaded-shard hashes, materialized sequence hashes, source counts, prepared-corpus verification, and the standard per-arm data-capacity checks all passed before production started. The runtime `data_resampling: error` guard prevents silent wrapping. Sharing this corpus between arms is intentional; the no-repeat guarantee applies within each run.

## Execution and receipts

Remote study root: `/scratch/muchenli/Nano-ProteinLM/.worktrees/ar-260913-contact50/outputs/contact-ablations-100k-20260919`. Existing allocations are job 5218651 on kn086, job 5218652 on kn087 and job 5339730 on kn088, each reserving exactly four L40S GPUs under `aip-lsigal`, partition `gpubase_l40s_b5`. All three were confirmed `RUNNING` with an end time of September 26 at 14:23:51 UTC. Slurm, rather than this study, restarted them after node failures. The requested trainer relaunch signals only identified study `torchrun` processes; allocation holders and code-server processes remain running.

All three environments passed the four-L40S checks. The current `fresh-worker.sh` launches exactly one arm per node without a resume argument and refuses an existing output directory. `validate-fresh.py` verified all frozen source and orchestration hashes, data-manifest identity and the matched three-arm configuration before each launch. All 32 selected regression tests passed after adding two historical configuration fixtures omitted from the first source export; both failed export checks are preserved in `preflight-attempt-1` and `preflight-attempt-2`. Full-size smoke tests and data preparation passed before production. `PREFLIGHT_PASSED.json` verifies all three 20-step runs, finite metrics, parameter counts, and matched data/model-token/LR histories. `DATA_READY.json` was written atomically after materialization and all capacity checks passed. The local source archive matches all 77 entries in `SOURCE_SHA256.json`.

Use `status.py` on CCK for a compact snapshot. Per-node worker logs, per-arm training/evaluation logs, `REGRESSION.json`, `SOURCE_SHA256.json`, `STUDY.json`, and per-run verified results remain under the study root. `COMPARISON.json` has been produced after all three 100k-step runs and their ten evaluations passed verification. `STATUS.json`, when present beside this report, is a copied point-in-time snapshot, not a live status endpoint.
