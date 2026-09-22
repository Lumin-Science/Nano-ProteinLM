# human+ai baseline-09-26

Three-seed confirmation of the current default recipe at the L40S search budget, requested September 21, 2026. This is one frozen recipe repeated across seeds, not a three-setting scale-up submission. All three runs and their final evaluations are complete. Validation loss is **2.589492 ± 0.009195** and P@L is **11.771370% ± 0.256263 percentage points**, reporting mean ± sample standard deviation across seeds 42, 43 and 44.

| Item | Contract |
|---|---|
| Source | `43f2996729e6b41a998de92674340217afacfd5b`; source-file hashes in `PLAN.json` |
| Recipe | `configs/default.yaml`; separate Q/K/V Muon updates enabled; no query centering/RMS restoration |
| Training | From scratch; 3,600 training seconds per seed; four L40S GPUs; FA2; context 512; global batch 1,024 |
| Task overrides | Seeds 42/43/44, warmup 554, constant peak LR afterward, no step/token cap, no periodic evaluation/checkpointing |
| Data | Original seven-shard benchmark corpus, 7,109,469 training proteins; identical hash-verified node-local copies; no resampling |
| Evaluation | Final checkpoint; 32 MLM validation sequences; all 20,775 contact chains; 5,000 chain-bootstrap replicates |
| Cost | 12.001692 L40S GPU training hours; 1.350473 allocated GPU-hours for final evaluation; setup separate |
| Artifacts | `/scratch/muchenli/Nano-ProteinLM/outputs/human-ai-baseline-09-26-20260921` on CCK |

| Seed | Allocation | Node | Validation loss ↓ | P@L ↑ (%) | Status |
|---|---:|---|---:|---:|---|
| 42 | 5218651 | kn086 | 2.591942 | 11.774792 | Complete |
| 43 | 5218652 | kn087 | 2.579320 | 11.513413 | Complete |
| 44 | 5339730 | kn088 | 2.597214 | 12.025905 | Complete |
| **Mean ± sample SD** | — | — | **2.589492 ± 0.009195** | **11.771370 ± 0.256263** | **3/3 seeds** |

`worker.sh` runs data verification, training, final evaluation and per-seed verification. The last successful worker writes `summary.json` using the standard `nanoprotein.summarize_training_runs` API. The preflight (`preflight.py`) checks source hashes, all copied training/validation token-store digests, and per-source capacity for a conservative 5,000-update workload with 5% headroom. That capacity check does not add a step cap to the one-hour runs. `verify.py` checks completed time budgets, batch/sample counts, no source recycling and final-checkpoint/evaluation identity before accepting a result.

The current default's mathematical settings remain fixed across seeds. Execution overrides match the existing L40S search task. The training corpus is reused independently across training seeds; sufficient capacity is required within each run, not disjoint data between runs. Chain-bootstrap confidence intervals and across-training-seed sample SD are reported separately.

Healthy training was confirmed at `2026-09-22T01:33:23Z` (September 21, 9:33 p.m. Toronto time). All three workers passed source/data verification and completed at least 20 updates, with no source recycling. All three training endpoints were reached at approximately 10:33 p.m. Toronto time; final evaluation subsequently completed. `STATUS.json` records the latest checked progress. The first preflight attempt stopped before training because its expected manifest hash came from the Fir copy; all 12 token-store digests matched. The corrected plan pins CCK’s verified manifest, and the original preflight logs remain archived on CCK.

All three seeds completed their 3,600-second training budgets at approximately 10:33 p.m. Toronto time. The original verifier stopped before evaluation because it interpreted the completion receipt’s stage-indexed source-epoch dictionaries as numeric counters. The corrected verifier checks every stage and source explicitly; all counters are zero. Evaluation-only recovery uses the preserved final checkpoints, with no replacement training. The original verifier and failure logs remain archived; see `VERIFIER_CORRECTION.json` and `evaluate-saved.sh`.

## Completed-run evidence

The aggregate summary (`summary.json`) was independently checked against each training/evaluation receipt; `SUMMARY_VERIFICATION.json` records the checks, exact values and SHA-256 digests. Each seed’s P@L was recomputed as the mean of all 20,775 chain rows, and both metric means and sample SDs were recomputed across the three training seeds. The completion receipts show fresh runs, matched configurations apart from seed, the required training-time endpoints and zero source repetition. Across-seed SD uses the sample denominator N−1.

| Seed | Updates | Non-padding model tokens | Training seconds | Evaluation seconds | P@L chain-bootstrap 95% CI (%) | Receipts |
|---|---:|---:|---:|---:|---|---|
| 42 | 2,837 | 686,194,396 | 3600.046 | 258.979 | 11.682490–11.865806 | Training (`receipts/seed-42/TRAINING_COMPLETE.json`), evaluation (`receipts/seed-42/EVALUATION.json`), verification (`VERIFIED-seed-42.json`), preflight (`DATA_PREFLIGHT-seed-42.json`) |
| 43 | 2,824 | 683,351,599 | 3600.844 | 472.601 | 11.422142–11.601852 | Training (`receipts/seed-43/TRAINING_COMPLETE.json`), evaluation (`receipts/seed-43/EVALUATION.json`), verification (`VERIFIED-seed-43.json`), preflight (`DATA_PREFLIGHT-seed-43.json`) |
| 44 | 2,832 | 685,340,104 | 3600.633 | 483.846 | 11.930408–12.119306 | Training (`receipts/seed-44/TRAINING_COMPLETE.json`), evaluation (`receipts/seed-44/EVALUATION.json`), verification (`VERIFIED-seed-44.json`), preflight (`DATA_PREFLIGHT-seed-44.json`) |

Final evaluation used 32 sequences (994 masked targets) for MLM and the same ordered 20,775 chains for contact P@L. Chain-bootstrap intervals describe uncertainty over evaluated chains; the main table’s SD describes training-seed variation. Evaluation GPU-hours above are elapsed evaluator time multiplied by the four allocated GPUs, not measured GPU utilization. These are L40S search-budget measurements; no iso-token scale-up result is claimed.

This page publishes the study summary. Detailed receipts, launch scripts and verification helpers named above are retained in the remote study directory listed under Artifacts and in the local report directory; they are not included in this documentation-only publication.
