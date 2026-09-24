# human+ai baseline-09-26: H100 search benchmark

Three-seed confirmation of the current default recipe under the H100 search budget. The user requested three 20-minute runs on four H100 GPUs each, initially submitted to Fir with a three-hour job limit and subsequently launched in the user-provided allocation on `fc10219`. This is one recipe repeated across seeds, not a three-setting scale-up submission. Report validation loss and P@L per seed and as mean ± sample standard deviation.

| Item | Contract |
|---|---|
| Source | `43f2996729e6b41a998de92674340217afacfd5b`; exact source hashes in `PLAN.json`, matching the CCK study |
| Recipe | Current `configs/default.yaml`, separate Q/K/V Muon updates enabled; no query centering/RMS restoration |
| Training | From scratch, 1,200 training seconds, four H100 GPUs, FA3, context 512, global batch 1,024 |
| Task controls | Seeds 42/43/44; warmup 554; constant peak LR afterward; no step/token cap; no periodic checkpoint/evaluation |
| Data | 7,109,469 training proteins; all 12 training/validation token-store files identical to CCK; local copies hash-verified before each run; no resampling |
| Evaluation | Final checkpoint; 32 MLM validation sequences; 20,775 contact chains; 5,000 bootstrap replicates |
| Training allowance | 4/3 H100 GPU-hours per seed, 4 H100 GPU-hours total; setup/evaluation separate |
| Slurm limit | Three hours for the sequential benchmark step, including preparation, training, checkpoint writing and evaluation; the enclosing user allocation is retained |
| Execution | Step `60039326.1` on `fc10219`, account `rrg-lsigal_gpu`; seeds 42 → 43 → 44 sequentially on all four H100s |
| Original queued array | `60898535`, account `def-lsigal_gpu`; left unchanged in its original study directory |
| Artifacts | `/scratch/muchenli/Nano-ProteinLM-benchmarks/human-ai-baseline-09-26-h100-20260921-fc10219` on Fir |

| Seed | Steps | Validation loss ↓ | P@L ↑ | Status |
|---|---:|---:|---:|---|
| 42 | 2,563 | 2.613388 | 11.497828% | Complete; independently verified |
| 43 | 2,525 | 2.588710 | 11.087363% | Complete; independently verified |
| 44 | 2,538 | 2.602679 | 11.036266% | Complete; independently verified |
| Mean ± sample SD | 2,542 mean | **2.601592 ± 0.012375** | **11.207152% ± 0.253026 pp** | Three seeds |

`SUBMISSION.json` records the accepted request. The initial explicit partition combination was rejected before creating a job; omitting the partition as Fir's submission plugin recommended succeeded, and Slurm routed the job to its appropriate three-hour/backfill partitions. Every task retains `gpu:h100:4` and `03:00:00`. No prior allocations were canceled or changed.

The allocation step started on 2026-09-22 at 19:10 UTC after a Slurm GPU probe confirmed all four H100s were idle. `run-sequential.sh` invokes `worker.sh` for seeds 42, 43 and 44 in order and stops on a failure. `ALLOCATION_LAUNCH.json` records the allocation and step; `queued-array-60898535/` preserves the original local launch metadata. The existing helper and queued array remain unchanged; the latter can still start duplicate work until its owner cancels it.

The original `submit.sbatch` launches one worker per array task in the original directory; the active allocation worker uses the separate directory above. `preflight.py` verifies source/data hashes and conservative capacity for 5,000 optimizer updates with 5% headroom, without imposing that as a training step limit. `verify.py` checks the completed time budget, no source recycling and final-checkpoint/evaluation identity. The final successful worker aggregates the three seeds with the standard summarizer. `status.py` reads progress and completed results.

H100 uses the default FA3 backend; the CCK L40S profile uses FA2. The two hardware profiles have separate search-reward tables and different training-time allowances. The token-store contents and scientific evaluation are matched, while each cluster retains its own verified download-plan/manifest receipt. The CCK results were reported earlier. This completes the Fir profile; the approved follow-up stops after reporting these results.

Before training started, the verifier was corrected to read the completion receipt’s stage-indexed source-epoch counters. This changes only verification, preserving the training recipe, time budget and zero-repetition requirement. See `VERIFIER_CORRECTION.json`.

This page publishes the study summary. Detailed receipts, launch scripts and verification helpers named above are retained in the remote study directory listed under Artifacts and in the local report directory; they are not included in this documentation-only publication.

## Completed-run verification

All three seeds completed training and evaluation by **2026-09-22 20:31:08 UTC** (16:31 Toronto time). Step `60039326.1` completed successfully after 1:20:59, including preparation and evaluation. Its enclosing allocation and original pending array remain unchanged. Actual training consumed **4.000974 H100 GPU-hours**; final evaluation accounted for another **0.820892 allocated GPU-hours**, excluding environment/data preparation and other launcher overhead.

`SUMMARY_VERIFICATION.json` independently checks `summary.json` against the downloaded completion, preflight and evaluation receipts for every seed. Source and data hashes match the frozen plan; configurations differ only by seed; every run starts fresh and completes 1,200 training seconds without recycling data. Each evaluation matches its final checkpoint hash, and every P@L was recomputed from all 20,775 unique chain rows. The 32-sequence MLM protocol and ordered contact chains match the CCK study. Both aggregate means and sample SDs use all three seeds, with SD denominator N−1.

The results are recorded in the [H100 reference leaderboard](../../../docs/LEADERBOARD.md#search-budget). These are repetitions of one recipe at the 20-minute search budget; they do not supply three-setting scale-up results. Detailed receipts and the independent verification helper remain local and in the remote study directory.
