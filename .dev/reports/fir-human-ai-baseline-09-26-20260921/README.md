# human+ai baseline-09-26: H100 search benchmark

Three-seed confirmation of the current default recipe under the H100 search budget. The user requested three 20-minute runs on four H100 GPUs each, submitted to Fir with a three-hour job limit. This is one recipe repeated across seeds, not a three-setting scale-up submission. Report validation loss and P@L per seed and as mean ± sample standard deviation.

| Item | Contract |
|---|---|
| Source | `43f2996729e6b41a998de92674340217afacfd5b`; exact source hashes in `PLAN.json`, matching the CCK study |
| Recipe | Current `configs/default.yaml`, separate Q/K/V Muon updates enabled; no query centering/RMS restoration |
| Training | From scratch, 1,200 training seconds, four H100 GPUs, FA3, context 512, global batch 1,024 |
| Task controls | Seeds 42/43/44; warmup 554; constant peak LR afterward; no step/token cap; no periodic checkpoint/evaluation |
| Data | 7,109,469 training proteins; all 12 training/validation token-store files identical to CCK; local copies hash-verified before each run; no resampling |
| Evaluation | Final checkpoint; 32 MLM validation sequences; 20,775 contact chains; 5,000 bootstrap replicates |
| Training allowance | 4/3 H100 GPU-hours per seed, 4 H100 GPU-hours total; setup/evaluation separate |
| Slurm limit | Three hours per array job, including preparation, training, checkpoint writing and evaluation |
| Array | `60898535`, account `def-lsigal_gpu`; scheduler-selected `gpubase_bynode_b1,gpubackfill` |
| Artifacts | `/scratch/muchenli/Nano-ProteinLM-benchmarks/human-ai-baseline-09-26-h100-20260921` on Fir |

| Seed | Array task | Validation loss ↓ | P@L ↑ | Status |
|---|---|---|---|---|
| 42 | 60898535_0 | Pending | Pending | Queued |
| 43 | 60898535_1 | Pending | Pending | Queued |
| 44 | 60898535_2 | Pending | Pending | Queued |
| Mean ± sample SD | — | Pending | Pending | Awaiting three completed evaluations |

`SUBMISSION.json` records the accepted request. The initial explicit partition combination was rejected before creating a job; omitting the partition as Fir's submission plugin recommended succeeded, and Slurm routed the job to its appropriate three-hour/backfill partitions. Every task retains `gpu:h100:4` and `03:00:00`. No prior allocations were canceled or changed.

`submit.sbatch` launches one worker (`worker.sh`) per seed. `preflight.py` verifies source/data hashes and conservative capacity for 5,000 optimizer updates with 5% headroom, without imposing that as a training step limit. `verify.py` checks the completed time budget, no source recycling and final-checkpoint/evaluation identity. The final successful worker aggregates the three seeds with the standard summarizer. `status.py` reads progress and completed results.

H100 uses the default FA3 backend; the CCK L40S profile uses FA2. The two hardware profiles have separate search-reward tables and different training-time allowances. The token-store contents and scientific evaluation are matched, while each cluster retains its own verified download-plan/manifest receipt. The approved follow-up checks both profiles every 15 minutes, reports each when complete, and stops after both results are delivered.

Before training started, the verifier was corrected to read the completion receipt’s stage-indexed source-epoch counters. This changes only verification, preserving the training recipe, time budget and zero-repetition requirement. See `VERIFIER_CORRECTION.json`.

This page publishes the study summary. Detailed receipts, launch scripts and verification helpers named above are retained in the remote study directory listed under Artifacts and in the local report directory; they are not included in this documentation-only publication.
