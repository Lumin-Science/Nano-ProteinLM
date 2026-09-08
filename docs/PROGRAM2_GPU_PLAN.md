# Four-setting 100k H100 training plan

Status: **all four settings completed their 100k runs and full evaluations**.
Setting 4 finished at **September 8, 9:58:59 AM Toronto**. All four H100 technical trials
passed. Setting 1 ran as `58303658.14`; the completed sequential settings 2–4 queue was
`58303724.9`. The user authorized
settings **2 → 3 → 4 sequentially on fc10212** after qualification, and a
September 7, 2026 **4:00 AM Toronto** check to launch setting 1 on fc10111 if
its GPUs are free. This supersedes the earlier proposed two-node/two-wave plan.
The [launch record](../reports/fir-r02-rope10k-100k-20260906/README.md) holds
trial receipts, actual launch identities, and timestamped production state.

## Four independent runs

Each recipe trains **from scratch for 100,000 optimizer steps**. The sequence
below is cumulative in configuration only; no run resumes another checkpoint.
All use our completed R02 architecture with RoPE reset from 20k to **10k**.

| Setting | Config | Added change | Parameters |
|---|---|---|---:|
| 1. R02-RoPE10k | [r02_rope10k.yaml](../configs/archive/program2_h100_100k/r02_rope10k.yaml) | Completed R02 recipe, RoPE 10k | 170,559,856 |
| 2. + batch balance | [r04_batchbalance.yaml](../configs/archive/program2_h100_100k/r04_batchbalance.yaml) | Balance the same masked examples across ranks | 170,559,856 |
| 3. + sqrt loss | [r10_sqrtloss.yaml](../configs/archive/program2_h100_100k/r10_sqrtloss.yaml) | Square-root masked-target weighting | 170,559,856 |
| 4. + tied embeddings | [r29_tied.yaml](../configs/archive/program2_h100_100k/r29_tied.yaml) | Share input/output vocabulary weights | 170,510,704 |

Shared architecture: **24 layers, width 768, 12 heads, SwiGLU FFN width 2048,
parameter-free RMSNorm, learned residual/input routing, depth-scaled residual
projection initialization**. The prediction-head LayerNorm remains. R22
narrower FFNs are on [TODO](../TODO.md), outside these four runs.

## Common training contract

| Setting | Value for every run |
|---|---|
| Hardware | One node, four H100 80GB GPUs; four DDP ranks |
| Batch | 1,024 sequences = 64/GPU × 4 GPUs × 4 accumulation microsteps |
| Context | 512 |
| Update and schedule budget | `max_steps = schedule_steps = 100000` |
| Warmup / schedule | 1,000 optimizer steps, then constant LR |
| Base learning rate / weight decay | 0.0005 / 0.01 |
| Attention Muon configured LR / WD | 0.00045 / 0.0075 |
| FFN Muon configured LR / WD | 0.000375 / 0.0075 |
| Decayed AdamW LR / WD | 0.0005 / 0.01 |
| Non-decayed AdamW LR / WD | 0.0005 / 0 |
| Muon | Momentum 0.95, Nesterov, 5 NS steps, `match_rms_adamw` |
| AdamW | Betas (0.9, 0.95), epsilon 1e-8 |
| Gradient clipping | Global norm 1.0 |
| Seed | 20260824, matching the completed comparison |
| Execution | BF16 mixed precision, pinned FA3, no compile/checkpointing |
| Mixture | UniRef90 0.36, MGnify 0.11, OMG/IMG 0.54 (normalized by sampler) |
| Training guard | 57,600 seconds (16 hours), not the intended scientific budget |
| Sequence exposure | 102.4M sampled sequences per run |

Configured Muon LRs precede its internal matrix-shape adjustment. Sqrt loss
retains the imported implementation: normalize over the four ranks of each
256-example microstep, then average four gradients per update. Evaluation uses
the original sequence-mean loss. See [full recipe semantics](PROGRAM2_SCALEUP.md).

Use the same verified corpus as the completed runs, with manifest SHA-256
`43675d51421066ce8c5f68427886d57980e808c53c5bb1641de90cb74dda39ab`.
Freeze the new source commit in a fresh remote checkout and use unique output
directories named for the method and launch timestamp. The completed 100k
source checkout and outputs are historical references.

## Authorized placement and scheduling

| Fir node | Existing allocation | Assigned work |
|---|---|---|
| `fc10212` | `58303724` | Trial all four recipes, then full settings 2 → 3 → 4 sequentially |
| `fc10111` | `58303658` | Full setting 1, after the September 7 4:00 AM Toronto availability check |

At September 6, 7:35 PM Toronto, fc10212 had four idle H100s and about 53h 51m
remaining. That allows three 16-hour guards plus setup/evaluation margin. A
separate workload occupied all four fc10111 GPUs. It must finish or be stopped
by its owner before setting 1 can start; these launchers never stop it. Check
both Slurm and `nvidia-smi`, since that workload had no separate Slurm step.

The scheduled heartbeat initially ran **hourly on the hour**, including September 7 at
**04:00 America/Toronto = 08:00 UTC**. Its task is to monitor all four runs and
launch setting 1 at or after that timestamp once the node is free. The launcher
also enforces that timestamp. A busy node at 4 AM defers launch to a later hourly
check. Setting 1 starts from scratch and can run independently of the fc10212
queue. No additional allocation was requested for these Fir runs. All four are
now complete; the heartbeat continues every **two hours** for the separate
[Nibi baseline](../reports/nibi-baseline-b2048-100k-eval10k-20260908/README.md).

## Qualification and production queue

The frozen training source is
`253c3ea442f2a1657eeb0f3ce2127ac0b1adfb25`, in
`/scratch/muchenli/Nano-Protein-LM-r02-rope10k-100k-20260906-run`.
Artifacts and launchers are in the same path without the `-run` suffix.

1. Run each exact model/batch combination for **200 optimizer steps**, with a
   temporary **50-step warmup** and 200-step schedule. These technical trials
   exercise peak learning rates, four-rank accumulation, FA3 forward/backward,
   finite parameters/losses/gradients, balanced-token preservation, optimizer
   ownership and checkpoint/attention-feature reload. Each trial also runs a
   32-sequence MLM smoke evaluation. These losses are diagnostic, not a quality
   ranking or a substitute for full evaluation.
2. Require all four trial receipts and their config/checkpoint bindings to pass
   before writing `ALL_TRIALS_PASSED.json`. The production launchers refuse to
   start without that gate. Every full run restores the exact **100k-step,
   1,000-step-warmup** preset and initializes from scratch.
3. Run `queue-234.sh` in allocation 58303724 on fc10212. It trains setting 2,
   verifies completion, performs full MLM/P@L, then repeats for settings 3 and
   4. A failed stage stops the queue and preserves its logs. Unique output
   directories and the queue guard prevent accidental duplicate launches.
4. Before each full run, require idle GPUs, matching source/config hashes and
   at least **16h 15m** of allocation time. Keep the tested Torch 2.13.0+cu130
   and pinned FA3 environment; stage and verify the same corpus on node-local
   storage. The 16-hour training guard remains an operational limit, not a
   replacement for 100k steps.
5. Monitor at the user's requested cadence (initially hourly, now every two hours). Notify on launch, failure, stall, completion or required
   action, and update Toronto ETAs from measured progress. Healthy unchanged
   runs do not need repetitive notifications.

## Completion and evaluation

Require **100,000 optimizer steps**, **102.4M sequences**, `stop_reason=max_steps`
and a verified final checkpoint hash. Hitting the wall-time guard is incomplete
for this comparison. Then run the same **4,096-sequence MLM evaluation**
(256 × 16, context 512, seed 20260821) and **20,775-chain P@L** used for the
historical default/R02 comparison. Each checkpoint gets its own canonical
probe fit and 16 deterministic inference shards, followed by chain-coverage,
checkpoint-binding and shard-hash verification and a 5,000-resample bootstrap CI.

Publish the four results alongside the completed default and RoPE20k R02,
including each adjacent setting's delta. There is one matched training seed;
contact bootstrap intervals do not measure training-seed variability.

## Initial time and compute estimate

The completed RoPE20k R02 took **12h 56m 30s** (0.4659 s/step) on four H100s.
Use **13–14 hours of training per setting** until the new trials/live runs
provide measured estimates. Settings 2–4 therefore require about **39–42 hours**
plus setup and evaluation on fc10212. Setting 1 should finish around **5–6 PM
Toronto September 7** if it starts at 4 AM and has similar speed. These are
planning estimates, not fixed finish times.

Total training usage is roughly **208–224 H100 GPU-hours** for all four full
runs, plus trials/setup/evaluation. New trial timings and live ETAs are recorded
in the [launch record](../reports/fir-r02-rope10k-100k-20260906/README.md).
