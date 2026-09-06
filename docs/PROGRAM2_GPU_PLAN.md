# Four-setting 100k H100 training plan

Status: **prepared, not submitted or launched**. Resource snapshot:
September 6, 2026, **7:12 PM Toronto**. This plan supersedes the unlaunched
five-setting Program 2 architecture proposal.

## Four independent runs

Each recipe trains **from scratch for 100,000 optimizer steps**. The sequence
below is cumulative in configuration only; no run resumes another checkpoint.
All use our completed R02 architecture with RoPE reset from 20k to **10k**.

| Setting | Config | Added change | Parameters |
|---|---|---|---:|
| 1. R02-RoPE10k | [r02_rope10k.yaml](../configs/program2_h100_100k/r02_rope10k.yaml) | Completed R02 recipe, RoPE 10k | 170,559,856 |
| 2. + batch balance | [r04_batchbalance.yaml](../configs/program2_h100_100k/r04_batchbalance.yaml) | Balance the same masked examples across ranks | 170,559,856 |
| 3. + sqrt loss | [r10_sqrtloss.yaml](../configs/program2_h100_100k/r10_sqrtloss.yaml) | Square-root masked-target weighting | 170,559,856 |
| 4. + tied embeddings | [r29_tied.yaml](../configs/program2_h100_100k/r29_tied.yaml) | Share input/output vocabulary weights | 170,510,704 |

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

## Available resources and recommended placement

| Fir node | Existing allocation | Observed GPU state | Remaining allocation time | Planning use |
|---|---|---|---|---|
| `fc10212` | `58303724` | Four idle H100s, 5 MiB/GPU, 0% utilization | About 54h 14m | Available for two runs |
| `fc10111` | `58303658` | Four H100s busy, about 44,043 MiB/GPU, 98–100% utilization | About 54h 13m | Unavailable for this plan now |
| `fc10110` | `58047089` | GPU occupancy not checked | About 6h 49m | Insufficient time for a 100k run |

The `fc10111` workload was visible in `nvidia-smi` even though the Slurm step
table contained only helper steps. Check both sources immediately before any
future launch. Node availability is a snapshot, not a reservation.

**Recommended: use `fc10212` and request one additional four-H100 Fir node.**
The current matching partition is `gpubase_bynode_b4` (maximum 72 hours), with
account `rrg-lsigal_gpu`. Plan a **48-hour allocation**, one node, four H100s,
and 32 training CPUs on the additional node. This request is a proposal and
has not been submitted. Do not assign `fc10111` while its GPUs are occupied.

| Wave | `fc10212`, four H100s | Additional idle node, four H100s | Expected duration |
|---|---|---|---|
| 1 | Setting 1: R02-RoPE10k | Setting 2: + batch balance | About 13–14h, then evaluation |
| 2 | Setting 3: + sqrt loss | Setting 4: + tied embeddings | About 13–14h, then evaluation |

Start each second-wave run only after that node's first run has saved its final
checkpoint and completed evaluations. Since all runs start from scratch, the
additional node can begin its first run as soon as it becomes available; it
does not need to wait for a checkpoint from `fc10212`.

## Time and compute estimate

The completed R02 measured **46,589.86 seconds / 100,000 steps = 0.4659 s/step**,
or **12h 56m 30s** on four H100s. RoPE 10k leaves model dimensions unchanged;
batch redistribution, loss normalization and tying need measured timing before
their exact speeds are known. Use **13–14 hours of training per run** as the
initial planning range, not a benchmark result for the new recipes.

- **Two four-H100 nodes:** roughly **26–28 hours**, plus setup, evaluations and
  any additional-node queue delay. Allow about **27–30 hours once both nodes
  are available** for the operational plan. Two 16-hour guards leave room within
  the proposed 48-hour allocation.
- **Total training allocation usage:** about **208–224 H100 GPU-hours** for all
  four runs, before setup/evaluation. The previous R02 runtime alone projects
  about 207 GPU-hours; 208–224 is the rounded planning range.
- **One four-H100 node:** about **52–56 hours**, plus overhead. The remaining
  roughly 54 hours on `fc10212` leave insufficient reliable margin for all four
  sequential runs; use a longer fresh allocation or a second node.
- **Four four-H100 nodes:** about **13–14 hours**, plus overhead and queueing,
  but requires three additional nodes (16 H100s total).

The previous final MLM and full contact evaluations took minutes, rather than
hours. Reserve additional time for environment/data staging and final checkpoint
writes. Refresh the estimates from a short full-size timing qualification and
then from live 100k-run progress; do not change the 100k cap to fit a wall clock.

## Launch and evaluation sequence

1. Recheck allocation expiry, GPU occupancy, source/config hashes and local data
   receipts; qualify pinned FA3 on each node. Run short full-size checks with
   four ranks and four accumulation microsteps for all four configs, including
   finite gradients, preserved masks, tied-weight ownership and measured speed.
   These checks are planned; they have not been run on H100 for this revision.
2. Start fresh runs for the first wave, using the tested Torch 2.13.0+cu130 / FA3
   environment. Require successful loss/gradient progress before leaving them.
3. Keep the user's hourly monitoring cadence once runs launch. Report failures,
   stalls, and completions; update Toronto finish estimates from measured speed.
4. For every completed checkpoint require exactly **100,000 steps**, **102.4M
   sequences**, `stop_reason=max_steps`, and a saved checkpoint hash. A run that
   hits the guard is incomplete for this comparison.
5. Immediately run the same **4,096-sequence MLM evaluation** (256 batches × 16,
   context 512, evaluation seed 20260821) and **20,775-chain P@L**. Fit each
   checkpoint's probe on the same fixed structures; reuse the verified contact
   dataset and merge all shards. Preserve checkpoint-bound receipts and contact
   bootstrap intervals.
6. Advance to the second wave on each available node. Publish a table comparing
   all four runs with the completed default and RoPE20k R02. Report each adjacent
   configuration's delta. This first comparison has one matched training seed;
   repeated seeds are a later robustness check.

No new training, allocation request or monitoring automation has been started
by preparing this plan. The previous completed-run monitor remains paused.
