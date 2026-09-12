# Nibi Setting 3 Stage 2 — batch 2,048, 300k additional updates

Production launched September 11, 2026 at 17:52 Toronto on Nibi g27, Slurm step **12162637.44**, using all eight H100 80GB GPUs. The [live launch audit](LAUNCH_VERIFIED.json) passed at global step **400,530** at 17:58 Toronto: all logged losses and gradients were finite, optimizer and source history were restored, the intended decay clock was active, and the recipe/source bundle matched their project-storage copies. The observed training-loop rate was **0.556 seconds per update**. Including 30 evaluations at the measured qualification cost, the initial finish estimate was **September 13 at 18:00 Toronto**, or **22:38** with a 10% training slowdown. This is a launch-time estimate, not a completion result. The current allocation ends September 14 at 07:24:03 Toronto.

## Production evaluation results

**8 of 30 production Stage 2 evaluations are verified**, through global **480k**. Each [independent audit](full/evaluations/step-480000/LOCAL_AUDIT.json) checks checkpoint bindings, all 16 contact shards, all 20,775 historical chain identities, the unchanged MLM/probe protocols, and an independently recomputed 5,000-replicate bootstrap. The [machine-readable curve](learning-curve.json) excludes qualification trials.

| Checkpoint | Stage 2 updates | Validation loss ↓ | P@L ↑ | 95% chain-bootstrap CI |
|---|---:|---:|---:|---:|
| Stage 1 parent, 400k | 0 | 2.308892 | 42.241% | 41.992–42.495% |
| Stage 2, 410k | 10,000 | 2.306547 | 42.778% | 42.530–43.034% |
| Stage 2, 420k | 20,000 | 2.302875 | 43.023% | 42.776–43.274% |
| Stage 2, 430k | 30,000 | 2.302230 | 43.303% | 43.050–43.560% |
| Stage 2, 440k | 40,000 | 2.300549 | 43.587% | 43.339–43.840% |
| Stage 2, 450k | 50,000 | 2.297940 | 43.641% | 43.389–43.894% |
| Stage 2, 460k | 60,000 | 2.297293 | 43.762% | 43.514–44.019% |
| Stage 2, 470k | 70,000 | 2.294548 | 43.998% | 43.749–44.253% |
| Stage 2, 480k | 80,000 | 2.293236 | 44.225% | 43.974–44.477% |

Relative to 400k, validation loss changed by **-0.015656**, and P@L changed by **+1.984 percentage points**. The paired chain-bootstrap 95% interval for this P@L change is **+1.932 to +2.039 points**, using the same chains and 5,000 replicates. These intervals quantify uncertainty across evaluation chains, not variation across training seeds.

From 470k to 480k, validation loss changed by **-0.001312** and P@L by **+0.228 points**.

The latest independently checked training metrics reached global **482,100**. MGnify had no repeats, and the fixed Stage 2 decay schedule remained correct. The latest evaluation took **205.6 seconds**.

## Latest monitoring check

At **September 12, 07:02 Toronto**, the [monitoring receipt](monitoring/20260912T1102Z.json) recorded global **482,020**, or **82,020 / 300,000 Stage 2 updates (27.34%)**, with no production failure. Throughput was **0.555 seconds per update** since 470k. The allocation had **48.37 hours remaining**; estimated completion was **September 13 at 17:53 Toronto**, or **September 13 at 21:15** with a 10% training slowdown. MGnify had **31,420,895 unused records** versus **26,785,382 expected remaining draws**, with headroom of **17.31%**.

## Earlier monitoring check

At **September 11, 19:00 Toronto**, the [monitoring receipt](monitoring/20260911T2300Z.json) recorded **407,270 global steps**, or **7,270 / 300,000 Stage 2 updates (2.42%)**. Production step 12162637.44 remained active on g27, the source checkout was unchanged, and finite loss/gradient, decay schedule and source accounting checks passed. No production validation or P@L endpoint was due yet; the first 410k evaluation was estimated to finish around **19:29 Toronto**.

Training averaged **0.555 seconds per update** since the launch audit. Including the qualification-measured evaluation overhead, the revised finish estimate was **September 13 at 17:56 Toronto**, or **22:27** with a 10% training slowdown. The allocation had **60.39 hours remaining**, leaving **13.46 hours** of headroom at the central estimate. MGnify had **40,603,251 unused records**, **zero repeats**, and approximately **12.88% headroom** over its remaining expected draws. UniRef90 was in zero-indexed epoch 4 and OMG in epoch 1, consistent with their permitted complete global passes.

## Frozen recipe

The [recipe](../../../configs/setting3-nibi-stage2-b2048-300k.yaml), [transition methods](../../../docs/stage2-continuation.md), and [launcher](../../../runs/nibi_setting3_stage2.sh) are preserved on main. The production source is pinned separately at **f4b672d0a3a70e15740ee3876c202e535f727d3c**, so later report updates do not change running code.

| Setting | Value |
|---|---|
| Model | Setting 3, 170,559,856 parameters |
| Budget | 300,000 additional updates; global 400,000 → 700,000 |
| Hardware | Eight H100 80GB GPUs; BF16 and FlashAttention-3 |
| Global batch | 2,048 = microbatch 64 × 8 GPUs × accumulation 4 |
| Context | 2,048 |
| UniRef90 / MGnify / OMG | 63% / 6% / 31% |
| Optimizer and architecture | Existing Muon/AdamW groups, RMSNorm, learned residual routing, depth-scaled initialization, RoPE 10k, FFN 2,048, untied embeddings |
| Training objective | Batch balancing and sqrt-mask-count loss weighting |
| Base LR | Linear decay from 5e-4 to 5e-5 over Stage 2; no restarted warmup |
| Muon LR multipliers | Attention 0.9; FFN 0.75 |
| Weight decay | Base 0.01; Muon multiplier 0.75 |
| Checkpoint / evaluation interval | Every 10,000 global updates |

Production evaluations occur at global 410k, 420k, …, 700k, for 30 endpoints. The first 410k result was initially estimated for September 11 around 19:30 Toronto. Validation retains the historical 512-token protocol, 4,096 sequences and 139,963 masked residues. P@L retains all 20,775 chains, 16 component shards and 5,000 chain-bootstrap replicates. Qualification results below are separate from the production curve.

## Data expansion and continuation

The [data plan](DATA_PLAN.json) added **40,221,714 MGnify records in 28 shards**, downloading **5,727,839,735 bytes** from the same immutable release. The verified dataset now contains **467,737,935 training records**: 74,175,974 UniRef90, 130,716,775 MGnify and 262,845,186 OMG/IMG. The [readiness receipt](DATA_READY.json) proves the stored release and unchanged validation contract. Its manifest SHA-256 is `76bf671104f795af68594110010c64e047dd75d4254cb5508d02a8e0407d32b2`.

At the transition, MGnify had **41,496,700 unused records**, compared with **36,864,000 expected additional draws** and **38,707,200 required with 5% headroom**. The runtime refuses MGnify repeats. UniRef90 and OMG continue their existing permitted global passes. Context expansion uses the full sequences already represented in the token stores; it does not change the tokenizer or held-out data.

The [transition receipt](transition/checkpoint-stage2-start.json) binds the original 400k checkpoint, both datasets and every existing record/token prefix. It preserves the exact model and optimizer tensors, counters and RNG state, confirmed by the [independent optimizer restoration audit](transition/TRAINING_VERIFIED.json). The expanded MGnify queue retains its prior global and legacy histories through a nested origin; its new cursor begins at zero but its cumulative consumed count remains 89,220,075. Existing UniRef90 and OMG epoch/cursor values are preserved.

| Checkpoint | SHA-256 |
|---|---|
| Original Stage 1 400k | `c205ca56a3c8ecd47a5985444d2d012a2a4e5fa38de8b1e58469619b7eff6dc2` |
| Full-state Stage 2 start | `f3ebbfd3a38abec4d5e70c96722240c1549342959b5735f130a3c9d9b4132732` |

## Qualification

The [qualification gate](QUALIFICATION_PASSED.json) passed after 200 actual Stage 2 updates, a full evaluation at step 400,100, continued training after evaluation, and a full-state restart to step 400,210. Both post-training optimizer restoration audits passed. All source counts were conserved, MGnify remained non-repeating, and logged learning rates matched the fixed 400k→700k decay schedule. The local regression record covers 50 passing tests plus lint and shell syntax checks; see [LOCAL_CHECKS.json](LOCAL_CHECKS.json).

| Diagnostic checkpoint | Validation loss | P@L | 95% chain-bootstrap CI |
|---|---:|---:|---:|
| Stage 1 parent, 400k | 2.308892 | 42.241% | 41.992–42.495% |
| Qualification only, 400,100 | 2.308911 | 42.429% | 42.178–42.681% |

The qualification checkpoint is a diagnostic trial, not a production endpoint or evidence of a sustained Stage 2 gain. Production starts again from the verified Stage 2 start checkpoint at global 400k and follows the requested 10k evaluation cadence.

## Storage and monitoring

Run artifacts are under `/scratch/muchenli/Nano-Protein-LM-nibi-setting3-stage2-b2048-300k-20260911`, with production in `full/` and frozen source in the sibling directory ending `-run`. Prepared data is available both at `data/` under the run root and at `/localscratch/muchenli.12162637.0/nano-nibi-setting3-stage2-20260911` during the allocation. The original Stage 1 run and checkpoint remain separate.

The durable destination is `/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-setting3-stage2-b2048-300k-20260911`. It already contains the recipe, source bundle, transition receipts and qualification record. The launcher preserves full model/optimizer checkpoints at global 500k, 600k and 700k; scratch `full/checkpoint-latest.pt` updates every 10k. The final evaluation and report copies follow the final checkpoint audit. Later four-GPU continuation preserves global microbatch 512 with microbatch 128 per GPU and accumulation 4; its hardware memory/throughput still requires qualification.

The existing two-hour thread monitor is active for this Stage 2 run. [status.py](status.py) reads lightweight state, [launch.py](launch.py) records the gated Slurm workflow, and [verify_launch.py](verify_launch.py) checks the actual production configuration, state continuity, GPU use and preserved source. Checkpoints and large raw contact components remain remote; the curated report retains small receipts and launch metrics.
