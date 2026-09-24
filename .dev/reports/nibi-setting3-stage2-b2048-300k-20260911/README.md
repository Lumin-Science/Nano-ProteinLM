# Nibi Setting 3 Stage 2 — batch 2,048, 300k additional updates

Production completed **300,000 Stage 2 updates, global 400k → 700k**, on eight Nibi H100 GPUs. The [final verification](FINAL_VERIFIED.json) passed: all **30 production evaluations** are verified and the durable 700k checkpoint passed exact model and optimizer restoration. The production launcher finished September 13, 2026 at **17:52:59 Toronto**; the complete workflow finished at **17:53:59**. Production took **48h 00m 39s** including evaluations and checkpoint handling, with **46h 13m 23s** recorded in the training loop. The user-owned allocation remains retained.

## Production evaluation results

The final **700k** checkpoint has validation loss **2.248124** and P@L **46.264%** (95% CI **46.016–46.523%**), the best measured values on both metrics in this Stage 2 run. Each [independent endpoint audit](full/evaluations/step-700000/LOCAL_AUDIT.json) checks checkpoint bindings, all 16 contact shards, all 20,775 historical chain identities, the unchanged MLM/probe protocols and an independently recomputed 5,000-replicate bootstrap. The [complete machine-readable curve](learning-curve.json) excludes qualification trials.

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
| Stage 2, 490k | 90,000 | 2.290492 | 44.242% | 43.990–44.497% |
| Stage 2, 500k | 100,000 | 2.288431 | 44.329% | 44.080–44.588% |
| Stage 2, 510k | 110,000 | 2.285692 | 44.658% | 44.409–44.915% |
| Stage 2, 520k | 120,000 | 2.285323 | 44.580% | 44.332–44.834% |
| Stage 2, 530k | 130,000 | 2.281564 | 44.635% | 44.387–44.892% |
| Stage 2, 540k | 140,000 | 2.280197 | 44.732% | 44.481–44.987% |
| Stage 2, 550k | 150,000 | 2.277880 | 45.046% | 44.795–45.303% |
| Stage 2, 560k | 160,000 | 2.277282 | 45.044% | 44.794–45.298% |
| Stage 2, 570k | 170,000 | 2.272784 | 45.225% | 44.976–45.478% |
| Stage 2, 580k | 180,000 | 2.272021 | 45.215% | 44.965–45.469% |
| Stage 2, 590k | 190,000 | 2.270239 | 45.371% | 45.122–45.625% |
| Stage 2, 600k | 200,000 | 2.267967 | 45.330% | 45.082–45.587% |
| Stage 2, 610k | 210,000 | 2.266018 | 45.435% | 45.189–45.691% |
| Stage 2, 620k | 220,000 | 2.264071 | 45.557% | 45.309–45.813% |
| Stage 2, 630k | 230,000 | 2.262353 | 45.725% | 45.477–45.983% |
| Stage 2, 640k | 240,000 | 2.260115 | 45.791% | 45.542–46.047% |
| Stage 2, 650k | 250,000 | 2.258179 | 45.882% | 45.632–46.134% |
| Stage 2, 660k | 260,000 | 2.254962 | 45.971% | 45.722–46.227% |
| Stage 2, 670k | 270,000 | 2.254061 | 46.117% | 45.868–46.372% |
| Stage 2, 680k | 280,000 | 2.251548 | 46.122% | 45.872–46.377% |
| Stage 2, 690k | 290,000 | 2.248952 | 46.207% | 45.957–46.465% |
| Stage 2, 700k | 300,000 | 2.248124 | 46.264% | 46.016–46.523% |

Relative to 400k, validation loss changed by **-0.060768**, and P@L improved by **4.023 percentage points**. The paired chain-bootstrap 95% interval for this gain is **3.958–4.089 points**. These intervals quantify uncertainty across evaluation chains, not variation across training seeds.

## Final training and storage checks

Training stopped at the requested **700,000** global updates with no production failure. The source checkout remained clean at the frozen commit, and the final metrics followed the original Stage 2 decay clock. The completed run retains **zero MGnify repeats** and **4,635,828 unused MGnify records**. UniRef90 and OMG continued their permitted complete global passes.

The [durable final checkpoint](milestones/checkpoint-700000.json) and [independent restoration audit](milestones/checkpoint-700000-restore-verified.json) bind the evaluated 700k weights to the saved full model, optimizer and global sampler state. Checkpoint SHA-256: `69f9f07ae0462834029fa2ccbbb05b93034097bfc540157385fb362ad7a7abeb`. The [host receipt](monitoring/final-storage-check.json) confirms production step 12162637.44 ended while allocation 12162637 remains running on g27, with **12.64 hours remaining** at the **September 13, 18:45 Toronto** check.

## Earlier monitoring check

At **September 11, 19:00 Toronto**, the [monitoring receipt](monitoring/20260911T2300Z.json) recorded **407,270 global steps**, or **7,270 / 300,000 Stage 2 updates (2.42%)**. Production step 12162637.44 remained active on g27, the source checkout was unchanged, and finite loss/gradient, decay schedule and source accounting checks passed. No production validation or P@L endpoint was due yet; the first 410k evaluation was estimated to finish around **19:29 Toronto**.

Training averaged **0.555 seconds per update** since the launch audit. Including the qualification-measured evaluation overhead, the revised finish estimate was **September 13 at 17:56 Toronto**, or **22:27** with a 10% training slowdown. The allocation had **60.39 hours remaining**, leaving **13.46 hours** of headroom at the central estimate. MGnify had **40,603,251 unused records**, **zero repeats**, and approximately **12.88% headroom** over its remaining expected draws. UniRef90 was in zero-indexed epoch 4 and OMG in epoch 1, consistent with their permitted complete global passes.

## Frozen recipe

The [recipe](../../configs/nibi/setting3-nibi-stage2-b2048-300k.yaml), [transition methods](../../../docs/USAGE.md#training), and [launcher](../../../runs/nibi_setting3_stage2.sh) are preserved on main. The production source is pinned separately at **f4b672d0a3a70e15740ee3876c202e535f727d3c**, so later report updates do not change running code.

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

The durable destination is `/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-setting3-stage2-b2048-300k-20260911`. It contains the recipe, source bundle, transition and qualification records, full model/optimizer checkpoints at 500k, 600k and 700k, and the final report under `run-record/`. The final checkpoint is `checkpoint-700000.pt`; scratch `full/checkpoint-final.pt` contains the same evaluated state. Later four-GPU continuation uses microbatch 128 per GPU with accumulation 4 to retain batch 2,048; memory and throughput on that layout still require qualification.

The durable **500k** full model/optimizer checkpoint is [preserved and checksum verified](milestones/checkpoint-500000.json). An [independent restoration audit](milestones/checkpoint-500000-restore-verified.json) on allocated CPUs confirmed exact model and optimizer tensor restoration, finite state, and matching global source history. Its SHA-256 is `020dc374ff438da6a38656def5aa3fc72aca6ee2a3bc174349d1d88b3ad2a072`; it contains the full replicated optimizer state for future continuation.

The durable **600k** full model/optimizer checkpoint is also [preserved and checksum verified](milestones/checkpoint-600000.json). Its [independent restoration audit](milestones/checkpoint-600000-restore-verified.json) passed on allocated CPUs after access was restored, confirming exact model and optimizer restoration, finite state, and the same source history as the evaluated 600k checkpoint. Its SHA-256 is `585edf8c0188c25f3ca6b9e513a917b129e0475e6d08ac9bbf767e782cbf1d0b`.

The requested 700k training, evaluation and checkpoint-verification scope is complete. The two-hour monitor is to be paused after final publication; the allocation remains retained. [status.py](status.py) reads state, [launch.py](launch.py) records the gated workflow, and [verify_launch.py](verify_launch.py) checks the launch. Checkpoints and large raw contact components remain remote; this report retains small receipts, per-chain summaries and compressed metrics.