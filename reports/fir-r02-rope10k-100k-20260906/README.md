# Four cumulative R02 RoPE10k recipes: Fir training results

All four 200-step technical trials **passed**. **Settings 1–3 have completed
100k steps and full MLM/P@L evaluations.** Setting 4 started automatically and
is training. Source/configs are frozen at `253c3ea`; report updates on main
do not change the running checkout.

| Setting | Recipe | Production placement |
|---|---|---|
| 1 | R02 with RoPE10k | Completed 100k + evaluations September 7, 5:11 PM Toronto |
| 2 | + batch balance | Completed 100k + evaluations September 7, 8:35 AM Toronto |
| 3 | + sqrt loss | Completed 100k + evaluations September 7, 9:17 PM Toronto |
| 4 | + tied embeddings | fc10212, training since September 7, 9:18 PM Toronto in step 58303724.9 |

All full runs initialize from scratch for 100,000 steps, batch 1,024, warmup
1,000, base LR 5e-4 and base WD 0.01, with preserved R02 Muon groups, RoPE10k,
FFN2048, BF16 and FA3. [Full recipes and semantics](../../docs/PROGRAM2_SCALEUP.md).
R22 narrowing remains deferred in [TODO](../../TODO.md).

## Completed production results (three of four)

All three new completed runs processed **100,000 steps, 102.4M sequences and
24,200,224,761 model tokens**. Setting 1 finished its evaluations at **5:11:09 PM
Toronto September 7**; setting 2 finished at **8:35:53 AM**, and setting 3 at
**9:17:55 PM** the same day.

| Recipe | Training time | Validation loss ↓ | Perplexity ↓ | Full P@L ↑ | P@L 95% CI |
|---|---:|---:|---:|---:|---:|
| Historical default | 12h 00m 42s | 2.47436048 | 11.87411093 | 26.504938% | 26.294763–26.718848% |
| Historical R02, RoPE20k | 12h 56m 30s | 2.43698294 | 11.43847806 | 30.310361% | 30.078864–30.547478% |
| **Setting 1: R02 RoPE10k** | **12h 57m 54s** | **2.43780741** | **11.44791265** | **30.164593%** | **29.935870–30.393752%** |
| **Setting 2: + batch balance** | **12h 33m 42s** | **2.43871862** | **11.45834888** | **30.715194%** | **30.486543–30.947846%** |
| **Setting 3: + sqrt loss** | **12h 34m 40s** | **2.41871987** | **11.23147240** | **32.682480%** | **32.447348–32.919702%** |

Each result uses the same **4,096 validation sequences / 139,963 masked targets**
and **20,775 contact chains** as the historical comparison. All 16 contact-shard
hashes, checkpoint/probe bindings, unique chain coverage and the same probe
split/protocol were checked independently for each new result. Mean P@L and
5,000-resample bootstrap intervals were recomputed from the per-chain rows.

### Available adjacent comparisons

Deltas below are candidate minus baseline. Lower loss/time and higher P@L are
better. Each pair uses one matched training seed; timing also comes from one run
on each assigned node.

| Increment | Validation loss delta | P@L delta | Paired chain-bootstrap 95% CI for P@L delta | Training time delta |
|---|---:|---:|---:|---:|
| Historical RoPE20k R02 → setting 1, RoPE10k | +0.00082447 | −0.1458 pp | −0.2059 to −0.0867 pp | +1m 24s (+0.18%) |
| Setting 1 → setting 2, add batch balancing | +0.00091121 | +0.5506 pp | +0.4942 to +0.6067 pp | −24m 12s (−3.11%) |
| Setting 2 → setting 3, add sqrt loss | −0.01999875 | +1.9673 pp | +1.9056 to +2.0263 pp | +58s (+0.13%) |

In this matched-seed comparison, batch balancing shortened training and produced
higher contact P@L, while validation loss was slightly higher (+0.037%). Resetting
RoPE to 10k produced slightly higher loss and lower P@L than historical RoPE20k.
Adding sqrt-mask-count loss reduced validation loss by **0.02000 (0.82%)** and
increased P@L by **1.9673 percentage points**, with almost unchanged training
time (+0.13%). Setting 3 currently has the lowest validation loss and highest
P@L among the completed recipes, including both historical references.
These are observations for these checkpoints. The chain-bootstrap intervals
condition on the trained models; they **do not measure training-seed uncertainty
or establish reproducible training effects**. Repeated training seeds remain
necessary for that claim. The tied-embedding increment is pending.

Setting 4 launched automatically at **9:17:56 PM Toronto September 7**, following
setting 3's full evaluations. At **10:01 PM**, it was healthy at **5,690 steps**,
with training expected to finish around **9:53 AM Toronto September 8**, followed
by full evaluations. Its exact config, clean frozen source, four-rank FA3 run
contract and finite loss/gradient progress were verified. All user-owned Slurm
allocations remain intact.

[Machine-readable partial results](results.json) ·
[Adjacent deltas and paired intervals](ADJACENT_COMPARISONS.json) ·
[Setting 1 verification](full/r02_rope10k/RESULT_VERIFIED.json) ·
[Setting 2 verification](full/r04_batchbalance/RESULT_VERIFIED.json) ·
[Setting 3 verification](full/r10_sqrtloss/RESULT_VERIFIED.json) ·
[Setting 4 launch evidence](full/r29_tied/) ·
[Progress and queue snapshot](PROGRESS_SNAPSHOT.json).

Per-chain P@L and complete gzipped training traces are stored in each completed
run directory. Uncompressed `metrics.jsonl` files in those directories retain
initial launch snapshots. Full raw contact shards remain in the remote artifact
root and local `.exps` audit directory; their verified hashes are preserved here.
The [completion auditor](launch/verify_completed_result.py) reads the frozen
Git config and can be rerun from the repo root with a method name; the
[comparison script](launch/build_adjacent_comparisons.py) rebuilds adjacent
comparisons from locally verified completed artifacts.

## Scheduled setting 1 launch: September 7

The 4 AM Toronto check found all four fc10111 H100s idle (5 MiB per GPU,
0% utilization, no compute processes), with more than 45 hours left in
allocation `58303658`. The frozen source, all four passed trial receipts and
production config hashes were verified before dispatch. Setting 1 launched at
**04:02:34 Toronto / 08:02:34 UTC**, after the required timestamp.

At **4:08:27 AM Toronto**, setting 1 had reached
**270 / 100,000 optimizer steps** with finite logged loss and gradients.
The actual config has batch 1,024 and warmup 1,000; the clean source commit and
four-rank FA3 run contract match the qualified recipe. All four GPUs were
active at 96–98% utilization. Measured training speed was
**0.4637 s/update**, projecting completion around
**5 PM Toronto September 7**, followed by full MLM and P@L evaluation.

Setting 2 remained healthy at 65,290 steps, with an approximately
8:30 AM Toronto training ETA. Settings 3 and 4 remain queued behind its full
evaluations. Hourly monitoring continues.

[Setting 1 launch snapshot](SETTING1_LAUNCH_SNAPSHOT.json) ·
[Exact executed config and startup evidence](full/r02_rope10k/) ·
[4 AM GPU availability check](setting1-4am-gpu-preflight.txt) ·
[Active GPU check](setting1-live-gpus.txt).

## Trial protocol

Each trial uses the full model and 64 examples per GPU × four GPUs × four
accumulation microsteps. Only max/schedule steps (200), warmup (50) and the
wall guard (1,200 seconds) differ from production. The shortened warmup is a
technical stress check at configured peak learning rates. Finite gradients,
checkpoint/optimizer reload, attention-feature layout, token preservation,
unique tied-weight ownership and a 32-sequence MLM smoke are checked. The
smoke MLM values are not a recipe ranking or production results.

## Qualification results

All four trials finished 200 updates with **204,800 sequences and 48,479,810
model tokens each**, finite logged losses and gradients, finite checkpoint
parameters, successful checkpoint/optimizer reload, and qualified FA3. The
shared tensor in setting 4 has one AdamW owner and state; its parameter count
is 170,510,704 versus 170,559,856 in settings 1–3. No trial failed.

| Setting | Seconds/update, last 100 updates | Peak allocated memory, max GPU | 100k training projection |
|---|---:|---:|---:|
| 1. r02_rope10k | 0.464259 | 22.87 GiB | 12h 53m |
| 2. r04_batchbalance | 0.451738 | 20.34 GiB | 12h 32m |
| 3. r10_sqrtloss | 0.452391 | 20.34 GiB | 12h 33m |
| 4. r29_tied | 0.452617 | 20.34 GiB | 12h 34m |

These are short single-trial timings, not replicated speed estimates. Projections
exclude startup, checkpoint writes and evaluation. Compare scientific quality
only after the full 100k runs and fixed full evaluation. The shortened warmup
and 32-sequence smoke evaluation are not used in production.

[Aggregate qualification gate](ALL_TRIALS_PASSED.json) · [Per-trial configs,
metrics, environment, completion, verification and MLM receipts](trials/).

## Initial production launch snapshot (September 6)

At **September 6, 7:56:55 PM Toronto** (23:56:55 UTC), setting 2 had reached
**190 / 100,000 steps**
with finite logged loss and gradients. Its exact production config, clean source
commit, four-rank run contract and FA3 qualification were verified. A live GPU
check showed all four H100s at 98% utilization.

| Setting | Estimated training finish, Toronto | Basis |
|---|---|---|
| 1 | September 7, 4:55 PM | Assumes 4 AM start and free fc10111 GPUs |
| 2 | September 7, 8:26 AM | Live measured speed |
| 3 | September 7, 9:10 PM | Sequential queue; trial speed plus evaluation/setup margin |
| 4 | September 8, 9:54 AM | Sequential queue; trial speed plus evaluation/setup margin |

ETAs exclude that run's final evaluation and may move with measured throughput.
[Timestamped launch snapshot](LAUNCH_SNAPSHOT.json) and [setting 2 launch evidence](full/r04_batchbalance/).

## Launch and monitoring

The trial Slurm step is `58303724.8`. The production queue on fc10212 executes
settings 2 → 3 → 4 sequentially. It stops on any failed training or evaluation
stage and retains logs. Every full run includes the fixed 4,096-sequence MLM
and full 20,775-chain P@L evaluation, with 16 inference shards and a 5,000-sample
chain bootstrap CI. There is one matched training seed, not a seed-variance
estimate.

The active hourly heartbeat `launch-r02-rope10k-on-fir-at-4-am` performed the
**September 7, 04:00 America/Toronto (08:00 UTC)** availability check and launched
setting 1 after confirming free GPUs. Its timestamp guard prevented earlier
launch. Existing workloads were not interrupted. The heartbeat continues to
monitor all four runs; each full launch requires at least 16h 15m of allocation
time.

- [Launch plan and script digests](LAUNCH_PLAN.json).
- [Exact launch/verification scripts and evaluator snapshot](launch/).
- [GPU schedule](../../docs/PROGRAM2_GPU_PLAN.md).
- Historical [default and RoPE20k R02 results](../fir-171m-100k-20260906/README.md).
- Remote artifacts: `/scratch/muchenli/Nano-Protein-LM-r02-rope10k-100k-20260906`.
  Source checkout: the same path with `-run` appended. Local working receipts:
  `.exps/fir-r02-rope10k-100k-20260906` (untracked).
