# Four cumulative R02 RoPE10k recipes: Fir launch record

All four 200-step technical trials **passed**. **Setting 2 is the first completed
100k run**, including full MLM and P@L evaluation. Settings 1 and 3 are training;
setting 4 remains queued. The source/configs are frozen at `253c3ea`; report
updates on main do not change the running checkout.

| Setting | Recipe | Production placement |
|---|---|---|
| 1 | R02 with RoPE10k | fc10111, training in step 58303658.14 since September 7, 4:02 AM Toronto |
| 2 | + batch balance | Completed 100k + evaluations September 7, 8:35 AM Toronto |
| 3 | + sqrt loss | fc10212, training since September 7, 8:36 AM Toronto in step 58303724.9 |
| 4 | + tied embeddings | fc10212, after setting 3 and its evaluations |

All full runs initialize from scratch for 100,000 steps, batch 1,024, warmup
1,000, base LR 5e-4 and base WD 0.01, with preserved R02 Muon groups, RoPE10k,
FFN2048, BF16 and FA3. [Full recipes and semantics](../../docs/PROGRAM2_SCALEUP.md).
R22 narrowing remains deferred in [TODO](../../TODO.md).

## Completed production results (one of four)

Setting 2 completed **100,000 steps, 102.4M sequences and 24,200,224,761 model
tokens**. Training took **12h 33m 42s**; full evaluations finished at **8:35:53 AM
Toronto September 7**. The checkpoint SHA-256 is
`618bc9dfa69736610ed21682fbe30f09f6cb361f8f2725f1c94048b8bf2c2fe8`.

| Recipe | Validation loss ↓ | Perplexity ↓ | Full P@L ↑ | P@L 95% CI |
|---|---:|---:|---:|---:|
| Historical default | 2.47436048 | 11.87411093 | 26.504938% | 26.294763–26.718848% |
| Historical R02, RoPE20k | 2.43698294 | 11.43847806 | 30.310361% | 30.078864–30.547478% |
| **Setting 2: R02 RoPE10k + batch balance** | **2.43871862** | **11.45834888** | **30.715194%** | **30.486543–30.947846%** |

The new result uses the same **4,096 validation sequences / 139,963 masked
targets** and **20,775 contact chains** as the historical comparison. All 16
contact-shard hashes, checkpoint/probe bindings, unique chain coverage and the
same probe split/protocol were checked independently. The mean P@L and
5,000-resample bootstrap interval were also recomputed from the per-chain rows.
These intervals describe variation across chains, not training-seed uncertainty.

Compared with historical RoPE20k R02, setting 2 has **0.001736 higher validation
loss** and **0.4048 percentage points higher P@L**. That comparison changes both
RoPE and batch balancing. **The isolated batch-balancing comparison awaits
setting 1**; no conclusion about its incremental quality effect is available yet.

At 9:01 AM Toronto, setting 3 was healthy at 3,270 steps after starting
automatically at 8:36 AM. Its estimated training finish was **9:10 PM September
7**. Setting 1 remained healthy at 37,880 steps with a **5 PM** ETA; setting 4
will follow setting 3 and its full evaluations.

[Machine-readable partial results](results.json) ·
[Independent result verification](full/r04_batchbalance/RESULT_VERIFIED.json) ·
[Per-chain P@L](full/r04_batchbalance/contact-per-chain.tsv) ·
[Complete training trace, gzip](full/r04_batchbalance/metrics-complete.jsonl.gz) ·
[Progress and queue snapshot](PROGRESS_SNAPSHOT.json).
The uncompressed `metrics.jsonl` in the setting 2 report directory retains its
initial launch snapshot. Full raw contact shards remain in the remote artifact
root and local `.exps` audit directory; their verified hashes are preserved here.

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
