# Four cumulative R02 RoPE10k recipes: Fir launch record

All four 200-step technical trials **passed** on fc10212. The sequential
production queue is training setting 2 as Slurm step **58303724.9**. Production
training results are pending. The source/configs are frozen at `253c3ea`;
documentation and launch-record updates on main do not change the running
checkout.

| Setting | Recipe | Production placement |
|---|---|---|
| 1 | R02 with RoPE10k | fc10111, at or after September 7, 4:00 AM Toronto, once GPUs are free |
| 2 | + batch balance | fc10212, training in step 58303724.9 |
| 3 | + sqrt loss | fc10212, after setting 2 and its evaluations |
| 4 | + tied embeddings | fc10212, after setting 3 and its evaluations |

All full runs initialize from scratch for 100,000 steps, batch 1,024, warmup
1,000, base LR 5e-4 and base WD 0.01, with preserved R02 Muon groups, RoPE10k,
FFN2048, BF16 and FA3. [Full recipes and semantics](../../docs/PROGRAM2_SCALEUP.md).
R22 narrowing remains deferred in [TODO](../../TODO.md).

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

## Production launch snapshot

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

The active hourly heartbeat `launch-r02-rope10k-on-fir-at-4-am` also checks
fc10111 at **September 7, 04:00 America/Toronto (08:00 UTC)**. A timestamp guard
prevents earlier launch. Occupied GPUs defer setting 1 to a later hourly check;
existing workloads are never interrupted. Both nodes must have at least
16h 15m remaining before each full run.

- [Launch plan and script digests](LAUNCH_PLAN.json).
- [Exact launch/verification scripts and evaluator snapshot](launch/).
- [GPU schedule](../../docs/PROGRAM2_GPU_PLAN.md).
- Historical [default and RoPE20k R02 results](../fir-171m-100k-20260906/README.md).
- Remote artifacts: `/scratch/muchenli/Nano-Protein-LM-r02-rope10k-100k-20260906`.
  Source checkout: the same path with `-run` appended. Local working receipts:
  `.exps/fir-r02-rope10k-100k-20260906` (untracked).
