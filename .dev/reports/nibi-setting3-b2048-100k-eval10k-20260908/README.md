# Nibi Setting 3: batch 2,048, 100k steps, evaluation every 10k

**Status: running.** Setting 3 started from scratch at **16:45:16 Toronto on
September 8, 2026**, in step **12162637.14**, after the Nibi AdamW baseline finished
training, all ten evaluations and durable checkpoint preservation. The full
periodic-evaluation qualification and **eight-to-four-GPU Muon/AdamW checkpoint
continuation test both passed**. See [launch verification](LAUNCH_VERIFIED.json),
[qualification](QUALIFICATION_PASSED.json), and [baseline hash checks](BASELINE_READY.json).

At **20:00 Toronto on September 8**, production had reached **25,160 / 100,000 steps** on eight
H100s, batch **2,048**, with finite loss/objective/gradients and pinned FA3.
Both full evaluations at 10k and 20k passed, and training continued. The current ETA,
including remaining evaluations, is **about 05:41 Toronto on September 9**.
The training source remains **`c76a07998987a4746d4bffc89d878a758cd735cb`**.
Two-hour monitoring continues through final evaluation and checkpoint preservation.

The recipe is the best completed Fir setting: R02 RoPE10k + batch balance + sqrt
loss. See the [completed Fir comparison and detailed explanation](../../../docs/BEST_RECIPE_VS_BASELINE.md)
for its batch-1,024 result: validation loss **2.418720**, P@L **32.682%**.
The [original queue activation](QUEUE_RECORD.json) and [login-node controller
recovery](QUEUE_RECOVERY.json) remain archived below.

## Production learning curve

| Optimizer step | Sequences seen | Validation MLM loss | Perplexity | P@L | 95% chain-bootstrap CI | Evaluation pause |
| --- | --- | --- | --- | --- | --- | --- |
| 10,000 | 20,480,000 | 2.536788 | 12.63900 | 22.31377% | [22.12376%, 22.50470%] | 211.61 seconds |
| 20,000 | 40,960,000 | 2.485602 | 12.00834 | 26.56299% | [26.35130%, 26.77835%] | 209.70 seconds |

The independent [10k](full/evaluations/step-010000/LOCAL_AUDIT.json) and
[20k](full/evaluations/step-020000/LOCAL_AUDIT.json) audits check
all 16 shard hashes, checkpoint bindings, the fixed 4,096 MLM sequences and exact
20,775-chain population, probe protocol, and a recomputed 5,000-replicate bootstrap
CI. These are production results; qualification scores are separate.
See the [machine-readable learning curve](learning-curve.json).

At the matched **20k-step / batch-2,048** checkpoint, Setting 3 has validation
loss **2.485602** versus the AdamW baseline's **2.536814**, and P@L **26.56299%**
versus **18.84834%**. The difference is **+7.71465 percentage points**; a paired
chain-bootstrap 95% CI for that difference is **[7.62296, 7.80763] pp**
(5,000 replicates, same 20,775 chains). This is an intermediate comparison from
one training seed per recipe. See the [matched comparison receipt](full/evaluations/step-020000/BASELINE_COMPARISON.json).

## Production configuration

The [config](../../../configs/archive/esmc-171m-setting3-nibi-fa3-b2048-stage1-100k.yaml)
retains every scientific setting from
[`r10_sqrtloss.yaml`](../../../configs/archive/program2_h100_100k/r10_sqrtloss.yaml).
The changes are eight-GPU execution, checkpoint/evaluation cadence, and a 24-hour
training guard. The larger global batch comes from the larger GPU count.

| Setting | Value |
| --- | --- |
| Initial state | From scratch; seed 20260824 |
| Parameters | 170,559,856 |
| GPUs / precision / attention | 8 × H100; BF16 mixed precision; pinned FlashAttention-3 |
| Context | 512 |
| Global batch | **2,048 = 64 per GPU × 8 GPUs × 4 accumulation microsteps** |
| Budget | **100,000 optimizer steps; 204,800,000 sequences** |
| Schedule | Warmup 1,000 steps, then constant Stage-1 LR |
| Base LR / WD | **5e-4 / 0.01** |
| Muon attention matrices | LR 4.5e-4; WD 0.0075 |
| Muon FFN matrices | LR 3.75e-4; WD 0.0075 |
| AdamW remainder | LR 5e-4; WD 0.01 for decay group, 0 for no-decay group; betas 0.9/0.95 |
| Muon options | Momentum 0.95, Nesterov, 5 Newton–Schulz iterations, `match_rms_adamw` |
| Gradient clipping | Global norm 1.0 |
| Architecture | RoPE 10k, parameter-free transformer RMSNorm, learned residual routing, depth-scaled residual initialization, FFN 2,048, untied embeddings |
| Batch balance | Redistribute complete, already-masked rows across ranks by valid-token load; preserve rows and equal sequence counts |
| Training loss | `sqrt_mask_count`: sequence mean CE weighted by square root of masked-target count |
| Evaluation / checkpoint cadence | Every 10,000 steps, including 100,000 |
| Compile / gradient checkpointing | Both disabled |

The Muon LR/WD group multipliers remain those of the winning recipe; doubling batch
does not trigger another LR or weight-decay scaling. Its sqrt-loss denominator now
spans **512 proteins per distributed microstep**, followed by an average of four
microstep objectives. The prior four-GPU Fir run normalized over 256 proteins per
microstep. Consequently this is a larger-batch experiment with twice the sequence
exposure, not a replication of the batch-1,024 trajectory.

## Queue and qualification

`queue-after-baseline-system.sh` is the active detached, locked queue controller.
It reads small
metadata on the login node once per minute. It never cancels jobs or alters the
predecessor. After the baseline is complete and no other compute step remains,
it requests a step inside the **existing** allocation:

```bash
srun --jobid=12162637 --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=64 --gres=gpu:8 --nodelist=g27 \
  /bin/bash --login /scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908-run/reports/nibi-setting3-b2048-100k-eval10k-20260908/run-queued.sh
```

All expensive checks, training and evaluation run on the allocated compute node.
The controller checks that GPUs are idle, verifies the baseline's persistent
checkpoint hash and ten evaluation artifacts, then runs these qualification steps:

1. Train Setting 3 for 20 steps on eight GPUs, perform the **full** evaluation at
   step 10, and continue the same training processes to step 20.
2. Restore that checkpoint on four GPUs with accumulation 8, retaining batch
   2,048, and train from step 20 to 22. Check both Muon and AdamW state restoration,
   finite model/optimizer tensors, exact checkpoint load/save round trips, and
   complete ownership of all model parameters.
3. Require `QUALIFICATION_PASSED.json` from this exact source commit, then launch
   production from scratch. Qualification uses a five-step warmup solely for its
   short test; production uses the requested **1,000-step warmup**.

Any failed check stops this queue and leaves a failure receipt. A lock and existing
output checks prevent duplicate launches. The retained allocation and unrelated
workloads remain user-owned.

**Controller recovery, September 8:** the original controller (PID 4065953) exited
with code 126 while waiting because the login-node CVMFS Python executable became
unavailable (`Transport endpoint is not connected`). No qualification or production
had started. After checking the old process was absent and taking the queue lock,
its receipts were archived under `queue-attempts/attempt-1-cvmfs-failure`.
The [replacement controller](queue-after-baseline-system.sh) lives in the artifact
root, outside the frozen training checkout. It uses system-local Python 3.9 and
utilities for polling and checks the frozen launcher manifest and config SHA.
The eventual compute step starts a login shell so it can load the compute node's
software environment. A [CPU-only check](compute-environment-recovery.txt) confirmed
Git, clean source `c76a079` and the frozen Python 3.11.4 runtime on `g27`.
Full Git/source checks in the original launcher remain mandatory before training.
The replacement was confirmed detached, ignoring SIGHUP, and polling repeatedly.

At planning time (15:06 Toronto, September 8), the baseline was at 88,010 steps,
with completion estimated around **16:45 Toronto**. Allowing for qualification and
setup, Setting 3 should start around **17:00 September 8**. A provisional **13–14
hours** including evaluation would finish around **06:00–07:00 September 9**.
These are estimates from the existing runs; eight-GPU Setting-3 throughput will be
measured by qualification and the first production updates before refining the ETA.

## Evaluation and result integrity

Production evaluation runs at steps 10k, 20k, …, 100k:

- The same fixed **4,096-sequence MLM** validation set with **139,963 masked
  residues**, reporting unweighted sequence-mean NLL, regardless of training loss.
- The same **20,775-chain contact P@L** benchmark, all 16 scoring shards, frozen
  probe split and fitting protocol, and **5,000-replicate chain-bootstrap 95% CI**.
- A checkpoint SHA and optimizer-step binding in every receipt. Intermediate
  evaluation pauses training in place; it does not restart training. Evaluation
  wall time is recorded separately from training time.
- `verify_evaluation.py` independently checks downloaded shard hashes, chain sets,
  evaluation protocol and bootstrap intervals before small results are published.

The recurring two-hour monitor checks queue health, qualification, production
progress and newly completed evaluations. It publishes audited results to this
report on main and reports meaningful changes, failures or required action.

## Paths and continuation

| Artifact | Location |
| --- | --- |
| Artifact root | `/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-100k-eval10k-20260908` |
| Frozen checkout | Artifact root with `-run` appended |
| Queue receipts | `QUEUE_DRIVER_PID`, `QUEUE_STATUS.json`, `queue.log`, `compute-identity.txt` |
| Qualification | `trial/`, `resume4/`, `QUALIFICATION_PASSED.json` |
| Production | `full/`, with `evaluations/step-NNNNNN/` |
| Persistent final checkpoint | `/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-setting3-b2048-100k-eval10k-20260908/checkpoint-final.pt` |
| Local downloaded artifacts | `.exps/nibi-setting3-b2048-100k-eval10k-20260908/` (ignored by Git) |

The final checkpoint contains the complete model, **both Muon momentum and AdamW
moments/steps**, training counters, and all eight ranks' RNG/sampler state. DDP
optimizer state is replicated, so this is a full checkpoint, not eight optimizer
shards. The driver verifies its contents, copies it atomically to project storage,
checks the copied SHA, and preserves the configuration and resume documentation.
The destination is planned until `CHECKPOINT_PRESERVED.txt` is written.

For future continuation on four H100s, use `--resume` with that final checkpoint,
set `expected_world_size: 4`, keep micro-batch 64, and set accumulation **8** to
retain batch 2,048. Extend `max_steps` and `schedule_steps` to the desired total
endpoint (e.g. 200,000 for another 100,000 steps), increase the wall-time guard,
and use a fresh output directory. The existing 1,000-step warmup does not restart.
See [checkpoint continuation](../../../docs/checkpoint-resume.md).

Changing GPU count creates a new deterministic data stream instead of restoring
eight sampler streams into four ranks. At micro-batch 64, sqrt-loss normalization
also changes from 512 to 256 proteins per microstep, with eight rather than four
microsteps averaged per update. Full optimizer continuation is supported; future
updates are not expected to be bitwise identical to the eight-GPU trajectory.

Large checkpoints and raw contact shard dumps stay outside Git. Only configs,
launch scripts, small verification receipts and compressed per-chain tables belong
in the published report.
