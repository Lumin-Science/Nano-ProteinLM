# Nibi: paired 171M training with sufficient unique data

Both production runs launched successfully on September 8, 2026, at about
11:01 p.m. Toronto time in step **12162637.24**, allocation **12162637**, node **g27**.
The source is frozen at `bc54124abe193623abf64b44e65b881cded65f48`.
These results and code are committed locally; GitHub publication awaits explicit
permission after automatic approval review rejected the pushes.

| Run | Physical GPUs | Parameters | Verified progress at launch audit | Initial speed |
|---|---|---:|---:|---:|
| ESMC-like AdamW baseline | 0–3 | 170,671,168 | 430 / 100,000 steps | 0.790 s/step |
| Setting 3 | 4–7 | 170,559,856 | 410 / 100,000 steps | 0.818 s/step |

Both use **128 sequences per GPU × 4 GPUs × 4 accumulation steps = batch 2,048**.
This preserves the original 512-sequence distributed microstep and Setting 3's
sqrt-loss denominator size. Qualification completed 20 steps per model, including
full evaluation at step 10 followed by continued training. Peak allocated GPU
memory was 40.62 GB for AdamW and 38.60 GB for Setting 3. Full model and optimizer
states restored exactly. See [qualification audit](LOCAL_QUALIFICATION_AUDIT.json),
[launch audit](LAUNCH_VERIFIED.json) and [live GPU assignment proof](GPU_ASSIGNMENT_VERIFIED.json).

The two runs have matching source draws and model-token counts at every common
logged step. Losses and gradients are finite; all source epoch indices remain
zero. An early baseline CUDA allocation warning recovered, and training continued
for hundreds of steps. The monitor watches for actual failures or degradation.

## Recipe and evaluation

Both train from scratch for 100,000 optimizer updates, with RoPE10k, FFN2048,
untied embeddings, FA3/BF16, base LR 5e-4, WD 0.01 and 1,000 warmup steps followed
by constant Stage-1 LR. Setting 3 retains its hybrid Muon/AdamW groups, RMSNorm,
residual routing, depth-scaled initialization, batch balance and sqrt-mask-count
loss. Its attention/FFN Muon LRs remain 4.5e-4/3.75e-4 and Muon WD remains 0.0075.
The 48-hour walltime guard allows both fixed 100k-step budgets to finish.

Evaluate 4,096 MLM sequences and all 20,775 contact chains at each 10k endpoint;
report P@L with a 5,000-replicate chain-bootstrap confidence interval. Production
evaluations at each 10k endpoint through 80k are independently audited below.
Trial scores remain qualification evidence, separate from production results. Full final model, optimizer and
sampler checkpoints will be copied and hash-verified on project storage.

Initial finish estimates, including evaluation pauses, are approximately
**9:45 p.m. for AdamW and 10:30 p.m. for Setting 3 on September 9, Toronto time**.
These are early estimates. The existing two-hour monitor remains active.

## Audited learning curve

At 80k, both models consumed **163,840,000 distinct training records** and
**38,711,834,542 model tokens**, with matching source draws and zero source wraps.
All 16 contact shards, 20,775 chain IDs, checkpoint bindings, fixed probe settings
and the 5,000-replicate bootstrap intervals were independently checked at each
endpoint.

| Step | Recipe | Validation loss ↓ | P@L ↑ | P@L 95% CI | Training time |
|---:|---|---:|---:|---:|---:|
| 10,000 | ESMC-like AdamW | 2.586909 | 13.793% | 13.668–13.920% | 2h 11m 36s |
| 10,000 | Setting 3 | 2.535755 | 22.212% | 22.025–22.401% | 2h 16m 03s |
| 20,000 | ESMC-like AdamW | 2.533118 | 17.571% | 17.408–17.737% | 4h 23m 08s |
| 20,000 | Setting 3 | 2.483718 | 26.859% | 26.646–27.073% | 4h 31m 56s |
| 30,000 | ESMC-like AdamW | 2.501133 | 19.864% | 19.674–20.060% | 6h 34m 34s |
| 30,000 | Setting 3 | 2.453497 | 29.434% | 29.204–29.663% | 6h 47m 53s |
| 40,000 | ESMC-like AdamW | 2.480705 | 21.406% | 21.205–21.615% | 8h 46m 02s |
| 40,000 | Setting 3 | 2.433319 | 31.426% | 31.195–31.663% | 9h 03m 50s |
| 50,000 | ESMC-like AdamW | 2.462720 | 23.183% | 22.968–23.405% | 10h 57m 24s |
| 50,000 | Setting 3 | 2.416991 | 32.568% | 32.335–32.803% | 11h 19m 41s |
| 60,000 | ESMC-like AdamW | 2.448553 | 22.800% | 22.576–23.034% | 13h 08m 54s |
| 60,000 | Setting 3 | 2.405183 | 33.543% | 33.305–33.787% | 13h 35m 38s |
| 70,000 | ESMC-like AdamW | 2.437482 | 25.708% | 25.486–25.944% | 15h 20m 29s |
| 70,000 | Setting 3 | 2.395314 | 34.417% | 34.175–34.658% | 15h 51m 43s |
| 80,000 | ESMC-like AdamW | 2.428326 | 25.255% | 25.028–25.497% | 17h 32m 09s |
| 80,000 | Setting 3 | 2.388536 | 35.240% | 35.003–35.483% | 18h 07m 42s |

Setting 3 improves validation loss by **0.039789** and P@L by **9.986 percentage
points** at the matched 80k endpoint; the paired chain-bootstrap interval for the
P@L gain is **9.872–10.102 points**. The corresponding 70k P@L gain was
**8.709 points**, with a paired interval of **8.611–8.809 points**.
Both runs continue to 100k. AdamW's P@L decreased by 0.383 percentage points
between 50k and 60k despite improving validation loss. Both endpoints passed
the same contact-protocol audit, used the same chain set and selected probe
regularization `C=1.0`; no evaluation-protocol change was found.
Training time excludes evaluation pauses. The contact confidence
interval describes variation across chains, not training seeds. Exact values,
checkpoint hashes and the paired difference interval are in
[learning-curve.json](learning-curve.json). Full training-metric snapshots are
stored as `full/RECIPE/metrics.jsonl.gz`; raw downloaded artifacts stay in local `.exps`.

At the September 9, 6:32 p.m. Toronto check, AdamW was at **86,750 steps** and
Setting 3 at **83,170**. Both had fresh finite metrics, zero sampler wraps and no
fatal OOM exceptions or tracebacks. Downloaded source draws and token counts
matched at all 8,326 common logged steps, through step 83,250. Training continued
after both evaluations. Approximate finish times, including evaluation pauses, are
**9:35 p.m.** and **10:30 p.m. Toronto on September 9**, respectively.
SSH access was restored at this check after the earlier MFA access interruption;
the two-hour monitor has resumed normal checks.

## Data coverage

| Source | Prepared training records | Expected draws per model |
|---|---:|---:|
| UniRef90 | 74,175,974 | ~73.00M |
| MGnify | 22,984,503 | ~22.30M |
| OMG/IMG | 111,296,892 | ~109.50M |
| Total | **208,457,369** | **204,800,000** |

The 211 training shards plus three validation shards were checksum-verified and
converted with per-sequence hashing. Both the node-local stores and a preserved
scratch copy passed full checks. Validation tokens and indices are identical to
the earlier runs. [DATA_READY.json](DATA_READY.json) records the proof and
[PLAN.json](PLAN.json) records the immutable shard selection.

The prepared manifest SHA is
`c30da4a52028cb63bf4f764ff039009d965144db5a6fbf1069f4eebf97c7d3fa`.
Each source has headroom over its expected draws. `data_resampling: error` also
refuses to wrap a source partition, so unexpected source imbalance causes a
failure instead of silent reuse. The no-repeat condition applies within each
model's run; both models intentionally use the same dataset.

The previous Nibi baseline and partial Setting 3 used only 7,109,469 records.
Their sequence counters counted presentations, with the completed baseline
averaging 28.8 exposures per record. The user explicitly requested stopping
old Setting 3 step 12162637.14 at approximately 43,250 updates; those artifacts
remain available as repeated-data experiments. See
[data coverage and continuation](../../../docs/data-coverage.md).

## Locations and recovery

- Remote root: `/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909`.
- Frozen checkout: the same path with `-run` appended.
- Production outputs: `full/baseline` and `full/setting3` under that root.
- Data: node-local path in `PRODUCTION_DATA_ROOT.txt`, with a preserved copy at `ROOT/data`.
- Final checkpoint destinations: `/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-paired-unique-b2048-100k-20260909/baseline` and `/setting3`.
- Read-only monitor: `python status.py` on the Nibi login node.

The first production dispatch exited before either model configured or trained
because its wrapper removed Git from PATH. Preserving the inherited PATH fixed
it without changing the qualified source or recipes. The failed attempt is
archived; [QUEUE_RECOVERY.json](QUEUE_RECOVERY.json) records this resolved issue.
The allocation and its owning helper remain intact.
