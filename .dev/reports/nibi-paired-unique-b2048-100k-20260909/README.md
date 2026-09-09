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
evaluations at 10k, 20k, 30k and 40k steps are independently audited below.
Trial scores remain qualification evidence, separate from production results. Full final model, optimizer and
sampler checkpoints will be copied and hash-verified on project storage.

Initial finish estimates, including evaluation pauses, are approximately
**9:45 p.m. for AdamW and 10:30 p.m. for Setting 3 on September 9, Toronto time**.
These are early estimates. The existing two-hour monitor remains active.

## Audited learning curve

At 40k, both models consumed **81,920,000 distinct training records** and
**19,357,946,029 model tokens**, with matching source draws and zero source wraps.
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

Setting 3 improves validation loss by **0.047386** and P@L by **10.020 percentage
points** at the matched 40k endpoint; the paired chain-bootstrap interval for the
P@L gain is **9.916–10.124 points**. At 10k, 20k and 30k, the corresponding P@L
gains were 8.419, 9.288 and 9.570 points. Both runs continue to 100k.
Training time excludes evaluation pauses. The contact confidence
interval describes variation across chains, not training seeds. Exact values,
checkpoint hashes and the paired difference interval are in
[learning-curve.json](learning-curve.json). Full training-metric snapshots are
stored as `full/RECIPE/metrics.jsonl.gz`; raw downloaded artifacts stay in local `.exps`.

At the September 9, 10:01 a.m. Toronto check, AdamW was at **49,020 steps** and
Setting 3 at **47,030**. Both had fresh finite metrics, zero sampler wraps and no
fatal OOM exceptions or tracebacks. Downloaded source draws and token counts
matched at all 4,709 common logged steps, through step 47,080. Training continued
after both evaluations. Approximate finish times, including evaluation pauses, are
**9:35 p.m.** and **10:30 p.m. Toronto on September 9**, respectively.

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
