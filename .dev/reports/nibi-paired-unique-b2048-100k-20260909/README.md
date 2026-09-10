# Nibi: paired 171M training with sufficient unique data

**Both runs completed all 100,000 optimizer steps and all ten evaluations.**
AdamW finished at **9:34:09 p.m. Toronto on September 9, 2026**; Setting 3
finished at **10:30:16 p.m.** Both full model/optimizer/sampler checkpoints are
preserved on project storage, independently checksum-verified, and successfully
restored by the frozen verifier.

The runs launched on September 8 at about 11:01 p.m. Toronto in step
**12162637.24**, allocation **12162637**, Nibi **g27**. Source commit:
`bc54124abe193623abf64b44e65b881cded65f48`. The allocation remains intact;
all eight H100s were idle at the final inspection. The two-hour completion
monitor is paused because both runs and their artifacts are verified complete.
The frozen recipes, learning curves and verification receipts are recorded in
this repository. Publication to main was explicitly authorized on September 10.

## Final comparison

| Recipe | Validation loss ↓ | P@L ↑ | 95% chain-bootstrap CI | Training time | Launch through final evaluation/preservation |
|---|---:|---:|---:|---:|---:|
| ESMC-like AdamW | 2.414734 | 28.173% | 27.932–28.421% | 21h 55m 13s | 22h 32m 41s |
| Setting 3 | 2.375701 | 36.567% | 36.328–36.815% | 22h 39m 36s | 23h 28m 48s |

Setting 3 has **0.039033 lower validation loss** and **8.394 percentage points
higher P@L**. The paired chain-bootstrap interval for the P@L difference is
**8.297–8.492 points**.
Each model consumed **204,800,000 distinct training records** and
**48,391,423,362 non-padding model tokens**. Source counts and tokens match
at all 10,001 common logged steps. Every source epoch
index remained zero; there were no fatal OOM exceptions or tracebacks.
Training times exclude evaluation pauses. Confidence intervals describe
variation across contact chains, not variation across training seeds.

## Recipe and evaluation

Both use **128 sequences/GPU × 4 GPUs × 4 accumulation = batch 2,048**,
context 512, seed 20260824, FA3/BF16, base LR 5e-4, WD 0.01, and 1,000 warmup
steps followed by constant Stage-1 LR. AdamW occupied physical GPUs 0–3;
Setting 3 occupied GPUs 4–7. Setting 3 retains hybrid Muon/AdamW, RMSNorm,
learned residual routing, depth-scaled initialization, RoPE10k, FFN2048,
untied embeddings, batch balance, and sqrt-mask-count loss. Muon attention
and FFN LRs are 4.5e-4 and 3.75e-4, with WD 0.0075.

Each evaluation uses the same 4,096 MLM validation sequences and all 20,775
contact chains. All 16 contact shards, frozen probe settings, checkpoint
bindings, chain identities, and 5,000-replicate bootstrap intervals were
independently verified for all ten endpoints of both runs.

## Audited learning curves

| Step | AdamW validation loss ↓ | Setting 3 validation loss ↓ | AdamW P@L (95% CI) ↑ | Setting 3 P@L (95% CI) ↑ |
|---:|---:|---:|---:|---:|
| 10,000 | 2.586909 | 2.535755 | 13.793% (13.668–13.920%) | 22.212% (22.025–22.401%) |
| 20,000 | 2.533118 | 2.483718 | 17.571% (17.408–17.737%) | 26.859% (26.646–27.073%) |
| 30,000 | 2.501133 | 2.453497 | 19.864% (19.674–20.060%) | 29.434% (29.204–29.663%) |
| 40,000 | 2.480705 | 2.433319 | 21.406% (21.205–21.615%) | 31.426% (31.195–31.663%) |
| 50,000 | 2.462720 | 2.416991 | 23.183% (22.968–23.405%) | 32.568% (32.335–32.803%) |
| 60,000 | 2.448553 | 2.405183 | 22.800% (22.576–23.034%) | 33.543% (33.305–33.787%) |
| 70,000 | 2.437482 | 2.395314 | 25.708% (25.486–25.944%) | 34.417% (34.175–34.658%) |
| 80,000 | 2.428326 | 2.388536 | 25.255% (25.028–25.497%) | 35.240% (35.003–35.483%) |
| 90,000 | 2.420469 | 2.379697 | 27.183% (26.953–27.426%) | 35.932% (35.690–36.177%) |
| 100,000 | 2.414734 | 2.375701 | 28.173% (27.932–28.421%) | 36.567% (36.328–36.815%) |

![Validation loss and contact precision throughout the matched 100k runs.](learning-curves.png)

Exact values, checkpoint hashes, evaluation durations, cumulative training times,
and all ten paired comparisons are in [learning-curve.json](learning-curve.json).
Full training metrics are stored as `full/RECIPE/metrics.jsonl.gz`. Raw downloaded
components remain in local `.exps`.

AdamW's P@L decreased at 60k and 80k despite improving validation loss. These
endpoints passed the same evaluation audit; no protocol change was found.

## Checkpoint preservation

Project storage root:
`/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-paired-unique-b2048-100k-20260909/`

| Recipe | Relative checkpoint | Bytes | SHA-256 |
|---|---|---:|---|
| AdamW | `baseline/checkpoint-final.pt` | 2,048,443,150 | `138fe1300f6f4132472ae4cf7afd6b6500b7869d22898cf181d9095f2dcd90a5` |
| Setting 3 | `setting3/checkpoint-final.pt` | 1,367,402,225 | `d0f891cfd46e5517e1ce90a29f02bc88646a80023b7cf8e86fc3008311b66412` |

The frozen verifier checked all model and optimizer tensors, exact restoration,
and all four ranks' sampler/RNG states. Independent hashes of the preserved
files match; all ten preserved result receipts match the locally audited artifacts.
See [AdamW final audit](full/baseline/FINAL_AUDIT.json) and
[Setting 3 final audit](full/setting3/FINAL_AUDIT.json).

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
