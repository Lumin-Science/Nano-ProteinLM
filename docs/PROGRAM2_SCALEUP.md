# Program 2 versus the completed H100 R02, and matched 100k presets

The active plan contains **four settings based on our completed H100 R02**:
reset RoPE from 20k to 10k, then add batch balance, sqrt loss and tied embeddings
cumulatively. Every setting retains RMSNorm, learned residual/input routing,
depth-scaled initialization and **FFN width 2048**, with the same optimizer
groups and 100k training contract as the completed R02. Each run starts from
scratch; cumulative refers to recipe changes, not checkpoint continuation.

**R22 FFN narrowing is deferred to [TODO](../TODO.md)**. The earlier unlaunched
five-setting proposal with Program 2's LayerNorm architecture is superseded.
Historical Program 2 results below remain unchanged. None of the four revised
configs has been launched. See the [GPU training plan](PROGRAM2_GPU_PLAN.md).

## Published Program 2 results

Source: `autoresearch-171m-val-loss`, snapshot commit
`1a194f4` (September 6, 2026, 22:34:54 UTC / 6:34:54 PM Toronto).
There are 30 completed methods, 60 one-hour runs and five accepted changes.
R29 is the latest accepted endpoint. R30 EMA is still **pending in this
snapshot**; it is not part of this import. The report is a timestamped snapshot,
not live cluster status.

| Cumulative recipe | Validation loss, mean ± sample SD | Mean full P@L | Mean model tokens | Parameters |
|---|---:|---:|---:|---:|
| Original AdamW baseline | 2.638680 ± 0.013025 | 9.6483% | 639.45M | 170,671,168 |
| R01 Muon | 2.618073 ± 0.009455 | 9.7947% | 581.93M | 170,671,168 |
| R04 batch balance | 2.604151 ± 0.006502 | 9.3699% | 627.05M | 170,671,168 |
| R10 square-root loss weights | 2.594372 ± 0.005778 | **10.5332%** | 626.44M | 170,671,168 |
| R22 FFN width 1536 | 2.590952 ± 0.001323 | 9.8292% | 674.28M | 142,359,616 |
| R29 tied embeddings | **2.580568 ± 0.005442** | 9.5268% | 674.07M | 142,310,464 |

These means and sample SDs were independently recomputed from the 60 published
seed rows. All five accepted changes pass the campaign's stated rule: mean
improvement over the incumbent exceeds the candidate's sample SD. This is a
selection heuristic, not a significance test.

R29 improves validation loss **2.20%** relative to Program 2's original AdamW
baseline. Its contact P@L is slightly below that baseline; R10 has the highest
contact P@L among the accepted recipes. Selection on MLM loss does not establish
an improvement in contact quality.

### Comparison with our completed 100k runs

| Protocol | Program 2 | Completed Fir H100 comparison |
|---|---|---|
| Hardware and training budget | 4 L40S, 1 hour per seed | 4 H100, 100,000 steps (12.01h default / 12.94h R02) |
| Runtime | Torch 2.13.0+cu126, FA2 | Torch 2.13.0+cu130, FA3 |
| Training seeds | 42 and 43 | 20260824, one run per recipe |
| Global batch | 256 | 1,024 |
| Example exposure | About 2.46–2.85M sequences for accepted methods | 102.4M sequences each |
| Token exposure | About 0.58–0.67B for accepted methods | 24.20B each |
| Held-out MLM evaluation | 32 sequences, 994 masked targets | 4,096 sequences, 139,963 masked targets |
| Contact evaluation | Full 20,775 chains | Full 20,775 chains |
| Reported uncertainty | Sample SD over two training seeds | Contact bootstrap CI over chains; no seed SD |

All 60 archived Program 2 run contracts have the same training corpus manifest
and corpus-verification digests as our H100 comparison. The source data match;
exposure and evaluation sample size differ.

Our default reached **2.47436 validation loss / 26.50% P@L**; our R02 reached
**2.43698 / 30.31%**. Program 2's raw losses cannot be used to rank its recipes
against these endpoints: training exposure is roughly 36–42 times smaller for
its accepted recipes, and the MLM sample size differs. Equal contact protocols
also do not remove the training-budget confound.

[Completed H100 results](../reports/fir-171m-100k-20260906/README.md) ·
[Published Program 2 report](../reports/program2/README.md) ·
[Per-method statistics](../reports/program2/methods.tsv) ·
[Import provenance and archive verification](../reports/program2/IMPORT_PROVENANCE.json).

## Program 2 R01 is not our R02

Both use native Muon on the 96 transformer matrices and AdamW on the other
parameters. Both use momentum 0.95, Nesterov updates, five Newton–Schulz steps,
`match_rms_adamw`, Adam betas (0.9, 0.95), epsilon 1e-8 and clipping at 1.0.
Both have 24 layers, model width 768, 12 heads, and FFN width 2048 at this stage.

| Setting | Historical Program 2 R01 | Completed H100 R02 |
|---|---|---|
| Base LR | 0.000326599 | 0.0005 |
| Muon attention LR multiplier → configured peak | 1.0 → 0.000326599 | 0.9 → 0.00045 |
| Muon FFN LR multiplier → configured peak | 1.0 → 0.000326599 | 0.75 → 0.000375 |
| AdamW peak LR | 0.000326599 | 0.0005 |
| Base WD | 0.0183712 | 0.01 |
| Muon WD multiplier → decay | 1.0 → 0.0183712 | 0.75 → 0.0075 |
| Other decayed AdamW parameters | WD 0.0183712 | WD 0.01 |
| Warmup, then constant LR | 554 steps | 1,000 steps |
| Transformer normalization, including full-width Q/K | LayerNorm | Parameter-free RMSNorm |
| Learned residual/input routing | Off | On |
| Depth-scaled residual projection initialization | Off | On |
| RoPE base | 10,000 | 20,000 |
| Vocabulary embeddings | Untied | Untied |
| Cross-rank batch balance | Off | Off |
| Training loss | Sequence mean | Sequence mean |
| Trainable parameters | 170,671,168 | 170,559,856 |
| Execution | BF16 / FA2, batch 256 | BF16 / FA3, batch 1,024 |

Configured Muon group LRs are shown before internal matrix-shape adjustment.
R02's generic `muon_lr_scale: 0.8` is overridden by its explicit 0.9/0.75 groups.
The prediction-head LayerNorm remains in both models. Program 2 also tested
RMSNorm, depth-scaled initialization and learned routing individually, but did
not accept them under its one-hour MLM selection rule. Those outcomes do not
establish how the combined changes behave after 100k steps.

## Exact cumulative changes

| Prepared config | Increment over preceding row | Balanced ranks | Training loss | FFN hidden width | Tied embeddings | Parameters |
|---|---|---|---|---:|---|---:|
| [1. R02 RoPE10k](../configs/program2_h100_100k/r02_rope10k.yaml) | Completed R02 recipe, `rotary_base: 10000.0` | No | Sequence mean | 2048 | No | 170,559,856 |
| [2. + R04 batch balance](../configs/program2_h100_100k/r04_batchbalance.yaml) | `balance_batches_across_ranks: true` | Yes | Sequence mean | 2048 | No | 170,559,856 |
| [3. + R10 sqrt loss](../configs/program2_h100_100k/r10_sqrtloss.yaml) | `training_loss_reduction: sqrt_mask_count` | Yes | Square-root target weights | 2048 | No | 170,559,856 |
| [4. + R29 tied](../configs/program2_h100_100k/r29_tied.yaml) | `tie_word_embeddings: true` | Yes | Square-root target weights | 2048 | Yes | 170,510,704 |

The first row changes only the completed R02's effective RoPE setting; explicit
default fields in the YAML clarify the unchanged FFN/loss/embedding settings.
Compare it with the existing RoPE20k R02 result to evaluate that change. Then
compare rows 2 versus 1, 3 versus 2, and 4 versus 3 for the incremental effects.
Every row also remains comparable with the completed default and R02 endpoints
under the same 100k update and evaluation contract.

**R04** reassigns already-masked whole examples among GPUs, greedily balancing
sequence lengths while preserving equal example counts on every rank. It
preserves the sequences, original masks and labels, RNG state, and mathematical
sequence-mean gradient. Its main mechanism is reducing rank imbalance, so part
of its one-hour gain may come from processing more updates. At fixed 100k steps,
that throughput gain mainly changes runtime; the quality gain may shrink.

**R10** assigns sequence `i`, with `m_i` masked targets and mean target NLL `l_i`,
weight `sqrt(m_i)`: `sum(sqrt(m_i) * l_i) / sum(sqrt(m_i))`. This gives longer
target sets more influence than equal sequence weighting, but less than full
token weighting. Zero-target sequences have zero weight. Held-out MLM evaluation
remains sequence-mean NLL; only the training objective changes.

**R22 is deferred.** Its historical Program 2 result narrows the SwiGLU
intermediate dimension by 25%, removing 28,311,552 parameters. That separate
capacity/compute experiment is not included in the active four settings.

**R29** shares the input embedding matrix and final vocabulary projection,
preserving the separate output bias. Both roles contribute gradients to one
AdamW-owned parameter with one optimizer state. It removes another **49,152**
parameters. In the new four-setting plan it retains both data/loss changes and
the full-width FFNs; this differs from historical Program 2 R29, which inherited
R22's narrower FFNs.

## Matched scale-up contract and controls

All four new configs match the completed runs' base LR **5e-4**, base WD
**0.01**, warmup **1,000**, batch **1,024**, context **512**, seed **20260824**,
mixture, **100,000 optimizer and schedule steps**, BF16/FA3 and 16-hour guard.
All Muon variants additionally match our R02's **actual configured group LRs
and WDs**: attention 0.00045/0.0075, FFN 0.000375/0.0075, decayed AdamW
0.0005/0.01, non-decayed AdamW 0.0005/0. This intentionally replaces historical
Program 2's uniform multipliers of 1.0; the new plan retains our completed
R02's configured optimizer settings. All four FFNs remain width 2048, and no
additional LR or WD rescaling is applied.

The architecture starts from our completed R02's **parameter-free RMSNorm,
learned residual/input routing and depth-scaled initialization**, with RoPE
reset to **10k**. The original Program 2 R01 is no longer an active setting.
R02-RoPE10k → R04 → R10 → R29 defines the four cumulative recipes. The completed
default and RoPE20k R02 remain measured historical references; all four new
results are pending.

Batch 1,024 uses 64 examples/GPU × 4 GPUs × 4 accumulation microsteps. The
imported R10 implementation normalizes weights **across ranks within each
256-example microstep**, then averages four microstep gradients. It does not
normalize once jointly across all 1,024 examples. That retained accumulation
behavior is explicit in the [machine-readable manifest](../configs/program2_h100_100k/manifest.json).

Use the same verified training data and evaluate each final checkpoint with
`--validation-batches 256 --validation-batch-size 16 --validation-context 512`
plus full 20,775-chain contact P@L. More paired training seeds are needed before
claiming training-seed robustness; the prepared seed matches the existing runs.

The imported tests cover complete-example preservation, distributed gradients,
default behavior, full-size parameter counts, tied-gradient summation, optimizer
ownership and checkpoint/contact-feature loading. All **32 tests** passed for
the imported implementation, which is unchanged by this recipe revision.
Full-size meta-model checks confirm the revised parameter counts, full-width
FFNs, base equivalence except RoPE, cumulative config differences, and identical
optimizer LR/WD groups. The new combinations have not yet received an H100
throughput or full training qualification.
