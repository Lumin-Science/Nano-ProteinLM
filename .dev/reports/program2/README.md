# Program 2: validation-loss autoresearch progress

**Figure update, September 8, 2026:** the [run log through R38](runs-through-r38.tsv)
contains 39 completed methods / 78 runs. R30–R38 were all discarded; R29 remains
the best accepted recipe at **2.58056811 ± 0.00544206**. The
[updated curve](validation-loss.png) includes all 38 candidate rounds with
sample-SD error bars across seeds 42 and 43.

The added log is a byte-for-byte copy of
`.dev/ar/ar_l40x4_val_loss_r40_results.tsv`; despite the filename, it contains no
R39 or R40 results. Its first 60 rows match [the original run log](runs.tsv).
SHA-256: `ac3225ea909e2aceffbb1b2b8bd92817f14e03b6b3be145e547fcbd52164ff75`.
The report, per-method table, and audit archive below describe the original
R29 snapshot; only the TSV supplies the added R30–R38 results.

Snapshot: **2026-09-06T22:34:54Z**. **30 completed methods / 60 full-hour runs / 5 accepted changes.** The completed runs represent 240 GPU-hours of scheduled training, excluding preflight and evaluation.

Current accepted recipe: **R29 shared input/output embeddings**, mean validation loss **2.58056811 ± 0.00544206** (sample SD across seeds 42 and 43). This is **0.05811235 lower (2.20%)** than the original AdamW baseline. Accepted code: [`fc0a125`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC/commit/fc0a1253a8919615714fd7f95bbe1a7a00235172).

Program 2 selects on frozen held-out MLM loss. Contact P@L and training loss are diagnostics; the repository’s separate P@L campaign has its own results and selection rules.

## Research curve

![Validation-loss search history with five numbered improvements.](validation-loss.png)

All 38 rounds are shown; points and bars are two-seed means ± sample SD.
The line follows the retained recipe. Changes 4–5 use roughly 142M parameters
and predate the current fixed-size task. See the [source TSV](runs-through-r38.tsv)
and [current task definition](../../../task/171m-validation-loss.md) for their distinct protocols.

## R29 improvement

R29 ties the input embedding to the final vocabulary projection. Both roles contribute gradients to one AdamW-owned parameter; the rest of R22 is retained. It removes 49,152 parameters, giving **142,310,464** trainable parameters.

| Seed | R22 validation loss | R29 validation loss | Reduction |
|---|---:|---:|---:|
| 42 | 2.59001710 | 2.58441623 | 0.00560087 |
| 43 | 2.59188774 | 2.57671999 | 0.01516775 |
| **Mean** | **2.59095242** | **2.58056811** | **0.01038431 (0.40%)** |

The gain **0.01038431** exceeds R29’s sample SD **0.00544206**, so it passes the exact acceptance rule. All 32 tests, full-size L40S qualification, checkpoint sharing and optimizer-state checks, four-rank data replay, and frozen evaluations passed.

| Seed | Final-100 train loss | Full contact P@L | Steps | Tokens (M) |
|---|---:|---:|---:|---:|
| 42 | 2.641091 | 0.090178 | 11160 | 674.847 |
| 43 | 2.645231 | 0.100358 | 11129 | 673.285 |

## Accepted sequence

Each accepted change is applied to the previous accepted recipe.

| Method | Change | Mean loss | Sample SD | Gain vs. incumbent | Commit |
|---|---|---:|---:|---:|---|
| baseline | Original AdamW baseline | 2.638680 | 0.013025 | +0.000000 | baseline |
| r01_muon | Hybrid Muon | 2.618073 | 0.009455 | +0.020608 | [`03949ae`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC/commit/03949ae1d927df901a5c4ea3033383406d0c3dfe) |
| r04_batchbalance | Balance masked batches across ranks | 2.604151 | 0.006502 | +0.013922 | [`0a2b17e`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC/commit/0a2b17e6899da79a87d2592bedf092ade9a3ce2a) |
| r10_sqrtloss | Square-root target-count weighting | 2.594372 | 0.005778 | +0.009779 | [`bc06899`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC/commit/bc06899d1873489d88c0e57e7f9e30a18fb8b70b) |
| r22_ffn1536 | FFN width 1536 | 2.590952 | 0.001323 | +0.003420 | [`19e9a97`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC/commit/19e9a97309ae2cc83f0e115280fbb5f03fbcf5ab) |
| r29_tied | Shared input/output embeddings | 2.580568 | 0.005442 | +0.010384 | [`fc0a125`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC/commit/fc0a1253a8919615714fd7f95bbe1a7a00235172) |

## All completed rounds

Positive gain means lower loss than the incumbent **at that round**. A lower mean alone does not qualify: gain must exceed that candidate’s sample SD.

| Method | Change | Axis | Seed 42 | Seed 43 | Mean ± SD | Gain | Decision |
|---|---|---|---:|---:|---:|---:|---|
| baseline | Original AdamW baseline | Baseline | 2.629470 | 2.647891 | 2.638680 ± 0.013025 | +0.000000 | **baseline** |
| r01_muon | Hybrid Muon | Algorithm | 2.624758 | 2.611387 | 2.618073 ± 0.009455 | +0.020608 | **keep** |
| r02_rmsnorm | Transformer RMSNorm | Model | 2.608600 | 2.624505 | 2.616552 ± 0.011247 | +0.001520 | **discard** |
| r03_tokenmean | Global token-mean loss | Loss | 2.602256 | 2.615474 | 2.608865 ± 0.009347 | +0.009208 | **discard** |
| r04_batchbalance | Balance masked batches across ranks | Data / systems | 2.599553 | 2.608748 | 2.604151 ± 0.006502 | +0.013922 | **keep** |
| r05_depthinit | Depth-scaled initialization | Model | 2.600757 | 2.631933 | 2.616345 ± 0.022045 | -0.012194 | **discard** |
| r06_splitqkv | Independent Q/K/V orthogonalization | Algorithm | 2.618506 | 2.626523 | 2.622515 ± 0.005668 | -0.018364 | **discard** |
| r07_maskcount | Rounded mask counts | Data | 2.613311 | 2.618730 | 2.616021 ± 0.003831 | -0.011870 | **discard** |
| r08_residual | Learned residual/input routes | Model | 2.607424 | 2.611956 | 2.609690 ± 0.003205 | -0.005539 | **discard** |
| r09_context | 128-to-512 context curriculum | Data | 2.603465 | 2.621608 | 2.612536 ± 0.012829 | -0.008386 | **discard** |
| r10_sqrtloss | Square-root target-count weighting | Loss | 2.590286 | 2.598458 | 2.594372 ± 0.005778 | +0.009779 | **keep** |
| r11_mask20 | 20% masking | Data | 2.589194 | 2.598755 | 2.593974 ± 0.006761 | +0.000398 | **discard** |
| r12_headnorm | Headwise Q/K normalization | Model | 2.597066 | 2.596978 | 2.597022 ± 0.000063 | -0.002650 | **discard** |
| r13_polarexpress | Polar Express orthogonalization | Algorithm | 2.582039 | 2.615145 | 2.598592 ± 0.023410 | -0.004220 | **discard** |
| r14_headgate | Query-dependent attention gates | Model | 2.604305 | 2.614624 | 2.609464 ± 0.007297 | -0.015092 | **discard** |
| r15_adamuon | AdaMuon | Algorithm | 2.608915 | 2.609879 | 2.609397 ± 0.000682 | -0.015025 | **discard** |
| r16_geglu | GEGLU FFNs | Model | 2.590316 | 2.612368 | 2.601342 ± 0.015593 | -0.006970 | **discard** |
| r17_muonscale | Double Muon update amplitude | Algorithm | 2.579673 | 2.602124 | 2.590898 ± 0.015875 | +0.003473 | **discard** |
| r18_tokendrop | ESM-2 mask-embedding rescaling | Model / data | 2.579772 | 2.602810 | 2.591291 ± 0.016290 | +0.003081 | **discard** |
| r19_bertcorrupt | 80/10/10 corruption | Data | 2.613637 | 2.617784 | 2.615710 ± 0.002932 | -0.021339 | **discard** |
| r20_lastsubset | Compute final FFN/head at targets only | Systems | 2.591641 | 2.599351 | 2.595496 ± 0.005452 | -0.001124 | **discard** |
| r21_batch32 | Global batch 128 | Data / optimization | 2.601979 | 2.593341 | 2.597660 ± 0.006108 | -0.003288 | **discard** |
| r22_ffn1536 | FFN width 1536 | Model / compute | 2.590017 | 2.591888 | 2.590952 ± 0.001323 | +0.003420 | **keep** |
| r23_zloss | Masked logit z-loss | Loss | 2.598989 | 2.600840 | 2.599915 ± 0.001310 | -0.008962 | **discard** |
| r24_ffn1024 | FFN width 1024 | Model / compute | 2.595070 | 2.601190 | 2.598130 ± 0.004327 | -0.007178 | **discard** |
| r25_ffnprofile | Profile-guided FFN allocation | Model / data | 2.595098 | 2.593249 | 2.594174 ± 0.001308 | -0.003221 | **discard** |
| r26_valueres | First-layer value residuals | Model | 2.597137 | 2.603569 | 2.600353 ± 0.004547 | -0.009401 | **discard** |
| r27_cwd | Cautious Muon weight decay | Algorithm | 2.598741 | 2.599409 | 2.599075 ± 0.000472 | -0.008123 | **discard** |
| r28_ema | EMA of weights | Algorithm | 2.578740 | 2.590970 | 2.584855 ± 0.008648 | +0.006098 | **discard** |
| r29_tied | Shared input/output embeddings | Model | 2.584416 | 2.576720 | 2.580568 ± 0.005442 | +0.010384 | **keep** |

## What the results support

- **Algorithm:** hybrid Muon was accepted. Alternative orthogonalization, adaptivity, and cautious decay failed their paired tests. R28 EMA lowered both seed losses but its 0.006098 mean gain did not exceed its 0.008648 SD; it was discarded.
- **Data and execution:** balancing the same masked batch across ranks was accepted. Changes to mask count, mask rate, corruption, context curriculum, and smaller batches did not qualify.
- **Loss:** globally normalized square-root target-count weighting was accepted. Global token averaging narrowly missed the threshold; added logit z-loss did not help.
- **Model:** FFN width 1536 and shared vocabulary embeddings were accepted. Further narrowing to 1024 processed more tokens but worsened loss; layerwise width allocation and value residuals also failed.

These are results under the fixed one-hour protocol and 32-sequence validation sample, not claims about longer training or general protein-task accuracy. SD measures variation across two training seeds; it is not a confidence interval or a formal significance test.

## Work in progress and preflight-only branches

**R30: EMA on accepted R29.** Both seeds run from scratch, with EMA decay 0.999 starting after update 554. Shared weights are averaged once, all averaging cost counts inside the hour, and only the predeclared EMA is evaluated. R30 must beat the new R29 mean of 2.580568107776344 by more than its own sample SD. Its result is **pending**.

- Seed 42, kn081: running, last logged step **6370** at this snapshot.
- Seed 43, kn056: running, last logged step **6300** at this snapshot.

The next automated check is scheduled for **2026-09-06 23:09:18 UTC / 7:09:18 p.m. Toronto**. See [running.json](running.json) for the timestamped snapshot; this document is not live telemetry.

Two branches were stopped after preflight and have no model-quality result: batched Muon saved only about 0.7 ms in an optimizer-only benchmark, and FFN activation normalization added about 18.47% to synthetic step cost without supporting its intended gradient-balance mechanism. They are excluded from the completed-round count.

## Fixed protocol and accepted recipe

- Seeds 42 and 43 run simultaneously on kn081 and kn056, four L40S GPUs per seed, from scratch, at most 171M trainable parameters, for a synchronized 3,600-second training budget.
- Every learning-rate group warms linearly for exactly 554 optimizer steps and then remains at its fixed peak. The corpus, mixture, tokenizer, dependency lock, hardware class, and evaluators are frozen.
- Selection uses `sequence_mean_nll` over 8 batches × 4 sequences, context 512, evaluation seed 20260821: 32 sequences and 994 masked targets. Acceptance is `incumbent_mean - candidate_mean > candidate_sample_SD` with `ddof=1`.
- Current model: 24 layers, width 768, 12 heads, SwiGLU FFN width 1536, whole-projection Q/K LayerNorm, RoPE base 10,000, and tied vocabulary embeddings.
- Native Muon handles 96 transformer matrices; AdamW handles embeddings, prediction head, and scalar parameters. Global batch 256, context 512, 15% masking, balanced ranks, and square-root target-count weighting remain accepted.

Full contract: [program2.md](../../../docs/archive/program2.md). Accepted seed configs: [seed 42](../../../configs/archive/program2/r29_tied_seed42.yaml), [seed 43](../../../configs/archive/program2/r29_tied_seed43.yaml). The runner requires `CONFIG` to select these explicitly; its default belongs to the separate P@L campaign.

## Evidence and research provenance

- [runs.tsv](runs.tsv): all 60 completed run rows, including loss, P@L, time, tokens, memory, and decisions. `NA` commit entries mean the candidate was uncommitted during execution; accepted commits are mapped in the method summaries and decision receipts.
- [methods.tsv](methods.tsv): per-method statistics and diagnostics, independently recomputed from run rows when exporting this report.
- [experiments.json](experiments.json): proposals, source/config hashes, references, and final decisions. Historical proposal statuses are preserved; the result summary and decision receipt determine the outcome.
- [JOURNAL.md](JOURNAL.md): full chronological research reasoning, implementation qualifications, and outcomes, including rejected ideas.
- [audit.tar.gz](audit.tar.gz), with [SHA-256 manifest](audit-manifest.json): archived candidate code/configs/patches, preflight evidence and retained licenses, local orchestration/verification helpers, and all completed run summaries and receipts. Some early candidates are represented by configs and accepted Git history rather than a full source snapshot. Active R30 is explicitly unaccepted. The archive contains no model checkpoints or corpus data; host-specific paths in receipts identify original provenance.

Literature informed hypotheses rather than establishing outcomes here. Examples include [Muon](https://arxiv.org/abs/2502.16982), [Hydraulis](https://arxiv.org/abs/2412.07894), [weight tying](https://aclanthology.org/E17-2025/) and the [ESM-2 implementation](https://github.com/facebookresearch/esm/blob/main/esm/model/esm2.py), [value residual learning](https://aclanthology.org/2025.acl-long.1375/), and [constant-rate weight averaging](https://arxiv.org/abs/2602.03702). The journal records each adaptation and its limitations.
