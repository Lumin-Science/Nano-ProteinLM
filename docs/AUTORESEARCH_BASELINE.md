<div class="ai">

# Karpathy-style sequential AutoResearch

</div>

<div class="ai">

Our implementation of Karpathy-style sequential search proposes one change, trains and evaluates the candidate, keeps or discards it, and repeats from the retained recipe. Candidates are explored sequentially; training and evaluation can use multiple GPUs. This page describes the method's pipeline, two-seed measurements, improvement criteria and historical results. The shared design boundaries, fixed round allowance, per-round compute and final evaluation budget are defined in [the protocol](autoresearch.md); [LEADERBOARD.md](LEADERBOARD.md) collects the recorded measurements.

</div>

<div class="ai">

## Running the example loop

</div>

<div class="ai">

[autoresearch/program.md](../autoresearch/program.md) contains the executable instructions for baseline measurement, iteration, comparison and logging. The included task entry points, [171M validation loss](../tasks/171m-validation-loss.md) and [171M contact P@L](../tasks/171m-p-at-l.md), retain the historical one-hour/four-L40S measurement profile. They measure each candidate with two training seeds through the standard training/evaluation APIs. The current [benchmark protocol](autoresearch.md#search-budget) specifies 20 minutes on four H100s per round; its hardware and time settings must be applied when using this method in that benchmark.

</div>

<div class="ai">

![Example sequential AutoResearch loop: evaluate a baseline, propose a change, train and evaluate two seeds, keep or discard, record and repeat.](../.dev/reports/readme-figures-20260914/autoresearch-loop.png)

</div>

<div class="ai">

The diagram shows our two-seed loop followed by the owner-run scale-up test of the selected recipe. Under the H100 protocol, each candidate comparison uses **two of the 72 rounds**, or **8/3 H100 GPU-hours**; measuring the starting recipe also consumes two rounds. The historical L40S commands below spend **8 L40S GPU-hours** per candidate. Those recorded runs retain their original hardware and budgets.

</div>

<div class="ai">

| Method setting | Included historical L40S implementation |
|---|---|
| Training seeds per candidate | **42 and 43**, both trained from scratch |
| Training per seed | **1 hour on 4 L40S GPUs**, FlashAttention-2 |
| Schedule | **554 warmup steps**, followed by constant peak learning rate |
| Cost per complete candidate | **2 training runs = 8 L40S GPU-hours**, excluding setup and evaluation |
| Checkpoint | Final checkpoint from each run; no periodic checkpointing or evaluation |
| MLM evaluation | **32 fixed sequences**, context 512 |
| Contact evaluation | **20,775 chains**, one shared probe per checkpoint, 32 workers across four GPUs, 5,000 chain-bootstrap replicates |
| Candidate score | Mean of the selected metric across the two seeds, with per-seed values and sample SD retained |

</div>

<div class="ai">

Prepare the frozen corpus with `bash runs/setup.sh --training-shards 7` in a dedicated `DATA_ROOT`. The general setup default prepares 30 training shards; these task entry points use the frozen seven-shard selection. From the repository root, choose the measurement command for the selected objective.

</div>

<div class="ai">

```bash
# Validation-loss objective
bash tasks/171m-validation-loss_ar.sh configs/default.yaml experiment-001

# Contact-P@L objective
bash tasks/171m-p-at-l_ar.sh configs/default.yaml experiment-p-at-l-001
```

</div>

<div class="ai">

Each command snapshots the recipe, trains and evaluates both seeds, and writes their aggregate metrics to `summary.json` under the experiment's `OUTPUT_ROOT` directory. Use a fresh experiment name for every candidate. Both seeds must complete training and evaluation before the method compares that candidate with the incumbent. Preserve the source revision, resolved configuration, data receipts, commands, checkpoints, per-seed metrics and keep/discard decision.

</div>

<div class="ai">

To run the included validation-loss example, give your coding agent this instruction:

</div>

<div class="ai">

```text
Read autoresearch/program.md and start autoresearch for tasks/171m-validation-loss.md.
```

</div>

<div class="ai">

For contact P@L as the example objective:

</div>

<div class="ai">

```text
Read autoresearch/program.md and start autoresearch for tasks/171m-p-at-l.md.
```

</div>

<div class="ai">

The current program averages the task's metric over training seeds and compares seed-level 95% confidence intervals. Orient reward so higher is better, then keep a candidate only when `candidate.ci95_low > incumbent.mean` and `candidate.mean > incumbent.ci95_high`. Use negative validation loss for the loss-task comparison and retain raw loss in reports; P@L already increases with improvement. Record every candidate and keep/discard decision. These are our method's improvement criteria. Seed-level intervals are distinct from contact-chain bootstrap intervals.

</div>

<div class="ai">

The manual scale-up example in [EVALUATION.md](EVALUATION.md#manual-test-of-progress) also uses seeds 42 and 43, training the selected recipe and reference to 24,200,224,761 model tokens per seed on four H100s. This is a concrete implementation of the final evaluation; a comparison of AutoResearch methods uses the common final seed list and repeat count declared by that benchmark.

</div>

<div class="ai">

## Historical 38-round example

</div>

<div class="ai">

The completed campaign evaluated a baseline and 38 candidate recipes, each with seeds 42 and 43 and one training hour on four L40S GPUs per seed: **78 runs, or 312 L40S GPU-hours**, excluding setup and evaluation. The historical records call each two-seed candidate comparison a round; this campaign predates the current 72-round H100 protocol. Its earlier acceptance rule kept a candidate when its mean validation-loss reduction exceeded that candidate's own two-seed sample SD. Preserve those recorded decisions when reproducing the curve; they were not generated by the current interval-based rule.

</div>

<div class="ai">

The [improvement figure in the README](../README.md#benchmarking-agentic-autoresearch-systems) shows trial means and sample-SD error bars in orange and the retained recipe in blue. Numbers 1–5 mark the accepted changes: Muon, batch balance, sqrt loss, narrower FFNs and tied embeddings. R30–R38 were discarded, leaving R29 as the final retained historical recipe. See the [historical search table](#detailed-search-results) and [full campaign record](../.dev/reports/program2/README.md#numbered-improvements).

</div>

<div class="ai">

Changes 4–5 use roughly 142M parameters and predate the current ±5% size bound. Their original rules remain in the [archived instructions](archive/program2.md). The later 171M scale-up comparison skips the FFN reduction and applies tied embeddings directly to Setting 3. Its Muon recipe also includes RMSNorm, residual routing and depth-scaled initialization, so the search-time and scale-up rows are not identical single-component ablations. See the [historical test table](#detailed-scale-up-results).

</div>

<div class="ai">

## Understanding the three scale-up improvements

</div>

<div class="ai">

<aitofix resolved>Here add quick understanding of all 3 improvement with related works listed, I want some brief level of intros here and referred to a more comprehensive documents. Fixed: Added the three brief method introductions below, primary related-work links and links to the full method guide; moved them here with the sequential-search example.</aitofix>

</div>

<div class="ai">

**1. Muon recipe.** Transformer matrices use Muon while embeddings and the MLM head retain AdamW, following the optimizer partition described in the [Muon implementation](https://github.com/KellerJordan/Muon). The tested package also changes transformer normalization to parameter-free RMSNorm, mixes the current hidden stream with the original embeddings at each layer, and scales residual-projection initialization with depth. [RMSNorm (Zhang and Sennrich, 2019)](https://arxiv.org/abs/1910.07467) motivates normalization by root mean square without mean subtraction. The experiment measures the combined package, so it cannot assign the gain to Muon alone. See [the full component and optimizer settings](leaderboard/BEST_RECIPE_22_09_26.md#2-exactly-what-differs-from-the-baseline).

</div>

<div class="ai">

**2. Batch balance.** Proteins have different lengths, so equal protein counts can leave GPUs with unequal token workloads. Redistributing already-masked examples balances non-padding token counts while preserving the sampled examples, labels and number of examples per rank. This can reduce time spent waiting at gradient synchronization; [PyTorch's DDP discussion of skewed processing speeds](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#skewed-processing-speeds) describes the underlying workload-balancing problem. Token count is a proxy for work, not an exact FLOP estimate. See [the partitioning procedure and loss normalization](leaderboard/BEST_RECIPE_22_09_26.md#3-batch-balance-equalize-work-across-gpus).

</div>

<div class="ai">

**3. Sqrt loss.** Weight each protein's mean masked-token loss by the square root of its number of masked targets, then normalize by the sum of those weights. Proteins with more targets contribute more than under equal-protein weighting, but less than under equal-target weighting. [BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805) is related background for masked-language-model pretraining; the sqrt weighting is the recipe studied here, not a result attributed to BERT. Validation retains equal-protein weighting, so the evaluation metric stays fixed. See [the formula, worked example and distributed normalization](leaderboard/BEST_RECIPE_22_09_26.md#4-sqrt-loss-change-protein-weighting-not-the-validation-metric).

</div>

<div class="ai">

These three historical steps describe the earlier scale-up comparison. The present default additionally uses separate Q/K/V Muon updates following later P@L optimization and the [three-arm CCK study](../.dev/reports/cck-contact-ablations-100k-20260919/DEFAULT_PROMOTION.md). Historical measurements keep their original recipes; they are not retroactively scores for the current default.

</div>

<div class="ai">

## Detailed search results

</div>

<div class="ai">

The earlier sequential campaign completed 38 candidate rounds plus a baseline, with two one-hour runs on four L40S GPUs per candidate: **78 runs and 312 L40S GPU-hours**. Each two-seed candidate therefore cost 8 GPU-hours. Values are mean ± sample SD over seeds 42 and 43, from the [method statistics](../.dev/reports/program2/methods.tsv); the [historical example above](#historical-38-round-example) explains its acceptance rule and figure.

</div>

<div class="ai">

| Historical retained recipe | Search validation loss ↓ | Approximate parameters |
|---|---:|---:|
| AdamW baseline | 2.638680 ± 0.013025 | 171M |
| 1: Muon | 2.618073 ± 0.009455 | 171M |
| 2: + batch balance | 2.604151 ± 0.006502 | 171M |
| 3: + sqrt loss | 2.594372 ± 0.005778 | 171M |
| 4: + narrower FFN | 2.590952 ± 0.001323 | 142M |
| 5: + tied embeddings | 2.580568 ± 0.005442 | 142M |

</div>

<div class="ai">

The 142M endpoints predate the current ±5% parameter rule and are outside that rule. Historical search and scale-up recipes also differ: the scale-up Muon row includes RMSNorm, routing and initialization changes, and its tied-embedding row applies tying directly to the 171M sqrt-loss recipe, skipping FFN narrowing. Preserve these distinctions when attributing improvements.

</div>

<div class="ai">

## Detailed scale-up results

</div>

<div class="ai">

These are completed **100k-step, single-seed** comparisons. Each run used four H100s, seed 20260824, batch 1,024, context 512, base LR 5e-4, weight decay 0.01 and 1,000 warmup steps, consuming **24,200,224,761 model tokens**. Times exclude evaluation. The P@L intervals bootstrap 20,775 contact chains and do not estimate training-seed uncertainty.

</div>

<div class="ai">

| Historical recipe | Validation loss ↓ | P@L ↑ | P@L 95% CI | Training time |
|---|---:|---:|---:|---:|
| Baseline: ESMC-like AdamW | 2.47436 | 26.505% | 26.295–26.719% | 12h 01m |
| 1: + Muon recipe | 2.43781 | 30.165% | 29.936–30.394% | 12h 58m |
| 2: + batch balance | 2.43872 | 30.715% | 30.487–30.948% | 12h 34m |
| **3: + sqrt loss (historical default)** | **2.41872** | **32.682%** | **32.447–32.920%** | **12h 35m** |
| 5: + tied embeddings | 2.42304 | 31.884% | 31.651–32.123% | 12h 33m |

</div>

<div class="ai">

The first improvement row adds the full Muon/RMSNorm/routing/initialization recipe, so it is not a Muon-only ablation. Subsequent rows are cumulative. See [recipe details](leaderboard/BEST_RECIPE_22_09_26.md) and [run records](../.dev/reports/fir-r02-rope10k-100k-20260906/README.md). The current default additionally uses separate Q/K/V Muon updates; that later change is supported by the [September 2026 CCK comparison](../.dev/reports/cck-contact-ablations-100k-20260919/DEFAULT_PROMOTION.md) and is not included in this historical table.

</div>

<div class="ai">

### L40S 100k-step component ablations

</div>

<div class="ai">

The later P@L search added separate Q/K/V Muon updates and query centering with RMS restoration in the final eight layers to the previous default. Its three-arm comparison trained each setting from scratch for **100,000 updates on four L40S GPUs**, using FA2, seed 42, global batch 1,024, context 512, 170,559,856 parameters, base LR 5e-4 and 1,000 warmup steps followed by constant LR. Each run consumed **102,400,000 sequences and 24,196,983,520 non-padding model tokens** from the expanded 111-shard corpus, with no repeated source epochs. This historical endpoint differs from the manual scale-up procedure's 24,200,224,761-token target.

</div>

<div class="ai">

| Recipe | Separate Q/K/V Muon | Query centering + RMS restoration | Validation loss ↓ | P@L ↑ | P@L chain-bootstrap 95% CI | Training hours on 4 L40S |
|---|---|---|---:|---:|---:|---:|
| **Current default / human+ai baseline-09-26** | On | Off | **2.410035** | **33.449770%** | 33.214952–33.679334% | 35.4727 |
| Full P@L search recipe | On | On | 2.413881 | 33.485122% | 33.251392–33.716149% | 37.7056 |
| No separate Q/K/V ablation | Off | On | 2.416150 | 32.767462% | 32.535210–32.998056% | 37.2700 |

</div>

<div class="ai">

Final-checkpoint evaluation used the same 4,096 MLM validation sequences and all 20,775 contact chains, with 5,000 chain-bootstrap replicates. Each run completed all ten scheduled P@L evaluations at 10k-step intervals; its final checkpoint also had its highest observed P@L. The intervals above measure variation across contact chains, while training-seed SD is unavailable because each setting used only seed 42. Training times exclude setup and evaluation. See the [comparison receipt](../.dev/reports/cck-contact-ablations-100k-20260919/COMPARISON.json), [completion receipt](../.dev/reports/cck-contact-ablations-100k-20260919/COMPLETION_SUMMARY.json) and [study report](../.dev/reports/cck-contact-ablations-100k-20260919/STATUS.md).

</div>

<div class="ai">

With query centering and RMS restoration enabled, separate Q/K/V updates improved observed P@L by **0.717660 percentage points**. Adding centering and RMS restoration to the split-Q/K/V recipe changed P@L by **+0.035352 points** and validation loss by **+0.003846**. The owner selected the split-Q/K/V recipe without centering or RMS restoration as the current default; see the [promotion decision](../.dev/reports/cck-contact-ablations-100k-20260919/DEFAULT_PROMOTION.md). There is no matched arm with both additions disabled, so this study cannot isolate the standalone split-Q/K/V gain over the previous default. These are single-seed observations; paired significance was not evaluated.

</div>

<div class="ai">

### Longer-training reference

</div>

<div class="ai">

| Model | P@L ↑ | Estimated training FLOPs |
|---|---:|---:|
| ESMC-600M | 58.031% | 2.491 × 10²² |
| ESMC-300M | 53.867% | 1.480 × 10²² |
| **AutoResearch 171M** | **46.264%** | **2.334 × 10²¹** |

</div>

<div class="ai">

The longer-trained 171M recipe reaches **46.264% P@L**; see its [run record](../.dev/reports/nibi-setting3-stage2-b2048-300k-20260911/README.md). All three models use the same frozen 20,775-chain contact evaluation, but their training corpora and compute budgets differ. These are capability references, not entries in the iso-token agent leaderboard. FLOPs are [estimates with stated token and context assumptions](../.dev/reports/readme-figures-20260914/README.md#training-compute-estimates).

</div>

<div class="ai">

## Evidence and plot regeneration

</div>

<div class="ai">

The [78-run TSV through R38](../.dev/reports/program2/runs-through-r38.tsv), [per-method statistics through R29](../.dev/reports/program2/methods.tsv), and [original import and audit](../.dev/reports/program2/README.md) preserve the campaign. The [scale-up run records](../.dev/reports/fir-r02-rope10k-100k-20260906/README.md) and [best-versus-baseline guide](leaderboard/BEST_RECIPE_22_09_26.md) explain the subsequent comparison. The [detailed scale-up results](#detailed-scale-up-results) include the longer-training reference; [LEADERBOARD.md](LEADERBOARD.md) collects the recorded measurements.

</div>

<div class="ai">

The [matched batch-2,048 comparison](../.dev/reports/readme-figures-20260914/README.md), its [AdamW training records](../.dev/reports/nibi-baseline-b2048-100k-eval10k-20260908/README.md) and [improved-recipe records](../.dev/reports/nibi-setting3-b2048-100k-eval10k-20260908/README.md) are separate from the batch-1,024 comparison above.

</div>

<div class="ai">

To regenerate the campaign curve from the repository root without changing the training environment:

</div>

<div class="ai">

```bash
python3 -m venv /tmp/nano-esmc-plot
/tmp/nano-esmc-plot/bin/python -m pip install 'matplotlib==3.11.1'
/tmp/nano-esmc-plot/bin/python .dev/scripts/plot_autoresearch_history.py
```

</div>

<div class="ai">

The script validates seed means, sample SDs and historical keep/discard decisions, then writes PNG and SVG files to `.dev/reports/program2/`. It reads `.dev/reports/program2/runs-through-r38.tsv` by default; pass `--input path/to/results.tsv` for another supported per-run or per-method log. Its retained-recipe line is teal, trial points have sample-SD bars, and amber labels flag the two roughly 142M changes. The README uses the separately styled [README figure and source record](../.dev/reports/readme-figures-20260914/README.md).

</div>
