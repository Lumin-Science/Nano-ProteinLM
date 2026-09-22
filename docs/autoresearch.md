<div class="ai">

# Benchmarking Agentic AutoResearch Systems

</div>

<div class="ai">

NanoProteinLM benchmarks an agent's ability to discover better protein-model training recipes. There are two search tracks: fixed rounds with a predefined validation-loss reward, and a total compute budget with an agent-chosen search reward. Both end with **three frozen recipe submissions, three tests at equal training-token budgets, and a reported top-1 result**. Any search strategy can participate; the included [Karpathy-style sequential search](autoresearch-sequential-search.md) is one baseline.

</div>

<div class="ai">

## Shared experimental contract

</div>

<div class="ai">

Before search starts, the benchmark owner publishes a versioned task with the starting code and recipe, corpus manifests and source mixture, tokenizer, model-size bound, environment, evaluation assets, seed policies, budget accounting and final ranking metric. All agents in a comparison receive the same contract. The 171M task family keeps actual trainable parameters within **±5% of the original 171M model**, prohibits pretrained weights and held-out training, and protects evaluation code and input receipts. Training-recipe, model and implementation changes are permitted within those boundaries. A larger-model transfer study needs its own shared contract.

</div>

<div class="ai">

The final scale-up objective is fixed by the owner before search, even when the agent chooses its own search reward. Measure both sequence-mean MLM validation loss (lower is better) and contact P@L (higher is better); specify which one determines the top-1 setting and agent ranking. Also publish the confirmation seed count **N**, its seed list, and the scale-up seed count **N_test** and seed list before launch. The owner specifies these values for each task version before launch. Use at least two seeds wherever a mean ± sample standard deviation is required.

</div>

<div class="ai">

Pin the training data available during search and the common scale-up data before any agent starts. The scale-up corpus must support every submitted setting's exposure budget: verify per-source coverage, source mixture and sampler behavior before training, and prepare more data for all settings together if needed. Record unique proteins, sampled sequences and non-padding model tokens separately. See [data coverage](data-coverage.md). The existing seven-shard example tasks retain their own historical data contract; that download is not a capacity guarantee for a new scale-up benchmark.

</div>

<div class="ai">

## Search tracks and hardware budgets

</div>

<div class="ai">

GPU-hours are **GPU count × elapsed training hours**, summed across all jobs, including parallel jobs. They are not the elapsed time of a four-GPU machine. H100 and L40S results use separate hardware profiles; the allowances below do not assert equal throughput between devices.

</div>

<div class="ai">

| Track | Hardware | Training allowance per round | GPU-hours per round | Maximum rounds | Total search training allowance |
|---|---|---:|---:|---:|---:|
| 1: fixed rounds | 4 × H100 | 20 minutes on all four GPUs | 4/3 H100 GPU-hours | 96 | 128 H100 GPU-hours (32 four-GPU hours) |
| 1: fixed rounds | 4 × L40S | 60 minutes on all four GPUs | 4 L40S GPU-hours | 96 | 384 L40S GPU-hours (96 four-GPU hours) |
| 2: total budget | H100; four-GPU reference allocation | Agent chooses | Agent chooses | No fixed round cap | 144 H100 GPU-hours (36 × 4; 36 four-GPU hours) |
| 2: total budget | L40S | Not specified | Not specified | Not specified | No L40S allowance defined yet |

</div>

<div class="ai">

### Track 1: fixed rounds and validation-loss reward

</div>

<div class="ai">

Every round trains one candidate from scratch under the same hardware profile and per-round training conditions. The reward is the owner's predefined held-out **sequence-mean MLM validation loss**; lower is better. Freeze the validation examples, masking, context and metric before search. The existing small-budget evaluator uses 32 validation sequences; a benchmark using a different sample count must declare it for every agent. Run at most 96 rounds. Any agent-run baseline, repeat or failed training attempt uses a round and is recorded; seed repeats during search are not an extra free allowance. A shared owner-supplied baseline can be measured separately and must be identified as such.

</div>

<div class="ai">

At search completion, freeze the candidate with the lowest valid search loss and submit it for **N-seed confirmation at the same per-round budget**. Retrain that fixed candidate from scratch on the predeclared confirmation seeds and report the arithmetic **mean ± sample standard deviation**, together with every seed value. Rank the Track 1 search-time leaderboard by this confirmed mean loss. The best individual seed is not the score. Confirmation is owner-run after submissions are frozen; its additional training cost is **N × the per-round GPU-hours** and is reported separately from the 96-round search allowance.

</div>

<div class="ai">

### Track 2: total search budget and agent-chosen reward

</div>

<div class="ai">

The only search scheduling constraint is a total of **36 × 4 = 144 H100 GPU training hours**. There is no fixed per-round duration or 96-round cap. Within the shared scientific contract, the agent chooses the number and length of trials, replication, search strategy and reward, and may revise its reward while recording the definition and changes. The agent is told in advance that its recipes will be judged at the common token budget calibrated to a **48 H100 GPU-hour scale-up run**. It must optimize for that published final objective.

</div>

<div class="ai">

Log the chosen reward, its direction, evaluation data and observed values. Agent-chosen proxy rewards cannot be compared numerically across agents when their definitions differ. The Track 2 search-time leaderboard records those rewards and consumed compute; the common scale-up evaluation supplies the comparable ranking. Track 1's N-seed search confirmation is not an additional requirement for Track 2.

</div>

<div class="ai">

### Accounting and reproducibility

</div>

<div class="ai">

Count all training attempts, warm starts, retries and exploratory training against the relevant search allowance, including training that is later discarded. Track 1 requires from-scratch rounds; Track 2 may allocate its training budget across continuations. Setup, final checkpoint writing and evaluation are outside the training clock but their GPU time, wall time and costs must be reported separately, alongside agent model/version, inference usage and cost. Record the exact timer definition and periodic-evaluation/checkpoint policy in the task so the same work is charged for every agent. No additional unreported training is permitted under an evaluation or setup label.

</div>

<div class="ai">

Preserve the code revision, resolved config, data and environment receipts, seed, training tokens, steps, elapsed training time, metrics, checkpoint identity and outcome for every attempt. Record incomplete runs and any resumed compute. Confirmation, scale-up, shared controls and agent inference each have separate cost totals; a search-budget number must not be presented as the total benchmark cost.

</div>

<div class="ai">

## Three submitted settings and the scale-up test

</div>

<div class="ai">

At the end of either search track, the agent submits **exactly three complete training settings**. A setting includes immutable source/config revisions and the full scale-up recipe, including batch, optimizer, schedule and any permitted model changes. Freeze all three before the owner returns confirmation or scale-up results. Identify the search-time top-1 separately; it does not automatically become the scale-up winner. Settings cannot be revised or replaced after test feedback.

</div>

<div class="ai">

The owner trains **all three settings from scratch**, using the same predeclared scale-up seed list, training corpus and token target. Evaluate each at the token endpoint with the same frozen evaluator. For each setting, report all seed values and mean ± sample SD when N_test is at least two; report a single-seed score with seed SD unavailable when N_test is one. Select the **best of the three setting-level scores** on the predeclared final ranking metric. That setting is the agent's reported **top-1 scale-up result** in both tracks. Publish all three results and the winning setting ID, rather than selecting a different winner for each diagnostic metric. A tie uses a predeclared deterministic rule; incomplete tests remain visible and the submission is pending until all three finish.

</div>

<div class="ai">

### Equal training-token budgets

</div>

<div class="ai">

Scale-up tests are **iso-training-token** comparisons. The reference budget of **48 H100 GPU-hours** means approximately **12 hours on four H100s** for the reference recipe, not 48 hours on four GPUs. The existing reference target is **24,200,224,761 non-padding model tokens**, including BOS/EOS (about 24.20B). Freeze that target for every setting and seed in a comparison; a new calibrated target requires a new benchmark version.

</div>

<div class="ai">

Stop at the first completed optimizer update reaching the token target, report the actual token count and overrun, and require the completion receipt to confirm the token endpoint. Different batch sizes can produce different update counts and a small final-update overrun. A faster recipe may finish sooner and a slower one may take longer than 48 GPU-hours; wall time is recorded, not used to truncate a valid iso-token comparison. An early safety-walltime stop is incomplete and must resume to the endpoint before scoring. No setting receives extra tokens because unused time remains.

</div>

<div class="ai">

Evaluate **4,096 fixed MLM validation sequences** and **all 20,775 frozen contact chains**, with the same masking and contact-probe protocol for all settings. Score the final token-endpoint checkpoint; intermediate or best-search checkpoints may be retained for diagnostics but do not replace this endpoint. Contact-chain bootstrap 95% intervals and across-training-seed SD measure different variation and must be labeled separately. Evaluation assets currently overlap those available during research, so this measures transfer to a larger training budget rather than performance on a blind holdout.

</div>

<div class="ai">

Three scale-up settings cost nominally **3 × 48 = 144 H100 GPU-hours per scale-up seed**, or **144 × N_test** in total, before evaluation or shared reference runs. Actual GPU-hours depend on each recipe's throughput. Repeat the [manual token-budget training and evaluation procedure](EVALUATION.md#manual-test-of-progress) for every frozen setting and test seed, using the submitted recipe's permitted settings rather than silently replacing them with historical command defaults.

</div>

<div class="ai">

## Overall leaderboard

</div>

<div class="ai">

The overall leaderboard reports scale-up performance: the top-1 setting among each agent's three frozen submissions, tested at the same training-token budget and ranked by the task's predeclared final metric. Show both MLM validation loss and P@L for that same winning setting, together with seeds and compute. Search rewards are reported separately in the hardware tables below. The detailed historical recipe comparisons and longer-training references are in [sequential-search results](autoresearch-sequential-search.md#detailed-search-results).

</div>

<div class="ai">

| Entry | Role / search profile | Scale-up settings completed | Top-1 setting | Scale-up MLM loss ↓ | Scale-up P@L ↑ | Evidence |
|---|---|---:|---|---|---|---|
| human+ai baseline-09-26 | Current-default reference / L40S 1 hour; H100 20 minutes | Not run | — | — | — | [L40S](../.dev/reports/cck-human-ai-baseline-09-26-20260921/README.md) / [H100](../.dev/reports/fir-human-ai-baseline-09-26-20260921/README.md) search measurements |

</div>

<div class="ai">

No best-of-three scale-up submission under the new protocol is complete yet. The human+ai row identifies the reference recipe; its three-seed search measurement is one setting repeated three times. The completed L40S 100k-step study appears below as historical scale-up evidence, with its original protocol retained in the [sequential-search guide](autoresearch-sequential-search.md#l40s-100k-step-component-ablations).

</div>

<div class="ai">

### L40S 100k-step reference results

</div>

<div class="ai">

All three settings completed **100,000 updates, batch 1,024, seed 42, on four L40S GPUs**, consuming exactly **24,196,983,520 model tokens per run** without resampling. Scores use the final checkpoint, **4,096 MLM validation sequences** and **20,775 contact chains**. These single-seed results have no training-seed SD and use a slightly different token endpoint from the new 24,200,224,761-token benchmark. Training GPU-hours below exclude setup and evaluation.

</div>

<div class="ai">

| Entry / recipe | MLM validation loss ↓ | P@L ↑ | L40S training GPU-hours | Evidence |
|---|---:|---:|---:|---|
| **human+ai baseline-09-26 / current default**: separate Q/K/V; no query centering or RMS restoration | **2.410035** | **33.449770%** | 141.8907 | [Default selection](../.dev/reports/cck-contact-ablations-100k-20260919/DEFAULT_PROMOTION.md) |
| Full P@L search recipe: separate Q/K/V + query centering and RMS restoration | 2.413881 | 33.485122% | 150.8225 | [Full recipe receipt](../.dev/reports/cck-contact-ablations-100k-20260919/runs/full/VERIFIED.json) |
| Ablation: fused Q/K/V + query centering and RMS restoration | 2.416150 | 32.767462% | 149.0801 | [Ablation receipt](../.dev/reports/cck-contact-ablations-100k-20260919/runs/no-split-qkv/VERIFIED.json) |

</div>

<div class="ai">

The current default had the lowest observed validation loss; the full search recipe had the highest observed P@L. All three final checkpoints were also their runs' best observed P@L checkpoints. See the [detailed L40S comparison](autoresearch-sequential-search.md#l40s-100k-step-component-ablations) for component settings, contact-chain confidence intervals and the limits of this one-seed study.

</div>

<div class="ai">

## L40S search-reward leaderboard

</div>

<div class="ai">

Use one hour of training on four L40S GPUs per seed (4 L40S GPU-hours), with the same task, corpus and evaluation settings. Rank completed entries by confirmed mean MLM validation loss, with sample SD across training seeds; report P@L as a diagnostic with its own mean and sample SD. The current reference uses seeds **42, 43 and 44**, **32 fixed MLM validation sequences**, and **all 20,775 contact chains**. Its three-seed confirmation costs **12 L40S GPU-hours** of training, plus separately reported setup and evaluation. A baseline confirmation does not itself claim a 96-round search campaign.

</div>

<div class="ai">

| Entry | Recipe / role | Seeds | MLM validation loss ↓, mean ± SD | P@L ↑, mean ± SD | Training GPU-hours | Status / evidence |
|---|---|---:|---|---|---:|---|
| human+ai baseline-09-26 | Current default with separate Q/K/V Muon updates; reference | 42, 43, 44 | **2.589492 ± 0.009195** | **11.771370% ± 0.256263 pp** | 12.0017 | [Complete; per-seed results and receipts](../.dev/reports/cck-human-ai-baseline-09-26-20260921/README.md) |

</div>

<div class="ai">

The same default recipe's completed [100k-step reference result](#l40s-100k-step-reference-results) is listed above. Its validation score uses 4,096 sequences; this one-hour confirmation uses 32, so the loss values have different evaluation sample counts as well as different training budgets.

</div>

<div class="ai">

## H100 search-reward leaderboard

</div>

<div class="ai">

The fixed-round H100 profile uses **20 minutes on four H100 GPUs per seed**, or **4/3 H100 GPU-hours**, and confirms the best search candidate over the task's predeclared seeds. The current-default reference uses seeds 42, 43 and 44 with FA3, 32 MLM validation sequences and all 20,775 contact chains, for 4 H100 GPU training hours across the three seeds. Report validation loss and P@L with seed mean ± SD. Track 2 entries must identify their 144-H100-GPU-hour total search allowance and chosen proxy reward; different proxy definitions are not a common numerical ranking. Keep the track, per-run training duration, data and evaluation profile explicit. Measurements at other durations belong to separate profiles rather than the 20-minute ranking.

</div>

<div class="ai">

| Entry | Track / H100 profile | Seeds | MLM validation loss ↓, mean ± SD | P@L ↑, mean ± SD | Search training GPU-hours | Evidence |
|---|---|---:|---|---|---:|---|
| human+ai baseline-09-26 | Track 1 reference / 20 minutes × 4 H100, FA3 | 42, 43, 44 | Pending | Pending | 4 planned | [Fir array 60898535: queued](../.dev/reports/fir-human-ai-baseline-09-26-20260921/README.md) |
| No completed entries | Track 2 / 144 H100 GPU-hours total | — | — | — | — | — |

</div>

<div class="ai">

## Execution and records

</div>

<div class="ai">

The existing [validation-loss](../tasks/171m-validation-loss.md) and [contact P@L](../tasks/171m-p-at-l.md) scripts implement the earlier two-seed, one-hour/four-L40S measurement profile. They remain useful examples of the shared training/evaluation APIs, but running them unchanged does not implement either new track's full accounting and three-submission workflow. In particular, their two-seed candidate costs 8 L40S GPU-hours, and their fixed P@L search reward is an example objective rather than Track 1's required loss reward. Before launching a new benchmark, freeze its task parameters and configure the measurement and accounting commands to match this protocol.

</div>

<div class="ai">

Use [evaluation and token-budget commands](EVALUATION.md), [ordinary training commands](USAGE.md), and the [sequential-search guide](autoresearch-sequential-search.md) for implementation examples. Preserve original protocols alongside historical reports. Detailed sequential-search measurements and subsequent scale-up records are collected in [the sequential-search results guide](autoresearch-sequential-search.md#detailed-search-results).

</div>
