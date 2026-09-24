# Karpathy-style sequential AutoResearch

Our sequential-search method proposes one change, measures it, keeps or discards it and continues from the retained recipe. It comes in two programs that differ only in how they decide what to keep: [karpathy_ar_reward_gate.md](../autoresearch/karpathy_ar_reward_gate.md) applies a fixed two-seed reward rule, and [karpathy_ar_agent_gate.md](../autoresearch/karpathy_ar_agent_gate.md) leaves the decision to the agent's reasoning. Running both on the same task and budget compares the two acceptance policies. The [AutoResearch protocol](AUTORESEARCH.md) fixes the scientific boundaries and budgets. Two rounds of this method, together with human effort, produced [nanop-best-171m-round2](leaderboard/nanop-best-171m-round2.md).

## Running the example loop

Give any coding agent on your GPU compute node the [one-line launch prompt](../README.md#launch-autoresearch). It follows [setup_karpathy_ar.txt](../autoresearch/setup_karpathy_ar.txt) with the task and program named in the prompt and default paths for everything else, installs the loop skill, prepares the release and leaves the search agent in a tmux session with its prompt typed. Run `tmux attach -t nanoprotein-ar` and press Enter.

| Method setting | Reward gate | Agent gate |
|---|---|---|
| Seeds | 42 for every run; 43 when seed 42 beats the incumbent's mean | 42 by default; more seeds when the agent decides |
| Candidate cost | One round if screened out, otherwise two | One round per run the agent chooses |
| Acceptance | Two-seed mean gain larger than the larger of the two seed SDs | The agent's reasoning about the final evaluation, recorded with its evidence |
| Task measurement | 20 minutes on four H100 GPUs with FA3; 4/3 H100 GPU-hours per round | Same |
| Search evaluation | All 12,288 validation proteins and all 20,775 contact chains | Same |

Under the reward gate, each candidate takes one or two rounds, so 72 rounds cover 36–72 candidates when the baseline reuses the owner's four-H100 reference measurement, or 35–70 when it is measured. Under the agent gate, the count depends on how many rounds the agent spends on repeats and refinements. Each program lists the records to keep for every run.

## Two rounds under the previous search setting

Both rounds ran before the current search setting, so their numbers are reported in a different setting from the current configs and the [leaderboard](LEADERBOARD.md). Each candidate trained for one hour on four L40S GPUs per seed, with seeds 42 and 43, the seven-shard corpus, 554 warmup steps and 32 MLM validation proteins. Round 1 used global batch 256 and started from plain ESMC at LR 3.27e-4 and WD 0.0184; round 2 used global batch 1,024. Compare numbers only within a round.

### Previous two-seed pipeline

![Previous sequential AutoResearch loop: evaluate a baseline, propose one change, train and evaluate two seeds, keep or discard, record the result and repeat; the retained recipe then goes to a scale-up test.](figures/readme/autoresearch-loop.png)

Both rounds ran this loop. The agent measured the starting recipe, proposed one change, trained and evaluated it with seeds 42 and 43, kept or discarded it, recorded the result, and proposed the next change from the retained recipe. After search, a human-run 100k-step scale-up tested the retained changes. Round 1 launched each seed as a separate run; round 2's task command trained both seeds and wrote their mean, per-seed values and sample SD to `summary.json`.

| Pipeline setting | Rounds 1 and 2 |
|---|---|
| Training seeds per candidate | **42 and 43**, both trained from scratch |
| Training per seed | **1 hour on 4 L40S GPUs**, FlashAttention-2 |
| Schedule | **554 warmup steps**, then constant peak learning rate |
| Global batch | 256 in round 1; 1,024 in round 2 |
| Cost per candidate | **2 training runs = 8 L40S GPU-hours**, excluding setup and evaluation |
| Data | Seven-shard corpus |
| MLM evaluation | **32 fixed sequences**, context 512 |
| Contact evaluation | All **20,775 chains**, one probe per checkpoint, 5,000 chain-bootstrap replicates |
| Candidate score | Mean of the objective over the two seeds, with per-seed values and sample SD |

Round 1 kept a candidate when its mean validation-loss reduction exceeded that candidate's own two-seed sample SD. Round 2's program kept a candidate only when `candidate.ci95_low > incumbent.mean` and `candidate.mean > incumbent.ci95_high`. Each interval was `mean ± t(0.975, 1) × s / sqrt(2)` over the two seeds, with the reward oriented so that higher is better; ties were discarded.

Each campaign ran in its own worktree on a branch named `ar-YYMMDD-<name>`. It appended one row per trial to `results.tsv` and the hypothesis, evidence and decision to `research.log`, and saved every trial's diff, commands and outputs. The [current programs](#running-the-example-loop) keep this loop: the reward gate screens each candidate with one seed before spending a second, and the agent gate leaves the keep decision to the agent.

### Round 1: validation loss

Round 1 optimized MLM validation loss from the plain ESMC recipe. It evaluated a baseline and 38 candidate recipes, each with seeds 42 and 43: **78 runs, or 312 L40S GPU-hours**, excluding setup and evaluation. It kept a candidate when its mean validation-loss reduction exceeded that candidate's own two-seed sample SD.

![Round-1 validation-loss search across 38 rounds: orange trial means with sample-SD error bars and the retained recipe in blue.](figures/readme/validation-loss.png)

Numbers 1–5 mark the accepted changes: Muon, batch balance, sqrt loss, narrower FFNs and tied embeddings. R30–R38 were discarded, leaving R29 as the final retained recipe. Values are mean ± sample SD over seeds 42 and 43.

| Round-1 retained recipe | Search validation loss ↓ | Approximate parameters |
|---|---:|---:|
| AdamW baseline | 2.638680 ± 0.013025 | 171M |
| 1: Muon | 2.618073 ± 0.009455 | 171M |
| 2: + batch balance | 2.604151 ± 0.006502 | 171M |
| 3: + sqrt loss | 2.594372 ± 0.005778 | 171M |
| 4: + narrower FFN | 2.590952 ± 0.001323 | 142M |
| 5: + tied embeddings | 2.580568 ± 0.005442 | 142M |

Changes 4–5 use roughly 142M parameters and predate the current ±5% size bound. A human-run 100k-step H100 scale-up then kept the Muon package, batch balance and sqrt loss as [nanop-best-171m-round1](leaderboard/nanop-best-171m-round1.md). It skipped the FFN reduction, and tied embeddings regressed. The scale-up's Muon package also includes RMSNorm, residual routing and depth-scaled initialization, so its rows are not single-component ablations of the search rows; the [round-1 page](leaderboard/nanop-best-171m-round1.md#1-the-complete-comparison) gives the comparison.

#### Round-1 changes in brief

**1. Muon recipe.** Transformer matrices use Muon while embeddings and the MLM head retain AdamW, following the optimizer partition described in the [Muon implementation](https://github.com/KellerJordan/Muon). The tested package also changes transformer normalization to parameter-free RMSNorm, mixes the current hidden stream with the original embeddings at each layer, and scales residual-projection initialization with depth. [RMSNorm (Zhang and Sennrich, 2019)](https://arxiv.org/abs/1910.07467) motivates normalization by root mean square without mean subtraction. The experiment measures the combined package, so it cannot assign the gain to Muon alone. See [the full component and optimizer settings](leaderboard/nanop-best-171m-round1.md#2-exactly-what-differs-from-the-baseline).

**2. Batch balance.** Proteins have different lengths, so equal protein counts can leave GPUs with unequal token workloads. Redistributing already-masked examples balances non-padding token counts while preserving the sampled examples, labels and number of examples per rank. This can reduce time spent waiting at gradient synchronization; [PyTorch's DDP discussion of skewed processing speeds](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#skewed-processing-speeds) describes the underlying workload-balancing problem. Token count is a proxy for work, not an exact FLOP estimate. See [the partitioning procedure and loss normalization](leaderboard/nanop-best-171m-round1.md#3-batch-balance-equalize-work-across-gpus).

**3. Sqrt loss.** Weight each protein's mean masked-token loss by the square root of its number of masked targets, then normalize by the sum of those weights. Proteins with more targets contribute more than under equal-protein weighting, but less than under equal-target weighting. [BERT (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805) introduced the masked-language-model objective that this weighting modifies. Validation retains equal-protein weighting, so the evaluation metric stays fixed. See [the formula, worked example and distributed normalization](leaderboard/nanop-best-171m-round1.md#4-sqrt-loss-change-protein-weighting-not-the-validation-metric).

### Round 2: contact P@L

Round 2 optimized contact P@L, starting from nanop-best-171m-round1. The search audit records 38 audited candidates through trial 039 and two accepted additions: trial 011 added query centering with RMS restoration in the final eight layers, and trial 031 added separate Q/K/V Muon updates. Trial 040's training finished, but its audit and ledger entry were incomplete, so trial 031 remained the incumbent.

| Search result | Mean contact P@L | Recorded status |
|---|---:|---|
| nanop-best-171m-round1 (start) | 10.977610% | Baseline |
| Trial 011 | 11.447751% | Accepted |
| Trial 031 | 11.826297% | Accepted incumbent |
| Trial 040 | 11.760089% | Worker results only; not accepted in the audited ledger |

A 100k-step three-arm study then removed each addition from the trial-031 recipe. The owner kept separate Q/K/V updates without query centering, giving [nanop-best-171m-round2](leaderboard/nanop-best-171m-round2.md); that page reports the study.

**4. Separate Q/K/V Muon updates.** Muon orthogonalizes each weight-matrix update, so a fused QKV matrix is treated as one matrix. Round 2 gives Muon three views of that matrix, so the query, key and value updates are orthogonalized and scaled separately without changing the model or its parameters. See [the round-2 page](leaderboard/nanop-best-171m-round2.md#separate-qkv-muon-updates).
