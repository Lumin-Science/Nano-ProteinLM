# Program 2 research notebook

Kickoff: 2026-09-05. Root control files are in
`/scratch/muchenli/nano-protein-autoresearch-171m/.ar/`; all campaign work is in
`/scratch/muchenli/nano-protein-autoresearch-171m/.dev/worktrees/autoresearch-171m-val-loss`.
Branch: `autoresearch-171m-val-loss`; tmux session: `program2-171m`; Codex pane: `%0`.
The original program.md campaign is preserved. The base code is 9ec883b and the
baseline setup commit is c2db915. No training code has been modified.

## Fixed comparison

Seeds chosen before launch: 42 on kn081 (allocation 5218667), 43 on kn056
(allocation 5218668). Use these same seeds and node assignment for each method.
Each node has four idle NVIDIA L40S GPUs at kickoff. Current allocations expire
2026-09-07 06:24 and 07:55 America/Toronto respectively; check actual scheduler
state on continuation. The environment is independently copied then reconciled
with `uv sync --frozen`; only corpus and evaluation assets are shared read-only.
The prepared data symlink is under `.exps/program2/data/`; download and UV caches
are campaign-local. Run `bash .dev/program2/rounds.sh METHOD` only with both
configs ready, checks passed, and both nodes idle. Logs and node mappings are in
`.dev/program2/launches.tsv` and per-run `.launcher.log` files.

Baseline is original LayerNorm / AdamW ESMC-171M (170,671,168 parameters),
24 layers, width 768, 12 heads, RoPE 10,000, default residuals/initialization.
Seed copies change only the seed; original config and frozen runner remain intact.
Wait for BOTH baseline ROUND_STATE files to say complete, verify all receipts,
then populate results.tsv before making any candidate training-code change.
Selection: arithmetic validation-loss mean, sample SD (ddof=1), keep only when
old incumbent mean minus candidate mean is strictly greater than candidate SD.
Training loss is last 100 recorded MLM losses; full 20,775-chain P@L is diagnostic.
No results from program.md may serve as the Program 2 baseline or incumbent.

## Research reasoning at every round

Review each axis below against new measurements, select one conceptual change,
record a mechanism and falsifiable expected outcome, and reconsider after the
paired result. This is an adaptive list of hypotheses, not an automatic grid.
Do not spend consecutive rounds just making small LR/decay adjustments while
ignoring algorithm, model, training-data presentation, and loss alternatives.

- **Algorithm:** isolate hybrid Muon on the original architecture. Liu et al.,
  *Muon is Scalable for LLM Training* (2025), https://arxiv.org/abs/2502.16982,
  motivates orthogonalized hidden-matrix updates with scale matching and decay.
  Hypothesis: faster representation learning per unit compute can outweigh the
  optimizer overhead. The locked Torch implementation and hybrid parameter
  grouping already exist. Start with shared Muon scale and decay 1.0 plus
  match_rms_adamw, retaining AdamW for embeddings, output, and scalar parameters;
  do not bundle residual/norm/RoPE changes from the old campaign. Final choice
  remains conditional on baseline diagnostics.
- **Model:** test RMSNorm alone, or depth-scaled residual initialization alone.
  Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019),
  https://arxiv.org/abs/1910.07467, motivates cheaper normalization. The local
  parameter-free RMSNorm is a variant, not an exact paper reproduction. Check
  parameter count and paired wall-time throughput; keep residual architecture,
  optimizer, and loss fixed for this comparison.
- **Data presentation / systems:** examine token counts, padding and step-time
  variation, then consider token-budget batching or length-aware buffering of
  the existing source draws. Krell et al., *Efficient Sequence Packing without
  Cross-contamination* (2021/2022), https://arxiv.org/abs/2107.02027, motivates
  eliminating wasted padding with isolated sequences. This repository already
  packs tokens through the transformer, so do not assume that generic packing
  will bring a new gain. Preserve corpus, source mixture and marginal sequence
  sampling; never add, filter or reweight data. No cross-protein attention.
- **Loss:** consider a training-only objective estimating the same sequence-mean
  MLM risk with lower target-sampling variance, or a training-only denoising
  objective after checking the contract. Wettig et al., *Should You Mask 15% in
  Masked Language Modeling?* (EACL 2023), https://arxiv.org/abs/2202.08005,
  distinguishes supervision density from corruption difficulty. This is NLP
  evidence, not proof for proteins. Any experiment must leave the tokenizer,
  shared mask_tokens function, evaluator masks/seeds and evaluation reduction
  fixed; never alter shared evaluation utilities to implement a training loss.
- **Hyperparameters:** use gradient norms, throughput and variability to
  motivate changes only when evidence supports them; exactly 554 warmup steps
  and constant configured peaks are immutable.

Prior campaign results were inspected only to generate hypotheses: many small
LR changes did not improve its P@L criterion, and its architecture bundled Muon,
RMSNorm, routing and initialization. That is a reason to isolate mechanisms,
not evidence that any Program 2 candidate qualifies.

Additional protein-specific evidence: Cheng et al., *Training Compute-Optimal
Protein Language Models* (NeurIPS 2024), https://arxiv.org/abs/2411.02142,
Appendix C (https://arxiv.org/html/2411.02142v1) tests 85M/154M models and
reports useful masking rates around 10-20%, with degradation above 25% in
that setup. Therefore the NLP result does not justify jumping to 40% for this
171M protein model. The paper also studies CLM-to-MLM objective transfer;
a from-scratch staged objective within the SAME one-hour clock is a possible
later algorithm/loss hypothesis, without loading any pretrained weights or
changing the fixed evaluation. Neither finding guarantees improvement here.

## Baseline completed; round 1 decision (2026-09-05 07:46 UTC)

Both baseline repeats are complete and independently verified, including
checkpoint bytes and embedded configs, environment/data/lock receipts, all
logged LR values and final optimizer-group LRs, finite complete training
traces, fixed 32-sequence/994-target validation, and 20,775 distinct P@L chains.
Results: seed 42 validation 2.62947016954422, seed 43 2.6478907465934753;
mean 2.6386804580688477, sample SD 0.013025314944897757. P@L is respectively
0.10070865962196426 / 0.09225794583513981. Steps 10,599 / 10,546;
model tokens 640.861176M / 638.029608M; last-100 train loss 2.693044 / 2.699835.
Seed 43 has one allocator retry warning in stderr; the run recovered, logged
all expected steps with finite values, stopped on wall time, and completed
both evaluations. There is no training exception or skipped-step recovery in
the code. Record this observation; do not change the baseline environment.

Selected **r01_muon**: change only `optimizer: muon` in seed-matched original
configs. Existing hybrid implementation applies Muon to block matrices and
AdamW to embeddings/output/scalar parameters. Muon defaults: update RMS matched
to AdamW, 5 Newton-Schulz steps, momentum 0.95, Nesterov, shared LR/WD scales 1.
LR peak remains 0.000326599, 554-step linear warmup then constant on ALL groups.
Parameters remain 170,671,168. No model, source code, batching, corpus or loss
change is bundled. The hypothesis is that improved matrix conditioning lowers
fixed-clock held-out MLM loss despite fewer optimizer steps. A flat or worse
paired mean, or gain no larger than the candidate SD, falsifies acceptance.

Other axes considered this round: RMSNorm could increase throughput but alters
the model and will be a separate ablation; baseline gradients are finite and
moderate late in training, so residual initialization is less urgent. The
transformer already packs tokens, making a generic packing rewrite redundant;
data-presentation work needs measured batch-shape variability to justify it.
Loss ideas remain supervision-variance reduction or a staged objective, with
protein masking evidence tempering higher corruption rates. Small LR searches
are deferred until the optimizer mechanism is measured. Next round must review
all axes again in light of these two new runs, not automatically tune Muon LR.

Candidate snapshots: `.dev/program2/candidates/r01_muon/METHOD.json` plus both
YAML files. Configs are uncommitted pending acceptance. On continuation run
`uv run --frozen python .dev/program2/collect.py --method r01_muon --incumbent baseline --description 'Hybrid Muon alone on original ESMC-171M; shared LR/WD scales 1, RMS matching, NS=5, momentum=.95.'`
after both runs are complete, then commit as keep only on a passing decision;
otherwise remove only the two r01 config files after preserving the snapshots.
Baseline results.tsv is now populated; no need to re-evaluate the baseline.

## Round 1 retained; round 2 model hypothesis (2026-09-05)

R01 passed all receipts and full P@L on both seeds. Validation losses
2.624758470803499 / 2.6113871037960052; mean 2.6180727872997522,
sample SD 0.009454984284733061. Improvement over original baseline is
0.02060767076909542 > 0.009454984284733061, so **keep**, commit
03949ae1d927df901a5c4ea3033383406d0c3dfe. This is a 0.781% loss reduction;
both matched seeds improved. P@L 0.09661221556083431 / 0.0992818447672858,
mean 0.09794703016406006 (diagnostic only). Token throughput decreased about
8.99%; steps 9,635 / 9,609, model tokens 582.567767M / 581.293546M.
Late mean step time rose from ~0.339 s to ~0.373 s, while train loss fell.
Late gradient norms remain finite and moderate (~0.44 / 0.41). Seed 43 again
has a recovered allocator warning, with all expected finite train records;
no recipe/environment change is introduced for it. Seed 42's contact
execution took 431 s; allow about 9 minutes beyond launch+one-hour for the
next wakeup, instead of assuming the fastest evaluator timing.

**Selected r02_rmsnorm:** replace transformer LayerNorm with the already
implemented parameter-free RMSNorm on the retained hybrid-Muon recipe.
Expected parameter count drops from 170,671,168 to 170,559,808 solely because
normalization affine parameters disappear. Optimizer and its parameter
selection policy, LR/WD, loss, batching, source mixture, tokenizer, residual
routing/default initialization and RoPE remain as in R01. No training/model
source edits are required. This local F.rms_norm/default-epsilon/no-affine
variant is inspired by Zhang & Sennrich (2019), not an exact reproduction:
https://arxiv.org/abs/1910.07467. Hypothesis: cheaper normalization recovers
some fixed-clock throughput and improves held-out loss without destabilizing
representations. Throughput alone is insufficient; the SAME strict paired
loss rule must pass relative to R01's mean, not the original baseline.

Other axes reviewed: algorithmic orthogonalization already helped; further
optimizer microtuning is deferred to measure an independent model mechanism.
Data-side token-budget batching may reduce step-time variability but must
preserve each source's draw distribution; the packed transformer means
padding fraction alone does not prove wasted transformer FLOPs. Loss-side
variance reduction or staged CLM/MLM remain candidates; no evidence yet
justifies a high corruption jump in a 171M protein model. Residual/init
changes remain separate model hypotheses, since late gradients are stable.

R02 snapshots are `.dev/program2/candidates/r02_rmsnorm/`. Incumbent is
**r01_muon**, seeds 42/43, mean 2.6180727872997522. Once both runs complete,
verify the method/config/source snapshots and use the collector with
`--method r02_rmsnorm --incumbent r01_muon --description 'Parameter-free transformer RMSNorm alone on retained hybrid Muon.'`.
Commit only if it passes; otherwise preserve snapshots/results and remove
only the two R02 config files. Original baseline and R01 metrics already
exist in results.tsv; leave them intact.

## Round 2 discarded; round 3 loss hypothesis (2026-09-05)

R02 passed all completion/receipt checks but FAILED selection: validation
2.6085998192429543 / 2.624504843726754, mean 2.616552331484854,
sample SD 0.011246550667433; delta vs R01 = 0.001520455814898014,
which does not exceed SD. P@L 0.1036218200537681 / 0.0991815478881813;
steps 10,238 / 10,207, tokens 619.045755M / 617.466485M. The throughput
hypothesis worked (~6.2% more tokens), but a reliable validation improvement
did not. Results and candidate snapshots are preserved; only the two R02
untracked configs were removed. Incumbent remains R01, commit 03949ae.

**Selected r03_tokenmean**, an independent training-loss design change on
R01's LayerNorm/Muon model. Replace per-sequence-normalized MLM gradient with
the global masked-token mean gradient. Keep the exact same corpus draws,
source mixture, 15% mask-all corruption, tokenizer, sequence batch size,
model, optimizer settings, 554-step warmup and one-hour clock. This changes
the relative loss weight of short versus long sequences; it does not change
which sequences are drawn. It may harm the fixed equal-sequence validation
criterion; only the paired evaluation can settle that tradeoff.

Motivation: Rives et al. (PNAS 2021), Eq. 1 formulates masked-site NLL sums:
https://pmc.ncbi.nlm.nih.gov/articles/PMC8053943/. The primary fairseq MLM
criterion uses summed masked-token loss and target count as its gradient
denominator: https://github.com/facebookresearch/fairseq/blob/main/fairseq/criterions/masked_lm.py.
This is inspiration, not a claim of reproducing ESM's complete recipe.
A separate training-only sample (seed 20260905, 8x64 sequences) has masked
counts with quantiles [4,10,15,27.5,49,73,94] at [0,.1,.25,.5,.75,.9,1].
Sequences with <=10 targets form 10.74% of sequences but only 2.56% of targets.
Hypothesis: reducing their relative gradient weight improves shared residue
prediction within fixed compute. This is not an assumption of equivalence to
the incumbent's objective, and variance reduction is a hypothesis to test.

Implementation is ONLY in nano_protein/train.py plus a dedicated test file.
Per-token CE is computed once, a detached sum/count is pooled across DDP
ranks, and local differentiable sums use world_size/global_target_count to
cancel DDP averaging. `loss` remains the previous sequence-mean diagnostic;
`objective_loss` records the global token mean; evaluation is untouched.
The candidate uses gradient_accumulation=1. The default sequence_mean path
has exactly equal scalar values and gradients in tests. 17 training budget
and loss tests passed, including real two-rank CPU DDP matching pooled-target
gradients for unequal and empty-target ranks. Ruff check and format passed.

Other axes considered: the optimizer change already improved learning,
whereas a normalization speed gain did not meet the loss criterion; this
motivates investigating objective weighting before more optimizer tuning.
Data token-budget batching and model residual initialization remain plausible
but will be separate experiments. High masking ratios are deferred given
protein-specific evidence. No learning-rate search is bundled with R03.

Snapshots and exact candidate file list: `.dev/program2/candidates/r03_tokenmean/METHOD.json`.
Once both runs complete, verify source/config snapshots, finite `loss` AND
`objective_loss`, frozen receipts, and use collect.py with `--method r03_tokenmean
--incumbent r01_muon --description 'Global token-mean MLM objective on R01; sequence-mean diagnostics and evaluation fixed.'`.
On keep, commit train.py, tests/test_training_losses.py and both R03 configs.
On discard/crash, retain snapshots/results, restore ONLY nano_protein/train.py
from 03949ae and remove ONLY the new test and two R03 configs, after checking
they still match the candidate snapshot. All earlier accepted configs and
results must remain. Read existing PROGRAM2_VERIFICATION receipts for prior
methods rather than trying to re-collect removed candidate config paths.

## Round 3 discarded; round 4 data-presentation hypothesis (2026-09-05)

R03 completed both one-hour runs and all frozen evaluations. Sequence-mean
validation 2.6022557988762856 / 2.615473762154579; mean
2.6088647805154324, sample SD 0.00934651146755618; delta versus R01
0.009208006784319878 is STRICTLY LESS than SD, so **discard** despite being
close. P@L 0.11212684567856275 / 0.1070221831730501 is diagnostic and cannot
change selection. Steps 9,633 / 9,671, model tokens 582.446392M / 585.053621M;
last-100 sequence loss 2.654069519042969 / 2.650487675666809. The candidate
and tests are snapshotted; restored only train.py from 03949ae and removed
only its new test and two configs. Incumbent remains R01 mean 2.6180727872997522.
A token-mean loss could deserve a later predeclared larger matched-seed study,
but it is not accepted under the current two-seed result.

**Selected r04_batchbalance**, a data-presentation / training-implementation
experiment on R01. Form exactly the same original global batch and masks,
then redistribute COMPLETE corrupted sequences, labels and attention masks
across the four DDP ranks with exactly 64 sequences per rank. Deterministic
longest-sequence-first assignment balances valid-token counts and falls back
to the original assignment if its maximum token load would worsen. No data
is resampled, omitted, duplicated, split or concatenated. Source mixture,
mask RNG progression, global sample order before redistribution, optimizer,
model, loss and evaluation remain unchanged. Equal counts preserve global
sequence-mean gradient weighting; different reduction order can still cause
floating-point differences in actual training. Model remains 170,671,168
parameters, original LayerNorm/Muon, 554 warmup steps and constant peak LRs.

Inspiration: Li et al., *Hydraulis: Balancing Large Transformer Model Training
via Co-designing Parallel Strategies and Data Assignment*,
https://arxiv.org/abs/2412.07894 (v2, SIGMOD 2026), studies sampling/packing
imbalance in variable-length training. This is a much smaller, short-context
linear-token scheduling ablation, not a reproduction of that system or its
reported speedups. On 64 training-only sampled global batches (16,384
sequences; separate seed 20260905), average max-rank/mean token ratio falls
from 1.0757285841118562 to 1.0000466643779797; original p90 is 1.1313095657945897.
Mean CPU assignment time is 0.20694 ms. Actual all-gather, CPU synchronization,
copy and attention costs are included in the one-hour clock and may erase
the predicted gain. Hypothesis: fewer stragglers increase training steps
without changing the objective or the sequence distribution; only paired
validation loss determines acceptance.

Implementation: new nano_protein/batch_balance.py and a 9-line integration in
train.py, enabled only by balance_batches_across_ranks. Packing for transport
keeps example boundaries and moves the exact pre-existing corrupted tokens,
labels, and masks. The underlying data sampler, tokenizer, model, evaluator,
runner, lock and LR/stop guards are unchanged. Tests prove no duplicates or
missing examples, equal counts, deterministic assignment, nonincreasing
predicted maximum load, exact mask/label/RNG preservation and equality to
the global sequence-mean gradient using real CPU DDP. Four new tests and
12 existing budget tests passed. The locked Torch warns that the supported
all_gather_into_tensor alias is deprecated; it still works, and no dependency
change is made. Record batch_balance rank_tokens_before/after in train logs.

Other axes reviewed: algorithmic Muon remains retained; loss reweighting was
close but fails the specified threshold, so it is not carried into R04.
RMSNorm improved throughput but altered model behavior; this experiment seeks
throughput while preserving the objective and global batches. Residual/init
and alternative loss designs remain separate future hypotheses. Hyperparameter
microtuning is deferred until the data-assignment mechanism is measured.

R04 candidate files, checksums and exact snapshots are under
`.dev/program2/candidates/r04_batchbalance/`. On continuation verify all source
and config hashes, both complete runs and receipts, and batch-balance counts:
sum(before)==sum(after)==step_model_tokens, max(after)<=max(before). At common
recorded steps, model_tokens/filled_residues/sequences_seen/source_counts_rank0
should match the R01 run with the same seed. Use collect.py with
`--method r04_batchbalance --incumbent r01_muon --description 'Balance the same masked examples across DDP ranks; model and sequence-mean loss unchanged.'`.
On keep, commit the five candidate files from METHOD.json; on discard/crash,
restore ONLY train.py from 03949ae and remove ONLY batch_balance.py, its new
test, and the two R04 configs after verifying their snapshot hashes. Preserve
all notes, candidate snapshots, results and earlier accepted config files.

## Round 4 kept; round 5 model-initialization hypothesis (2026-09-05)

R04 completed and verified both one-hour runs, fixed validation and all 20,775
contact chains per run. Validation loss 2.5995532013475895 / 2.608747873455286;
mean 2.6041505374014378, sample SD 0.006501614998139024; delta versus R01
0.013922249898314476 > SD, so **keep**. Accepted commit
`0a2b17e6899da79a87d2592bedf092ade9a3ce2a` is the new incumbent. R04 run rows
retain commit=NA because the candidate was uncommitted during execution.
P@L 0.09560949812257497 / 0.09178944972741465 declined but remains diagnostic;
selection follows the predefined validation-loss rule. Last-100 training loss
2.659829168319702 / 2.6589782524108885. Steps 10,396 / 10,339; model tokens
628.605445M / 625.485368M, gains 7.9025% / 7.6023% over matched R01 seeds.
This agrees with the straggler-reduction hypothesis, though reordering also
changes floating-point reduction order. All 1,040 / 1,034 logged batches
preserve token sums and never worsen predicted maximum rank load. All 964 /
961 logged steps shared with R01 have exactly matching cumulative token,
residue, sequence and source counters. Source/config snapshots and all
checkpoint/evaluation receipts passed. See BALANCE_VERIFICATION.json and
DECISION.json in candidates/r04_batchbalance plus per-run receipts.

**Selected r05_depthinit**, a single model-initialization experiment on R04.
Enable the existing `depth_scaled_residual_init: true` option. Initialize
attention output and FFN down-projection weights with standard deviation
0.02/sqrt(2*24) = 0.002886751345948129 instead of 0.02. These are the 48
residual-producing projections in the 24-layer model. No trainable parameters
are added; total remains 170,671,168. The runtime ESMC residual divisor
sqrt(24/36), every other initialized tensor, LayerNorm, Muon, sequence-mean
MLM objective, batch balancing, samples and frozen evaluation contract are
retained. Only the two candidate YAML files change; implementation already
exists in model.py. The extra initialization draws use the CPU RNG. Training
sampling has its own NumPy generator and masking uses CUDA RNG; still check
the data-prefix counters against R04 at common steps after launch.

Inspiration: Radford et al., *Language Models are Unsupervised Multitask
Learners* (2019), section 2.3, scales residual-layer initialization by
1/sqrt(N) to account for accumulation with depth:
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf.
Here N=48 residual sublayers. This is a GPT-2-inspired initialization ablation
in an ESMC encoder, not a claim to reproduce GPT-2 or its complete recipe.
Hypothesis: smaller initial residual contributions improve early gradient
propagation and learning efficiency during the fixed 554-step warmup and
one-hour budget, without a steady-state throughput cost. The incumbent is
already stable; avoiding divergence is not the claimed benefit. Muon's
normalized updates may quickly overwhelm the smaller initial projections,
and excessive suppression may instead slow useful feature learning. The
paired fixed validation loss, not initial gradient magnitude, decides.

All research axes reviewed: algorithmic work could examine Muon's behavior
for the paired SwiGLU/QKV matrices, but this would need a separate conceptual
ablation; no optimizer learning rates are tuned with R05. Data assignment
now nearly equalizes token loads, so further scheduling work needs measured
attention/communication costs before increasing complexity. Global token-mean
loss remains unaccepted after R03 narrowly failed; a larger predeclared
matched-seed study or a distinct variance-reduction loss could be considered
later. Higher masking is not presumed better: the protein-specific scaling
study cited above favored roughly 10-20% at nearby sizes. Model normalization
R02 failed the acceptance threshold despite speed gains; residual initialization
is a different model mechanism and does not carry RMSNorm into the incumbent.

Before launch, verify full-size from-scratch parameter count and that exactly
48 projection tensors differ while every other initialized tensor is identical
under a separate diagnostic seed; store INITIALIZATION_VERIFICATION.json.
Run existing model configuration and training budget checks plus required
Ruff check and format checks. Method/source/config snapshots live under
`.dev/program2/candidates/r05_depthinit/`. Collect completed repeats with
`--method r05_depthinit --incumbent r04_batchbalance --description 'Depth-scaled residual projection initialization on accepted Muon plus balanced batches.'`.
Apply strict delta > candidate sample SD before updating the incumbent. On
keep, commit ONLY the two R05 configs listed in METHOD.json. On discard/crash,
verify their snapshot hashes then remove ONLY those two configs; accepted
R04 source, tests and configs must remain. Preserve all experiment artifacts.

R05 launched concurrently at 2026-09-05T12:51:32Z: seed 42 on kn081 and seed
43 on kn056, run suffix 20260905T125132Z. Startup verified through step 230
on both: finite losses 2.7891135215759277 / 2.781238555908203, all eight L40S
GPUs active, correct warmup LR and matching frozen source/config receipts.
The first 24 logged records per seed exactly match R04's cumulative data
counters AND before/after rank assignments, with preserved global token
counts. Both runs remain under the one-hour synchronized guard. Full-size
initialization verification and 15 existing tests passed; required Ruff
checks passed locally and in both node wrappers. Next collection compares
against R04 mean 2.6041505374014378. On completion repeat the full source,
config, batch and data-prefix checks before using collect.py.

## Round 5 discarded; round 6 optimizer-geometry hypothesis (2026-09-05)

R05 completed both one-hour runs and every frozen evaluation. Validation loss
2.6007565408945084 / 2.631933022290468; mean 2.6163447815924883, sample SD
0.022045101408619455; delta versus R04 -0.01219424419105053, so **discard**.
P@L 0.10223602233487235 / 0.10107968960834805 remains diagnostic. Last-100
training loss 2.6592971086502075 / 2.658982651233673; steps 10,379 / 10,351;
model tokens 627.567674M / 626.214791M. Throughput is effectively unchanged
and the smaller initialization did not provide a reliable final-loss gain.
All 1,038 / 1,036 logged balance checks pass; 1,038 / 1,034 common recorded
steps match R04's exact data counters AND before/after rank assignments.
All launch source/config hashes and checkpoint/evaluation receipts verified.
Removed only the two R05 configs after checking their saved hashes; source
remains the accepted R04 implementation at 0a2b17e. Results and snapshots are
retained. Incumbent remains R04 mean 2.6041505374014378 on seeds 42/43.

**Selected r06_splitqkv**, a single optimizer design change on R04. Apply Muon
orthogonalization independently to the Q, K and V row blocks in each fused
attention.qkv.weight. Each of 24 matrices has shape [2304,768]; the optimizer
views it as three [768,768] blocks. All forward/backward model operations,
initial tensors, parameter counts, loss, sampling, masks, batch balancing,
FFN optimization, AdamW fallback, LR peaks, weight decay, momentum and five
Newton-Schulz iterations remain unchanged. No head-wise splits, FFN gate/value
splits, normalization or depth-scaled initialization are included.

Primary inspiration: Keller Jordan, *Muon: An optimizer for hidden layers in
neural networks*, explicitly reports an empirical preference for treating
Q/K/V separately: https://kellerjordan.github.io/posts/muon/. Boreiko, Bu and
Zha, *Towards Understanding Orthogonalization in Muon*, ICML 2025 ES-FoMo III
workshop, studies independent block-wise orthogonalization:
https://icml.cc/virtual/2025/51841. NVIDIA Megatron-LM's own implementation
also exposes split_qkv and separate QKV shapes:
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/emerging_optimizers.py.
These sources motivate an ablation; neither their results nor their complete
optimizer implementations are transplanted as evidence of a protein gain.

Hypothesis: a fused polar update couples distinct query/key/value gradients;
independent blocks can improve learning efficiency by letting each projection
receive its own orthogonalized direction. The existing match_rms_adamw rule
is evaluated on EACH block shape, exactly as for separate native parameters;
that necessary shape-dependent scaling is part of the predeclared algorithm,
not an extra LR search. Decoupled weight decay uses the original group LR and
is applied once to each element. Square-block Newton-Schulz and extra kernel
launches may cost more than the fused tall-matrix update; all overhead counts
against the same one-hour clock. Only paired frozen validation decides.

Implementation: SplitQKVMuon subclasses the locked torch.optim.Muon and expands
its temporary update lists into parameter/gradient/momentum views. Original
parameter groups, full fused momentum state, DDP gradient synchronization,
clipping, zero_grad and checkpoints retain their ordinary ownership. There
are no proxy trainable parameters or duplicate tensor storage. The integration
is confined to build_optimizer and enabled by muon_split_qkv: true; default
native Muon remains unchanged. The native preparation hook is private, so
compatibility is explicitly tested against the frozen Torch dependency.

Tests: five new optimizer tests cover exact multistep equivalence to independent
native Q/K/V optimizers (including unequal gradient scales, weight decay,
changing group LR and momentum), untouched non-QKV updates, checkpoint resume,
missing gradients, both gradient-clearing modes, invalid selection, default
optimizer identity, complete nonduplicated parameter coverage, and real
2-rank DDP pooled gradients plus exact independent updates. These and the
15 existing model/budget tests pass: 20 total. Ruff checks pass after fixing
only test import order. No evaluator, runner, dependency or model source edits.

Other axes considered: data balancing already buys about 7.8% more tokens;
further data work needs measured costs beyond token stragglers. Model RMSNorm
and depth initialization have both failed the threshold, motivating an optimizer
geometry change before another model tweak. Loss token-mean remains unaccepted;
a future larger predeclared paired study or per-sequence target-count variance
reduction is distinct work. Training masking changes deserve protein-specific
support rather than blindly copying high NLP masking rates. Corpus and source
mixture remain fixed in every case. No scalar hyperparameter tuning is bundled.

Snapshots and exact candidate list are in candidates/r06_splitqkv/METHOD.json.
On continuation, verify source/config snapshots, both complete run receipts,
finite logged gradients/losses, balanced-token invariants and exact common-step
data prefixes versus R04. Check final optimizer state still contains the
original 96 hidden matrices with full [2304,768] QKV momentum buffers and
constant configured LRs after warmup. Collect with
`--method r06_splitqkv --incumbent r04_batchbalance --description 'Independent Q/K/V Muon orthogonalization with fused model and optimizer storage on R04.'`.
Enforce delta > candidate sample SD. On keep, commit only train.py, new
split_muon.py, new test_split_muon.py and the two candidate YAMLs. On discard
or crash, verify snapshot hashes, restore ONLY train.py from 0a2b17e, remove
ONLY those four new files, and retain the accepted batch balancer, its tests,
all prior configs and all experiment artifacts.

R06 launched both seeds at 2026-09-05T14:09:49Z, suffix 20260905T140949Z.
Startup verified through step 350 on each node: finite losses
2.796502113342285 / 2.8137946128845215 and finite gradients. All eight reserved
L40S GPUs are active. Initial loss matches R04, and all 36 logged records per
seed have exactly matching data counters and rank assignments versus R04,
conserved global tokens and the required warmup LR. Source/config snapshots
match launch receipts. Next checkpoint is 2026-09-05T15:17:49Z, allowing the
one-hour run plus observed evaluation overhead. Collect both complete repeats
against R04 before making any selection decision.

## Round 6 discarded; round 7 training-mask distribution hypothesis (2026-09-05)

R06 completed both one-hour runs, fixed validation, and all 20,775 contact
chains per run. Validation 2.6185063757002354 / 2.6265226490795612; mean
2.6225145123898983, sample SD 0.005668361266366521; delta versus R04
-0.01836397498846054, so **discard**. Steps 9,821 / 9,793; tokens 593.827518M /
592.414059M (about 5% fewer than R04). Last-100 training loss
2.658048846721649 / 2.6617181134223937; P@L 0.0961745802822237 /
0.08376809528865857. The QKV split did not repay its overhead in this setting.
All 983 / 980 logged records match incumbent data prefixes and rank loads.
All 96 native fused momentum buffers, including 24 full QKV matrices, are
finite and have their original shapes. Source/config snapshots, checkpoint
hashes and evaluation receipts verified. Restored only train.py from 0a2b17e
and removed only the four new R06 files after verifying their hashes. All
snapshots and results remain. Incumbent is still R04 mean 2.6041505374014378.

**Selected r07_maskcount**, a single training-corruption distribution change
on R04, retaining the sequence-mean MLM loss. For n eligible canonical residues
and p=0.15, the incumbent uses K=max(1,Binomial(n,p)) for n>0; E[K] is therefore
mu=n*p+(1-p)^n. When n=0 there are no targets. Draw the new count as floor(mu)
plus Bernoulli(mu-floor(mu)), then choose that many eligible positions uniformly
without replacement. Every chosen target is replaced by MASK as before.
This preserves the expected count AND uniform per-position masking marginal
(including the forced-one correction) while minimizing integer count variance
at that mean. Floating-point computation introduces negligible approximation;
compare random draws to the fractional part instead of adding them to mu,
which avoids erroneous extra targets near integer FP32 expectations.

This changes the JOINT mask distribution and thus the training objective in
context; it is not an unbiased estimator of the old full MLM objective.
Lower count variance does not by itself prove lower gradient variance or
better held-out loss. Hypothesis: avoiding unusually sparse/dense target sets
makes the sequence-mean learning signal more consistent, especially on short
proteins, without changing their relative sequence weights. The earlier global
token-mean objective was close but failed the rule; it is not carried into R07.
The evaluator's original Bernoulli masks, 15% rate, seed and 32 fixed sequences
are untouched. Training sequences, crops, corpus mixture and optimizer/model
remain R04; masks and CUDA RNG progression intentionally differ. The NumPy
sampler is independent, so data prefixes and balanced rank assignments should
still match R04 at common steps. Initial losses need not match because masks
are different even though model initialization is unchanged.

Primary inspiration: Chen, *Variance-reduced Language Pretraining via a Mask
Proposal Network* (2020), https://arxiv.org/abs/2008.05333, separates data and
mask-sampling contributions to gradient variance. Zheng et al., *Improving
Self-supervised Pre-training via a Fully-Explored Masked Language Model*
(2020), https://arxiv.org/abs/2010.06040, studies structured mask sampling to
reduce variance. This experiment implements neither MAPNet nor their full
exploration strategy. Fairseq's own MaskTokensDataset stochastically rounds
the desired mask count and samples without replacement:
https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/mask_tokens_dataset.py.
R07 adapts that count-sampling idea to preserve the local forced-one expectation
and ESMC's canonical-only mask-all corruption; no learned proposal network,
span mask, replacement policy or additional model is included.

Training-only diagnostic: 2,048 sequences at separate seed 20260905, mean
236.3833 eligible residues, expected 35.4575 targets/sequence. Average analytic
conditional count variance falls from 30.138823547631837 to
0.1744063203646017. Empirical count means/variances and CPU timing of the final
implementation are saved in TRAIN_ONLY_ANALYSIS.json. This check establishes
the mask-count mechanism, not a claim about gradient variance or GPU throughput.
Sorting priorities adds work, which is included in the one-hour budget.

Implementation: new nano_protein/training_masks.py, a training-only function
selection in train.py, and rank-0 last-microbatch mask-count diagnostics. The
frozen tokenizer.py, model, evaluators, dependencies and runner are unchanged.
Diagnostics reconstruct eligible/selected counts from corrupted tokens and
labels AFTER the accepted batch redistribution, keeping row identity correct.
Six new tests cover exact small-n Bernoulli expectations, eligibility and
no-input-mutation, forced-one/empty cases, mask-all labels, reproducibility,
external RNG isolation, statistical uniform marginals and count variance over
32,768 masks, an FP32 rounding boundary, and diagnostics after permutation.
These plus 15 existing model/budget tests are required before launch.

Other axes considered: algorithmic QKV splitting was slower without validation
gain; keep original fused Muon. Model RMSNorm and depth initialization remain
discarded. Data rank balancing remains accepted; corpus filtering or source
mixture changes are prohibited, and additional packing must justify measured
costs. Loss design now focuses on mask-sampling variance while preserving the
specified sequence reduction. Higher masking ratios are not presumed better;
the prior protein-specific study supports staying near 10-20% at this scale.
A future predeclared token-mean replication or model residual-routing experiment
would be separate work. No scalar LR or mask-rate tuning is bundled here.

On continuation verify candidates/r07_maskcount/METHOD.json source/config hashes,
both completed runs and frozen receipts, finite losses/gradients, and unchanged
common-step data counters and rank assignments versus R04. For every logged
mask_counts row, selected must be floor(mu) or ceil(mu), zero iff n=0 and
between 1 and n otherwise. Check the actual FP32 expectation used by the helper;
do not require masks or initial losses to match the incumbent. Collect with
`--method r07_maskcount --incumbent r04_batchbalance --description 'Stochastically rounded target counts with preserved expected masking and sequence-mean loss on R04.'`.
Apply strict delta > candidate sample SD. On keep, commit only train.py,
training_masks.py, test_training_masks.py and the two R07 configs. On discard
or crash, verify snapshot hashes, restore ONLY train.py from 0a2b17e and remove
ONLY those four new files; retain accepted batch balancing and all artifacts.

Final R07 checks: 21 tests passed and required Ruff check/format passed before
launch (also in both node wrappers). Final training-only empirical target
means: incumbent 35.47607421875, candidate 35.4541015625; expected 35.457497675.
Conditional squared count deviations 30.7644375230 versus 0.1678552794. The
final sampler's CPU diagnostic time was 2.62854 ms per 64 sequences; GPU
runtime remains measured by the actual run.

R07 launched concurrently at 2026-09-05T15:30:16Z, suffix 20260905T153016Z.
Startup verified at steps 340 / 330: finite losses 2.7894744873046875 /
2.794991970062256 and gradients, correct warmup LR, and all eight L40S GPUs
active. All 35 / 34 logged data-prefix and rank-assignment checks match R04.
All 2,240 / 2,176 rank-0 example mask counts meet the new floor/ceiling count
rule and eligibility/empty-row bounds. Launch source/config snapshots match.
Next check: 2026-09-05T16:38:16Z. Collect against R04 only after both completed
runs and full frozen evaluations are verified.

## Round 7 discarded; round 8 learned residual-route hypothesis (2026-09-05)

R07 completed both one-hour runs and every frozen evaluation. Validation
2.6133114770054817 / 2.6187297999858856; mean 2.6160206384956837, sample SD
0.0038313329221025027; delta versus R04 -0.01187010109424591, so **discard**.
Steps 10,384 / 10,332 and tokens 627.879742M / 625.058478M are close to R04;
last-100 training loss 2.657855603694916 / 2.657810926437378. P@L
0.10214500196311922 / 0.09431551482731262 is diagnostic. Reduced mask-count
variance did not improve the fixed validation criterion. All 1,039 / 1,034
logged data-prefix and rank-assignment checks passed, as did the count rules
on all 66,496 / 66,176 logged rank-0 examples. Snapshot, checkpoint and all
evaluation receipts verified. Restored only train.py from 0a2b17e and removed
only the four new R07 candidate files. Results and snapshots remain. Incumbent
remains R04 mean 2.6041505374014378 on seeds 42/43.

**Selected r08_residual**, one model-design change on R04: enable the existing
learned_residual_routing option. Before each transformer block, compute
h <- a_l * h + b_l * h0, where h0 is the original embedding of the corrupted
input sequence. There is no access to clean masked residues through h0.
The 24 a_l scalars start linearly from 1.15 to 1.05, and the 24 b_l scalars
from 0.20 to 0.05, following the repository's existing implementation. These
48 additional parameters bring the actual total to 170,671,216, within the
171,000,000 limit. The existing parameter grouping puts both vectors into
AdamW's no-decay group at the same fixed configured LR; no special scalar LR
is introduced. Muon matrices, original LayerNorm, default residual projection
initialization, Bernoulli mask-all training corruption, sequence-mean loss,
data sampler and accepted rank balancing remain unchanged.

Primary inspiration: Pagliardini, Mohtashami, Fleuret and Jaggi, *DenseFormer:
Enhancing Information Flow in Transformers via Depth Weighted Averaging*
(NeurIPS 2024), https://arxiv.org/abs/2402.02622, learns mixtures of representations
across depth. This is NOT DenseFormer's all-history post-block weighted average;
it is a restricted pre-block mixture of the current stream and initial input.
Karpathy's nanochat source uses the same two-route expression and descending
initial scalar values:
https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py.
Its other architecture choices and specialized scalar learning rates are not
part of this ablation. The actual local implementation and frozen candidate
configs, not a moving upstream repository, define the experiment.

Hypothesis: direct learned routes to the initial residue representation can
preserve useful local information and let each layer control its contribution
within a short training budget. Unlike R05, this changes learnable connectivity
throughout training, not only initial projection magnitudes. It may instead
amplify early representations, underuse deep transformations, or cost enough
extra memory traffic and gradient reductions to hurt fixed-time progress.
Neither improved convergence nor throughput is assumed; the paired frozen
validation loss is decisive. All routing work is within the one-hour clock.

Only two configs are candidate edits: the model flag and its consequent
expected_parameter_count adjustment. Source code is already implemented and
unchanged from the accepted commit. Before launch, run existing model and
training-budget tests plus required Ruff checks, and perform a separate
full-size initialization/parameter-coverage check. All pre-existing initialized
tensors should be byte-identical under an independent diagnostic seed; only
the two new scalar vectors should be added. A small synthetic CPU forward,
backward and optimizer step checks that both routes receive finite nonzero
gradients and update. No held-out data is used for this mechanism check.

Other axes reviewed: algorithmic split-QKV Muon and mask-count variance changes
failed the fixed-time criterion, so neither is retained. Data rank balancing
remains the accepted throughput gain; further packing work needs profiling
and must preserve corpus/mixture. Token-mean loss failed the strict threshold
and can only return as a separately declared study. Model RMSNorm and depth
initialization were not wins, but this learned input-reuse mechanism is
separate from both. No LR, masking-rate or normalization sweep is bundled.

Snapshots live in candidates/r08_residual/METHOD.json and ROUTING_VERIFICATION.json.
On continuation, verify source/config snapshots and all frozen run receipts,
finite training metrics and exact data-prefix/rank-assignment counters versus
R04 at common steps. Check final parameter count 170671216, model_config flag,
finite shape-[24] residual_lambdas/input_lambdas, and both vectors changed from
the saved initialization. The scalar AdamW groups must follow exactly 554
warmup steps then the fixed peak LR. Initial loss need not match the incumbent
because routing changes the forward computation. Collect with
`--method r08_residual --incumbent r04_batchbalance --description 'Learned per-layer running-stream and initial-embedding routes on accepted R04.'`.
Apply strict delta > candidate SD. On keep, commit ONLY the two R08 configs;
on discard/crash, verify their snapshot hashes then remove ONLY those two
configs. All accepted source, earlier configs and experiment artifacts stay.

R08 prelaunch verification passed: 15 existing tests and required Ruff checks;
actual full-size parameter count 170,671,216; all original initialized tensors
byte-identical; complete optimizer parameter coverage; new route vectors in
the existing no-decay AdamW group. Both vectors received finite nonzero
synthetic gradients and changed after an optimizer step. See
ROUTING_VERIFICATION.json for the numerical gradients and initial vectors.

R08 launched concurrently at 2026-09-05T16:46:03Z, suffix 20260905T164603Z.
Startup verified at steps 300 / 290 with finite losses 2.827677011489868 /
2.801583766937256, finite gradients and correct warmup LRs. All 31 / 30 logged
data-prefix and rank-assignment checks match R04. All eight L40S GPUs are
active, with observed reserved memory about 28-32 GiB per GPU. Source/config
snapshots match launch receipts. Next check is 2026-09-05T17:55:03Z, allowing
69 minutes from launch because the latest full contact evaluations took up
to 459 seconds in addition to training/setup/checkpoint time. Verify both
complete evaluations and learned route vectors before collection against R04.

## Round 8 discarded; round 9 context curriculum (2026-09-05)

R08 completed both full runs and frozen evaluations. Validation
2.6074235662817955 / 2.6119563337415457; mean 2.6096899500116706, sample SD
0.0032051506083310703; delta versus R04 -0.00553941261023283, so **discard**.
Steps 9,545 / 9,444; tokens 577.136794M / 571.360267M, roughly 8-9% below R04.
Last-100 training loss 2.667121865749359 / 2.6667118859291077; P@L
0.10070740259556216 / 0.09994096819513408. Both learned scalar vectors are
finite, correctly shaped and changed from initialization. All 955 / 945
logged data-prefix/balance checks and source/checkpoint/evaluation receipts
passed. Removed only the two R08 configs after verifying their snapshots.
Incumbent remains R04 at 0a2b17e, mean 2.6041505374014378 on seeds 42/43.

**Selected r09_context**, one data-presentation curriculum on R04: context
128 for the first 900 synchronized training seconds, then context 512 until
the same 3600-second stop. Microbatch 64, accumulation 1 and the exact frozen
source mixture are identical in both stages. The same sampler objects and
RNG/row-sampler state continue through the transition. Model, parameter count
170,671,168, original fused Muon, LayerNorm, Bernoulli mask-all corruption and
sequence-mean loss remain accepted R04. Learning rates warm for exactly 554
GLOBAL optimizer steps and stay at their configured peaks across the context
transition; there is no LR reset or decay. The existing stage checkpoint and
synchronization cost is inside the one-hour wall clock. Final evaluation is
still the fixed context-512 protocol and full 20,775-chain contact diagnostic.

Primary inspiration: Press, Smith and Lewis, *Shortformer: Better Language
Modeling using Shorter Inputs* (ACL 2021),
https://aclanthology.org/2021.acl-long.427/, studies training initially on short
subsequences before longer ones. Li, Zhang and He, *The Stability-Efficiency
Dilemma: Investigating Sequence Length Warmup for Training GPT Models*
(NeurIPS 2022), https://arxiv.org/abs/2108.06084, investigates length curriculum
and gradient stability. This two-stage protein-MLM experiment imports neither
Shortformer's positional/recurrence scheme nor larger GPT batch sizes/LRs.
The incumbent is already stable; the hypothesis is earlier learning of local
residue patterns with more sequence updates per minute, then enough long-context
training to recover dependencies needed by the frozen evaluator. Shorter crops
also reduce context and target counts and increase optimizer overhead per
token, so token throughput or final loss may worsen. No speedup is assumed.

Implementation: build_stage_batchers in a new training_streams.py can share one
MixtureBatcher between stages only when mixtures match exactly. The default
independent-stage behavior is preserved. train.py selects this via
share_stage_data_stream, and logs actual context_length. This prevents the
existing two-stage constructor from restarting the same seeded corpus stream
at the transition. stage1_fraction=0.25 and the two context lengths predeclare
this curriculum. No model/evaluator/tokenizer/runner/lock modifications.

Unlike prior same-context ablations, a crop-length change alters crop RNG draws
and can change subsequent sampled source/sequence choices; equality to R04's
exact data prefix is NOT expected. Corpus membership and mixture probabilities
remain frozen. Validate against a deterministic replay of one persistent
MixtureBatcher per rank using this run's actual stage boundary, not R04's
counters. Stage checkpoint optimizer_step identifies the exact last short
step. The run still uses the same predeclared training seeds and node mapping.

All 19 tests passed (4 stream/schedule tests plus 15 existing model/budget tests),
and required Ruff check/format passed. Real training-only verification with a
separate seed 20260905 compared 3 short then 3 long batches on every rank
(1,536 examples) to a single persistent reference sampler: tokens, attention
masks, crops and accumulated source counts match exactly across the boundary.
See candidates/r09_context/STREAM_VERIFICATION.json. No held-out data used.

Other axes reviewed: Muon matrix geometry and learned residual routes lost
throughput without meeting the loss rule; keep the original accepted optimizer
and model. Rounded target-count masking did not help despite reducing count
variance; loss reweighting remains unaccepted. Sequence curriculum changes
available context and early data presentation instead of layering more model
operations onto every step. Source reweighting/filtering is outside this
campaign; no corpus or mixture change is made. No LR or batch-size tuning is
bundled. A later efficiency implementation or larger predeclared loss study
would be separate work.

Snapshots and exact candidate files: candidates/r09_context/METHOD.json. First
wakeup should be about 16 minutes after launch, to verify BOTH runs reached
stage2/context512 and continued healthy with fixed peak LR and continuous data
counts. After that, if healthy, schedule the completion checkpoint at about
launch+69 minutes. Do not collect a partial run or launch another experiment
while this pair is active. At completion verify source/config/checkpoint/lock/
data/evaluation receipts, stage boundary, no LR reset, balanced rank loads and
deterministic data replay using .dev/program2/verify_context.py.
Collect with `--method r09_context --incumbent r04_batchbalance --description 'Continuous-stream context curriculum: 900 seconds at 128, then 512, with fixed LR schedule on R04.'`.
Enforce delta > candidate sample SD. On keep, commit only train.py, new
training_streams.py, new test_training_streams.py and the two R09 configs.
On discard/crash, verify snapshot hashes, restore ONLY train.py from 0a2b17e,
remove ONLY those four new files, and retain all accepted code and artifacts.

R09 launched concurrently at 2026-09-05T18:02:49Z, suffix 20260905T180249Z.
Startup verified at steps 450 / 460 with finite losses 2.7924554347991943 /
2.8239386081695557 at context128, correct global LR warmup and all eight GPUs
active. Deterministic replay of ALL sampled batches through those steps across
all four ranks matches every one of the 46 / 47 logged token, residue, sequence,
source-count and before/after rank-load records exactly. See
candidates/r09_context/STARTUP_VERIFICATION.json. The reusable verification
command is `uv run --frozen python .dev/program2/verify_context.py`.

Next wake is 2026-09-05T18:18:49Z (launch+16 minutes). Run the verifier with
`--require-stage2` to confirm both transitions and continued sampler state;
it loads checkpoint-stage1.pt metadata for the exact last short-context step,
then reconstructs one persistent data stream per rank. It does not replay or
consume held-out data. Check both GPUs/runs remain healthy and fixed peak LRs
continue; then schedule the full completion check at 2026-09-05T19:11:49Z
(launch+69 minutes). At completion use `--require-complete` before the standard
collect.py call and strict paired keep/discard rule. The current two runs are
one experiment; do not launch a replacement during the transition check.

## R09 transition verified (2026-09-05T18:20Z)

Both runs reached context512 without restarting the data stream. Last short
steps are 4,911 / 4,919; transition checkpoint times are 900.1737002409063 /
900.0861971820705 training seconds. Every saved Muon and AdamW group has the
unchanged peak LR 0.000326599 after the 554-step global warmup. Deterministic
four-rank replay through steps 5,140 / 5,200 matches all 515 / 521 logged token,
residue, sequence, source-count, context and balanced-rank-load records,
including the transition. Losses 2.728407859802246 / 2.7466418743133545 and
all gradients are finite; all eight L40S GPUs remain active. Receipts:
TRANSITION_VERIFICATION.json and TRANSITION_LR_VERIFICATION.json in the R09
candidate directory. Source/config snapshots still match. No selection
decision is made while training is active.

Next check remains 2026-09-05T19:11:49Z. Once both ROUND_STATE files are complete,
run `uv run --frozen python .dev/program2/verify_context.py --require-complete`,
then the standard collect.py R09-versus-R04 command above. Verify all final
evaluation receipts and apply the strict paired rule before keep/discard and
choosing the next literature-informed conceptual experiment.

## R09 discarded; R10 square-root target weighting (2026-09-05)

R09 completed both full-hour runs and all frozen evaluations. Validation
2.6034652069211006 / 2.6216077022254467; mean 2.6125364545732737, sample SD
0.012828681457348213, delta versus R04 -0.0083859171718359: **discard**.
Steps 12,668 / 12,663; tokens 615.437781M / 614.836529M; last-100 sequence
training losses 2.654574842453003 / 2.6655976462364195; P@L
0.07773831406920009 / 0.08922287298656473. Complete deterministic four-rank
replay verified all 1,267 logged records per seed through step 12,660, with
short-stage boundaries 4,911 / 4,919; final checkpoints and all evaluation
receipts passed collect.py. Restored only train.py from 0a2b17e and removed
the four new R09 files after snapshot verification. R04 remains incumbent:
mean 2.6041505374014378 on seeds 42 and 43.

All axes reviewed before selecting the next full pair:
- Algorithm/implementation: prototyped batching equal-shaped Muon matrices,
  inspired by the primary Muon code (batched orthogonalization) and Dao AI
  Lab's 2026 Gram Newton-Schulz discussion of batching across layers:
  https://github.com/KellerJordan/Muon/blob/master/muon.py and
  https://dao-lab.ai/blog/2026/gram-newton-schulz/. This prototype used ordinary
  batched GEMMs, not Gram Newton-Schulz or new dependencies. Eighteen tests
  passed with exact CPU native-update/state comparisons. On an idle kn081
  L40S, all 96 real-size hidden matrices took median 40.4532 ms native versus
  39.7571 ms batched, only 1.0175x optimizer speedup. GPU update relative
  difference <=0.007877 and cosine >=0.999969; momentum exact. The estimated
  sub-millisecond saving is too small to prioritize a full pair. Archived
  source, tests, configs, BENCHMARK.json and DECISION.json under
  candidates/r10_batchedmuon, marked preflight_not_launched. It has no full
  runs/results.tsv rows and is not an accepted or discarded training method.
  Restored all its changes before constructing the selected candidate.
- Model: retain accepted 24x768 ESMC and initialization. Prior depth-scaled
  initialization and input/residual routes failed; additional per-layer work
  also reduced throughput. Width/depth redistribution is a separate future
  ablation requiring complete frozen-evaluator compatibility validation.
- Data: the short-context curriculum lost on mean validation and contact
  diagnostics; retain context512 and original Bernoulli corruption on the
  frozen corpus/mixture, with accepted R04 rank balancing. No filtering or
  source reweighting is permitted.
- Loss: R03 token-weighting improved over its then-incumbent by 0.009208 but
  failed its 0.009347 SD threshold. This motivates a distinct intermediate
  objective, not acceptance of R03 or a change to the decision rule.
- Hyperparameters: retain every R04 optimizer, LR, weight-decay, warmup and
  batch setting. No LR change, tuning sweep, curriculum or model change is
  bundled with the selected loss ablation.

**Selected r10_sqrtloss:** for sequence i with m_i masked targets and mean
masked NLL l_i, optimize sum(sqrt(m_i)*l_i)/sum(sqrt(m_i)) over the whole
256-sequence four-rank batch. Empty sequences have zero weight. Compared
with equal-sequence weighting (weight 1) and global token weighting (weight
m_i), the square root gives more weight to estimates based on more targets
while limiting long-sequence dominance. For target counts 1 and 16, relative
sequence weights are 1:4 rather than 1:1 or 1:16. The choice of exponent 1/2
is predeclared; it is our tempered-weighting hypothesis, not a published
protein-model result. Within-sequence correlations and objective mismatch
could make it worse, so only the fixed validation rule can establish a keep.

Literature connection: Rives et al., *Biological structure and function emerge
from scaling unsupervised learning to 250 million protein sequences* (PNAS
2021), https://pmc.ncbi.nlm.nih.gov/articles/PMC8053943/, motivates masked-site
likelihood learning. The primary fairseq MLM criterion sums target NLL and
normalizes by masked-target count, verified directly at
https://github.com/facebookresearch/fairseq/blob/main/fairseq/criterions/masked_lm.py.
Those sources motivate the token-weighted endpoint, not this square-root
variant. No claim that their results transfer to our one-hour setup.

Implementation adds training_losses to train.py while preserving the exact
original default sequence_mean path. The candidate computes a differentiable
local numerator, pools detached numerator/weight statistics, and scales the
local loss by world_size/global_weight to cancel DDP's gradient averaging.
Accumulation is fixed to 1. Log `loss` remains the original sequence-mean
NLL, and `objective_loss` records the globally weighted objective separately.
The evaluator and selection metric are unchanged. No sampling or RNG change.
All 20 tests passed, including real two-rank CPU DDP against an independent
pooled reference with unequal target counts and an empty rank, all-empty
zero gradients, and exact default behavior. Required Ruff checks passed.

Candidate scope/snapshots: candidates/r10_sqrtloss/METHOD.json lists train.py,
new tests/test_training_losses.py, and exactly two R10 configs. Both configs
differ from their corresponding R04 configs only by
training_loss_reduction: sqrt_mask_count. Parameters remain 170,671,168.
At completion run .dev/program2/verify_sqrtloss.py --require-complete, then
collect.py --method r10_sqrtloss --incumbent r04_batchbalance --description
'Square-root masked-target-count weighting of sequence NLL, globally normalized; original sequence-mean diagnostic on R04.'
Enforce delta > candidate sample SD. On keep commit exactly these four files;
on discard/crash restore ONLY train.py from 0a2b17e and remove ONLY the new
test and the two R10 configs after checking snapshots. Retain all artifacts.

R10 launched concurrently at 2026-09-05T19:27:28Z, run suffix
20260905T192728Z, seed42/kn081 and seed43/kn056. Startup verified through step
220 on both nodes: original sequence-mean diagnostics 2.8226523399353027 /
2.7697317600250244, weighted objective diagnostics 2.805037021636963 /
2.7905426025390625. All gradients are finite, the 554-step warmup is correct,
and all 23 logged data/source/balance records per seed exactly match a fresh
four-rank deterministic training replay. All eight L40S GPUs active at
89-100% utilization with about 25-30 GiB reported memory use. Source/config
snapshots and launch contracts match. Receipt: STARTUP_VERIFICATION.json.

Next useful check is 2026-09-05T20:36:28Z, 69 minutes after launch. Resume in
this worktree, verify completion and snapshots using verify_sqrtloss.py
--require-complete, then collect against R04 and apply the strict rule before
continuing to the next experiment. The batched-Muon prototype was not launched
and must not be collected. The R09 curriculum is discarded and fully restored.

## R10 kept; R11 training masking density (2026-09-05)

R10 passed the predeclared rule and is the new incumbent. Validation losses
2.5902860425412655 / 2.59845782071352; mean 2.5943719316273928, sample SD
0.005778319759953412. Delta against R04 is 0.00977860577404499, greater than
candidate SD. Both paired seeds improved. Steps 10,387 / 10,328; tokens
628.061636M / 624.818840M, essentially unchanged throughput. Last-100 original
sequence-mean training NLL 2.646664769649506 / 2.644200019836426; full contact
P@L 0.10791944170645766 / 0.10274424536456686. Selection uses validation only.
Complete four-rank replay checked all 1,039 / 1,033 logged records through
10,380 / 10,320, and final native Muon checkpoints contain all 96 finite
original matrix momentum buffers. All frozen evaluation/checkpoint/config/
source/data/lock receipts verified. Committed only the four candidate files
as **bc06899d1873489d88c0e57e7f9e30a18fb8b70b**. Run rows retain commit NA;
candidates/r10_sqrtloss/DECISION.json maps the execution snapshot to the keep.

**Selected r11_mask20**, one training-corruption-density change on accepted
R10: use 20% independent Bernoulli target selection instead of 15%, retaining
the existing force-one rule for rows with canonical residues and mask-all
replacement. The accepted square-root weighting remains unchanged. More
supervised targets per forward pass may improve the one-hour compute tradeoff,
but hiding more context makes each prediction harder and differs from the
fixed validation corruption. This is a predeclared one-rate experiment; no
rate sweep or curriculum. No validation masking change is permitted.

Literature inspiration: Cheng et al., *Training Compute-Optimal Protein
Language Models* (NeurIPS 2024), Appendix C,
https://arxiv.org/html/2411.02142v1#A3, studies 85M and 154M protein models and
reports competitive 10-20% rates, with degradation at 5% and above 25%.
Their final recipe uses 15% and 80/10/10 replacement; our candidate retains
the incumbent mask-all rule and tests 20% under our own frozen evaluation.
Wettig et al., *Should You Mask 15% in Masked Language Modeling?* (2022),
https://arxiv.org/abs/2202.08005, frames the corruption-versus-prediction tradeoff
in text MLM; larger NLP rates are not assumed suitable for proteins.

All axes reconsidered: loss design just produced a qualifying gain at nearly
unchanged throughput, so preserve it and test target density rather than
combining another reduction. Data/corpus and source mixture stay frozen;
mask density changes supervision while keeping samples and context512.
R07 changed count variance at fixed expected density and failed, which does
not answer this density question. Model alternatives include width/depth
redistribution or gated residuals, but they need separate compatibility and
cost studies, and prior residual/init variants failed. Algorithm work on
batched Muon saved less than one millisecond per step in the prior preflight;
Gram Newton-Schulz would require its own numerical and hardware qualification.
All optimizer rates, 554-step warmup then fixed peaks, weight decay, batch64,
accumulation1 and the 3600-second guard remain accepted R10. No hyperparameter
changes beyond the selected corruption-density setting are bundled.

Implementation passes training_mask_probability to the existing unmodified
mask_tokens function, default .15, validates its range, and logs actual per-row
selected/eligible counts for rank0 before balancing on logged steps. Those
counts are diagnostics and consume no RNG. The original tokenizer, model,
loss implementation, evaluator, runners and lock are unchanged. Twenty tests
passed, including accepted loss/DDP and schedule tests; required Ruff checks
passed. A separate training-only 2,048-sequence check validates target labels,
mask-all semantics and 20% count statistics including forced-one expectation,
and proves omitted probability and explicit .15 have identical outputs and
RNG state. See candidates/r11_mask20/MASK_PREFLIGHT.json.

Scope is exactly train.py plus two R11 configs, listed in METHOD.json. Each
config equals the corresponding R10 config plus training_mask_probability:
0.20. Parameters remain 170,671,168. At completion run verify_mask20.py
--require-complete, then collect.py --method r11_mask20 --incumbent
r10_sqrtloss --description 'Training-only 20% Bernoulli mask density with original force-one mask-all corruption on accepted R10 sqrt loss.'
The verifier checks every logged probability/count bound, observed count
statistics against the forced-Bernoulli expectation, exact four-rank data
replay/balance, fixed LR and unchanged native optimizer shapes. Require both
full frozen evaluations. Keep only if R10 mean 2.5943719316273928 minus
candidate mean exceeds candidate sample SD. On keep commit only the three
candidate files; on discard/crash verify snapshots, restore ONLY train.py
from bc06899 and remove ONLY the two R11 configs. Do not remove the accepted
R10 loss helper/tests/configs. Retain all run and research artifacts.

R11 launched concurrently at 2026-09-05T20:43:21Z, suffix
20260905T204321Z, seed42/kn081 and seed43/kn056. Startup verified at step
170 for both runs: original sequence-mean losses 2.805922031402588 /
2.8101391792297363 and objective losses 2.800431251525879 /
2.8062868118286133, all finite. Every one of 18 logged records per seed
matches exact four-rank data replay and R04's retained balancing algorithm.
Observed masked fractions 0.2002195464 / 0.1982107963 over 266,003 / 271,741
eligible logged rank0 residues; forced-Bernoulli count z-scores 0.2831 /
-2.3317 pass the predeclared 8-sigma diagnostic bound. Actual per-row eligible
counts match training-data replay. All eight GPUs active at 87-100%, with
about 25-27 GiB memory reported. Source/config snapshots, native environment,
and global 554-step LR warmup verified. See STARTUP_VERIFICATION.json.
Training loss now reflects the harder 20% corruption; compare selection only
with the unchanged frozen validation, as specified by program2.md.

Next useful check: 2026-09-05T21:52:21Z (launch+69 minutes). Run
verify_mask20.py --require-complete and the collect.py command above once
both ROUND_STATE files are complete. The current incumbent is now R10,
commit bc06899, mean 2.5943719316273928; do not compare R11 against R04.

## R11 discarded; R12 headwise query/key normalization (2026-09-05)

R11 completed both full training runs and frozen evaluations. Validation
2.5891935601830482 / 2.5987551733851433; mean 2.5939743667840958, sample SD
0.006761081534284215; delta versus R10 0.0003975648432970047. Improvement
does not exceed candidate SD: **discard**. Steps 10,368 / 10,319; tokens
626.905599M / 624.272331M; last-100 sequence-mean training loss
2.6507987546920777 / 2.649727873802185; P@L 0.10704054534234746 /
0.09852292353444607. All 1,037 / 1,032 logged records passed complete four-rank
replay, count bounds and frozen receipts. Observed mask fractions .2001912452 /
.1999864752 (z=1.8817 / -0.1329). Restored only train.py from bc06899 and removed
only two R11 configs after snapshot verification. R10 remains incumbent at
bc06899d1873489d88c0e57e7f9e30a18fb8b70b, mean 2.5943719316273928.

**Selected r12_headnorm:** change the scope of Q/K LayerNorm from all 768
projection channels to each 64-channel attention head independently. Retain
all 768 existing per-channel gains for each Q and K normalizer, bias-free,
epsilon 1e-5. Keep the existing 1/sqrt(64) attention scaling and RoPE unchanged.
All 48 Q/K normalization modules change their statistics scope; input/FFN/
final/head normalizers stay as R10. The 24-layer, 768-wide, 12-head model still
has exactly 170,671,168 parameters and identical state-dict tensor names and
shapes. Fresh initialization and RNG state are bitwise identical to R10.
Training returns to incumbent 15% Bernoulli mask-all corruption and retains
accepted square-root loss and rank balancing.

Primary inspiration: Henry et al., *Query-Key Normalization for Transformers*
(Findings EMNLP 2020), https://arxiv.org/abs/2010.04245, normalizes Q/K within
heads to control attention saturation. Its L2 normalization and learned
attention scale are NOT copied here; this is a single-scope LayerNorm ablation.
Qwen3's primary implementation likewise normalizes Q/K along head_dim:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py.
That implementation uses RMSNorm with head-dimension gains, while this candidate
retains ESMC's per-channel LayerNorm gains. These sources motivate head-local
statistics, not a claim of protein validation improvement.

Hypothesis: normalization over the full projection couples one attention head's
statistics to other heads' activation means and scales. Independent head
statistics may improve head specialization and optimization under the retained
loss. This also removes a possible useful interaction and adds an explicit
affine multiplication, which could reduce throughput. The fixed one-hour
paired experiment measures the net effect; no speedup or stability gain is
assumed. Do not bundle RMSNorm, new temperatures, QKV splitting or LR changes.

All research axes reviewed: model normalization scope is distinct from R02's
replacement of every transformer norm with parameter-free RMSNorm and from
R06's optimizer QKV-block orthogonalization. Algorithm/implementation retains
native Muon because the batched prototype gave negligible timing benefit;
more aggressive Gram reformulation needs separate numerical qualification.
Data density at 20% just failed, and the short-context curriculum previously
failed, so keep fixed context512, 15% corruption, corpus/mixture and sampler.
Loss keeps the newly accepted square-root count weighting; no further exponent
selection is bundled. All learning rates, weight decay, momentum, batch64,
accumulation1, 554-step warmup and 3600-second stop remain fixed.

Implementation: ESMCHeadwiseLayerNorm subclasses nn.LayerNorm to retain native
gain initialization and checkpoint names. Its forward unflattens the last
axis into heads, computes LayerNorm per head, flattens back, and applies the
original per-channel gain in the result dtype. ESMCConfig serializes the new
qk_norm_per_head flag; false preserves the original module implementation.
train.py only forwards this model option. The new flag currently requires
transformer_norm=layernorm, avoiding an untested mixed normalization recipe.
Neither dense nor packed attention layouts, evaluator code, tokenizer,
contact probe protocol, validation masks nor dependency lock changes.

All 24 tests passed: independent per-head LayerNorm reference for forward and
input/gain gradients, head independence, exact initialized state/RNG, parameter
budget, unchanged attention feature layout and the real frozen checkpoint
loader, plus accepted loss/DDP/schedule tests. Required Ruff checks passed.
L40S preflight compared every full-size initial tensor/RNG to the incumbent,
ran synthetic full-model forward/backward with all 48 gains receiving finite
nonzero gradients, and compared dense versus packed tiny-model execution:
maximum logit error 0.0078125, relative full-gradient difference 0.00798556.
No held-out examples used. See candidates/r12_headnorm/MODEL_PREFLIGHT.json.

Candidate scope: model.py, train.py, new tests/test_headwise_qk.py and two R12
configs, exact list and hashes in METHOD.json. At completion run
verify_headnorm.py --require-complete, then collect.py --method r12_headnorm
--incumbent r10_sqrtloss --description 'Per-head Q/K LayerNorm statistics with unchanged per-channel gains and attention scale on accepted R10.'
The verifier checks source/config snapshots, exact four-rank training replay,
learning rates, saved headwise flag, 48 finite trained gain vectors and the
unchanged 96 Muon matrix states. Require both full frozen evaluations and keep
only if R10 mean 2.5943719316273928 minus candidate mean exceeds candidate SD.
On keep commit exactly these five files. On discard/crash verify snapshots,
restore ONLY model.py and train.py from bc06899 and remove ONLY the new test
and two R12 configs. Preserve accepted R10 loss code/tests/configs and artifacts.

R12 launched concurrently at 2026-09-05T22:00:47Z, suffix
20260905T220047Z, seed42/kn081 and seed43/kn056. Startup verified at step
170 for both: sequence-mean diagnostics 2.8115339279174805 /
2.815455913543701 and weighted objectives 2.800428628921509 /
2.8097872734069824. All losses/gradients finite; every one of 18 logged
records per seed matches exact four-rank data replay, source counts and
before/after balanced rank loads. The global 554-step warmup is correct.
All eight L40S GPUs active at 88-100% with about 31-33 GiB reported memory;
source/config snapshots and environment qualification pass. Receipt:
STARTUP_VERIFICATION.json. No intermediate loss is used for selection.

Next check is 2026-09-05T23:09:47Z (launch+69 minutes). At completion run
verify_headnorm.py --require-complete, then collect R12 against R10 using
the command above. Incumbent remains bc06899, mean 2.5943719316273928.

## R12 discarded; R13 Polar Express orthogonalization (2026-09-05)

R12 completed and passed both full frozen evaluations, checkpoint/config/
environment receipts and all 980/976 logged-record four-rank replay checks.
Validation 2.597066130489111 / 2.5969777330756187; mean 2.597021931782365,
sample SD 0.00006250641051968778, delta against R10 -0.0026500001549720764:
**discard**. Steps 9,794 / 9,752; tokens 592.173248M / 589.943802M, roughly
6% fewer than R10. Last-100 diagnostic loss 2.6475751185417176 /
2.6482248449325563; P@L 0.11161040132670207 / 0.10413160251777778.
Saved all 48 Q/K gains and original 96 Muon state shapes were finite and
verified. Restored only model.py/train.py from bc06899 and removed only
R12's test and two configs after preserving exact candidate snapshots.
R10 remains incumbent, mean 2.5943719316273928, commit
bc06899d1873489d88c0e57e7f9e30a18fb8b70b.

**Selected r13_polarexpress:** replace the fixed repeated polynomial in native
Muon's five orthogonalization iterations with the first five stabilized
iteration-specific Polar Express polynomials. Primary source: Amsel et al.,
*The Polar Express: Optimal Matrix Sign Methods and Their Application to the
Muon Algorithm*, https://arxiv.org/abs/2505.16932; published implementation
https://github.com/NoahAmsel/PolarExpress. The paper develops minimax polynomial
compositions for more accurate polar-factor approximation and reports language
model improvements in its settings. Our protein/one-hour setting is a separate
hypothesis, requiring the frozen two-seed test.

Generated coefficients directly from the authors' Python routine, using its
l=.001, num_iters=10, degree=5, safety_factor_eps=.01 and cushion=.02 settings,
and retained its first five polynomials. Also retain its initial 1.01 Frobenius
safety factor. Raw reference source and MIT license, SHA256, generator settings
and exact coefficient values are preserved in REFERENCE.json and adjacent
files. This is not NVIDIA's separately rounded Polar Express coefficient
list or the distinct optimized quintic coefficient sequence by You et al.
The runtime uses native-style fused torch.addmm arithmetic, so it does not
claim bitwise agreement with the authors' unfused bf16 expression. Both use
five iterations, bf16 and the same matrix transposition rule. No compilation,
SVD, spectral estimation or extra training data is used in the training loop.

The custom class subclasses the pinned native Muon and retains its momentum
buffer initialization, .95 momentum, Nesterov, decoupled weight decay, native
match_rms_adamw shape adjustment, per-group LR schedule and all original 96
fused hidden matrices. Only the orthogonalizer changes; checkpoint groups
explicitly record polar_express5 and the used coefficients. Native inherited
ns_coefficients are unused under this marker; ns_steps is required to be five.
Own optimizer checkpoint round trips are supported and tested. AdamW fallback
for embedding/head/scalars and all model parameter tensors remain unchanged.

All axes reviewed: R12's per-head model normalization lost both quality and
throughput, so retain the original ESMC normalization/model. The earlier batched
Muon prototype offered only 1.7% optimizer-step speedup; this round tests the
orthogonalization algorithm's approximation geometry at the same iteration
count. Data retains original 15% corruption, context512, corpus/mixture and
R04's exact-sample rank balancing because R11 density and R09 length changes
did not qualify. Loss retains R10's accepted square-root target-count weighting
and original sequence-mean diagnostic. LR, momentum, decay, batch size and
warmup are unchanged; no hyperparameter search is bundled into this round.

24 unit tests passed, including actual multi-process loss/DDP tests. New tests
check the polynomial using an independent SVD-based scalar reference, zero
and non-mutating updates, optimizer partition/default behavior, exact own
checkpoint resume, and bitwise agreement of every other native Muon update
when its orthogonalizer is substituted (including Nesterov on/off, missing
gradients, changing group LRs and decay). Required whole-repository Ruff check
and format check pass. 69 frozen source/config/evaluation/dependency hashes
match kickoff; model.py again matches the frozen baseline.

L40S preflight on all four actual matrix shapes found 2.45-2.67% relative
bf16 differences from the authors' unfused expression; all finite. On synthetic
square/tall/wide matrices, relative error to the SVD polar factor was
0.1210/0.0892/0.0865 versus native NS5 0.2213/0.2076/0.2032. Zero, rank-one and
ill-conditioned synthetic inputs passed finiteness checks. All full-model
170,671,168 parameters and 96 momentum buffers were verified; equal supplied
gradients produced bitwise-identical native momentum. Three alternating timing
rounds, each five warm and 20 measured steps, gave median native 41.108 ms vs
Polar Express 42.248 ms, a 2.77% optimizer-only cost. This cost and the changed
singular-value profile may counteract approximation benefits; selection remains
actual frozen one-hour validation. No training or held-out examples were used
in preflight. Receipt: OPTIMIZER_PREFLIGHT.json.

Candidate scope: train.py, new nano_protein/polar_muon.py, new
tests/test_polar_muon.py, and two R13 configs. METHOD.json contains exact list,
source/config hashes and base commit; source copies and candidate.patch saved.
At completion run `uv run --frozen python .dev/program2/verify_polarexpress.py
--require-complete`, then `uv run --frozen python .dev/program2/collect.py
--method r13_polarexpress --incumbent r10_sqrtloss --description 'Five stabilized Polar Express orthogonalization steps in native Muon framework on accepted R10.'`.
Verifier replays every four-rank sampled batch, checks all logged losses,
fixed LR and saved exact algorithm coefficients/96 original momentum shapes.
Require both full frozen evaluations, then keep only if R10 mean
2.5943719316273928 minus candidate mean exceeds candidate sample SD. On keep
commit only these five files. On discard/crash, verify snapshots, restore
ONLY train.py from bc06899 and remove ONLY polar_muon.py, test_polar_muon.py
and the two R13 configs. Preserve accepted loss/helper/tests/configs and all
run/research artifacts.

R13 launched concurrently at 2026-09-05T23:24:08Z, suffix
20260905T232408Z, seed42/kn081 and seed43/kn056. Startup verification passed
at step130 for both: sequence-mean diagnostic losses 2.8300108909606934 /
2.833995819091797, weighted objectives 2.849313259124756 / 2.8282790184020996.
All 14 logged records per seed matched exact four-rank sample replay, source
counts, total tokens/residues/sequences and R04 rank loads, with finite losses
and gradients and correct global warmup LR. All eight GPUs active at 87-100%,
about 25-27 GiB reported memory. Required Ruff and frozen environment checks
passed separately on both nodes. Receipt: STARTUP_VERIFICATION.json.

Next useful check: 2026-09-06T00:33:08Z (launch+69 minutes). Run
verify_polarexpress.py --require-complete, then collect R13 against R10 with
the command above once both ROUND_STATE files are complete. Incumbent remains
bc06899, mean 2.5943719316273928; no intermediate losses determine selection.

## R13 discarded; R14 query-dependent attention head gates (2026-09-06)

R13 completed both one-hour runs and both full frozen evaluations. Validation
2.582039177417755 / 2.615145470947027; mean 2.598592324182391, sample SD
0.023409684654500607; delta against R10 -0.004220392554998398: **discard**.
One seed improved while the other worsened; the method does not qualify under
the predeclared rule. Steps 10,345 / 10,300; tokens 625.537155M / 623.126214M;
last-100 diagnostic loss 2.6460794472694396 / 2.6441706228256225; P@L
0.10876522007730915 / 0.10152549723652571. Complete four-rank replay passed all
1,035 / 1,031 logged records, with saved exact Polar Express coefficients and
96 finite native-layout momentum states. All frozen evaluation, checkpoint,
config, data and environment receipts passed. Restored only train.py from
bc06899 and removed only polar_muon.py, test_polar_muon.py and two R13 configs
after snapshot verification. R10 remains incumbent at mean 2.5943719316273928,
commit bc06899d1873489d88c0e57e7f9e30a18fb8b70b.

**Selected r14_headgate:** add query-token-dependent scalar gating of each
attention head's SDPA output, before the existing output projection. For each
layer and input token with normalized representation x, gates = 2*sigmoid(Wx),
where W is a new bias-free 12x768 parameter matrix initialized to zero. Each
of the 12 gates multiplies its own 64-channel head output. This gives initial
gates exactly one, preserving the incumbent's initial function and all old
parameter tensors/RNG. Gate scores can approach zero or two after learning.
All 24 layers add 221,184 parameters in total; full model 170,892,352 <=171M.
The identity-centered factor two and zero initialization are deliberate
adaptations, not an exact reproduction of the cited paper's unscaled sigmoid.

Primary inspiration: Qiu et al., *Gated Attention for Large Language Models:
Non-linearity, Sparsity, and Attention-Sink-Free*,
https://arxiv.org/abs/2505.06708 and its author implementation,
https://github.com/qiuzh20/gated_attention (modeling_qwen3.py). The work compares
headwise and elementwise gates, reports gains from query-dependent gating
after SDPA, and relates this to nonlinear context filtering and attention
sinks. Its headwise alternative uses few extra parameters. These results are
from much larger text models; they motivate the mechanism, not a prediction
that this 171M bidirectional protein model will improve. We retain full Q/K
LayerNorm scope, RoPE, attention softmax and output projection as R10.

Hypothesis: per-token head gates let the model suppress irrelevant context or
amplify useful head outputs without changing the frozen data or objective.
This adds an input-dependent nonlinear interaction absent from the original
value/output map. Identity initialization avoids a global residual-scale
change at startup. The added matrix and elementwise kernels can reduce tokens
processed per hour, and gates may fail to specialize in the short training
budget; the paired fixed-time evaluation measures the net effect.

All research axes reviewed: algorithm returns to native NS5 Muon because R13
failed despite its synthetic polar approximation gain; the earlier batched
implementation brought little speed benefit. Model work now tests output
gating, distinct from R12 normalization scope and R08 layer-level residual
routing. GEGLU (Shazeer, https://arxiv.org/abs/2002.05202) was considered as an
alternative feed-forward nonlinearity, but head gates offer explicit
query-dependent context filtering at a small parameter cost. Data retains
original 15% mask-all corruption and context512 after R11 density and R09
context curriculum failed; SpanBERT's span-corruption idea
(https://arxiv.org/abs/1907.10529) is left for a separate experiment because it
changes reconstruction difficulty relative to the frozen isolated masks.
Loss retains the accepted global sqrt(target-count) weighted objective and
original sequence-mean diagnostic. No LR, warmup, momentum, decay, dropout,
mask density, normalization scope or initialization of old weights is tuned.

Implementation: direct zero parameter matrices consume no random numbers and
are untouched by the existing Linear/Embedding initialization traversal.
Attention computes the same pre-normalized x once for QKV and gates. Both
whole-transformer packed FlashAttention and dense/attention-feature paths
apply gates at the corresponding head axis. ESMCConfig serializes the new
attention_head_gating flag, default false, and expected_parameter_count adds
exactly n_layers*n_heads*d_model when enabled. The frozen evaluator already
rebuilds the saved model config; no evaluator/feature/probe change is needed.
Existing Muon partitioning naturally includes 24 new hidden matrices, total
120 with original 96 shapes unchanged. AdamW fallback is unchanged.

24 tests passed, covering bitwise initial old tensors/RNG/logits and unchanged
old gradients at identity, an independent explicit softmax/per-head gate and
projection reference (including gate/input gradients), budget and unique
native Muon allocation of all gate matrices, and the real frozen checkpoint
loader restoring trained nonzero gates. Accepted loss/DDP/schedule/model tests
also passed. Required whole-repository Ruff check and format check pass.
68 frozen evaluator/data-source/config/dependency/runner hashes match kickoff.

L40S preflight verified every full-size original tensor/RNG and zero maximum
initial logit difference. All 24 gate matrices obtained finite nonzero gradients
and changed under native Muon updates. With nonzero gates, tiny-model packed
vs dense maximum logit difference was 0.0078125 and relative full-gradient
error 0.0072426058; attention feature layouts remain finite and unchanged.
Full training-step timing (forward/backward/clip/optimizer), batch64 synthetic
variable-length proteins, two alternating rounds of 3 warm+6 measured steps:
native median 349.0445 ms, head gates 371.5502 ms, 6.45% cost. This cost is
material, similar to R12's throughput penalty; it is recorded before launch
and must be offset by actual quality gains. Model parameter count 170,892,352,
120 Muon matrices, and all parameters finite. No training/held-out examples
used. See candidates/r14_headgate/MODEL_PREFLIGHT.json.

Candidate scope: model.py, train.py, new tests/test_attention_head_gate.py and
two R14 configs, with exact files/hashes in METHOD.json and snapshots/patch
alongside. At completion run `uv run --frozen python
.dev/program2/verify_headgate.py --require-complete`, then `uv run --frozen
python .dev/program2/collect.py --method r14_headgate --incumbent r10_sqrtloss
--description 'Query-dependent head output gates initialized at identity with 2*sigmoid, on accepted R10.'`.
Verifier checks every four-rank sampled batch, finite original/weighted losses,
fixed LR, the saved flag, 24 finite trained gate matrices and all 120 native
momentum states. Require both full frozen evaluations; keep only if R10 mean
2.5943719316273928 minus candidate mean exceeds candidate sample SD. On keep
commit exactly these five files. On discard/crash verify snapshots, restore
ONLY model.py and train.py from bc06899 and remove ONLY the new test and two
R14 configs. Preserve all accepted R10 code, configs and research/run artifacts.

R14 launched concurrently at 2026-09-06T00:42:23Z, suffix
20260906T004223Z, seed42/kn081 and seed43/kn056. Startup verification passed
at step180 for both: original sequence-mean losses 2.7837772369384766 /
2.798212766647339, weighted objectives 2.8001351356506348 / 2.8129801750183105.
All 19 logged records per seed matched exact four-rank sampler replay, source
counts, cumulative tokens/residues/sequences and retained LPT rank loads.
Losses and gradients finite, global 554-step LR warmup correct. All eight
L40S GPUs active at 79-100% with about 26-28 GiB reported memory. Both nodes
passed required Ruff and frozen environment qualification. Receipt:
STARTUP_VERIFICATION.json. No interim loss is used for acceptance.

Next useful check: 2026-09-06T01:51:23Z (launch+69 minutes). Once both
ROUND_STATE files are complete, run verify_headgate.py --require-complete and
collect R14 against R10 with the command above. Incumbent remains bc06899,
mean 2.5943719316273928. R14 changes both model.py and train.py; restore both
on discard, unlike R13 which changed only train.py among accepted source files.

## R14 discarded; R15 AdaMuon adaptive orthogonal updates (2026-09-06)

R14 completed both full training/evaluation runs. Validation 2.604304561391473 /
2.6146236211061478; mean 2.6094640912488103, sample SD 0.0072966770997155775;
delta against R10 -0.015092159621417522: **discard**. Steps 9,630 / 9,599;
tokens 582.261267M / 580.677011M, about 7% fewer than R10; last-100 diagnostic
loss 2.6535282015800474 / 2.6552809953689573; P@L 0.10854930696833652 /
0.0985924071157604. All 964/960 logged records passed exact four-rank replay,
24 gate matrices trained with finite weight norms 1.498-3.410 / 1.582-3.289,
and all 120 optimizer states verified. Every frozen checkpoint, evaluation,
data, config and environment receipt passed. Restored only model.py/train.py
from bc06899 and removed only test_attention_head_gate.py and two R14 configs
after snapshot verification. R10 remains incumbent: mean 2.5943719316273928,
commit bc06899d1873489d88c0e57e7f9e30a18fb8b70b.

**Selected r15_adamuon:** substitute AdaMuon v3's adaptive orthogonal update
for native Muon's update, retaining the original model, loss and data. Primary
sources: Si et al., *AdaMuon: Adaptive Muon Optimizer*,
https://arxiv.org/abs/2507.11005v3 (24 Dec 2025 revision), and authors' code
https://github.com/Chongjie-Si/AdaMuon. The current version couples a sign
transform before orthogonalization with second moments of the orthogonalized
updates, then normalizes update RMS to 0.2. Its earlier v1 lacked sign
stabilization and used bias correction; this experiment follows v3's coupled
mechanism. Reported text-model gains motivate testing on proteins; their
training schedule and large-scale recipe are not copied.

Algorithm details: retain native .95 momentum and Nesterov, apply sign to its
update, then call the frozen Torch native five-step Newton-Schulz function
with coefficients (3.4445,-4.775,2.0315), epsilon1e-7. Track elementwise second
moments of the orthogonalized update in fp32, beta2=.95 inherited from the
same momentum as authors' code. Divide by sqrt(second_moment)+1e-8; scale to
RMS0.2 with a 1e-8 Frobenius denominator guard. Apply original per-group LR and
decoupled weight decay. Do not apply the native shape scale again after this
explicit RMS normalization. Bias correction cancels under this normalization
apart from epsilon and is omitted as in v3. Native momentum state layout and
96 matrix shapes remain, with one full-size second moment and diagnostic
adaptive_step counter per matrix. AdamW fallback is unchanged.

This is one optimizer replacement using its coupled sign/variance/RMS design,
not a combination of unrelated model/data changes. Normalized native momentum
is mathematically a positive constant times authors' unnormalized buffer, so
its sign-based direction is equivalent in exact arithmetic; finite-precision
sign sensitivity is characterized below. The implementation keeps the
incumbent's native fused NS arithmetic and uses fp32 second moments/final
updates, while authors' unsharded reference uses unfused NS and bf16 moments/
communicated updates. It is an explicit precision variant, not a bitwise
reproduction. Source and Apache-2.0 license are saved with hashes in
REFERENCE.json. No new dependency or distributed optimizer communication is
introduced; existing four-rank DDP provides identical averaged gradients.

All research axes reviewed: model returns to ESMC's original architecture
since R12 head norms and R14 output gates reduced throughput without quality
gains. GEGLU and partial rotary encoding are plausible future architecture
ablations; neither is bundled here. Data retains original corpus, mixture,
context512 and 15% mask-all corruption after density/context tests failed;
BERT-style random/identity replacements and span corruption change the
reconstruction task and need separate evidence. Loss retains accepted sqrt
count weighting; BLOSUM-based soft targets were considered but would bias
the optimized probabilities away from the frozen exact-residue NLL. Algorithm
work tests coordinate variance after R13's more accurate polar approximation
alone failed. Learning-rate-only tuning was considered, but the current Muon
already uses Moonshot's RMS-matched convention, so copying original-Muon's
0.02 rate would be inappropriate. No peak LR, weight decay, batch size,
554-step warmup or one-hour budget change is included. All group LRs remain
constant after warmup; coordinate preconditioning is part of the optimizer,
as it is for the existing AdamW fallback. No WSD/cooldown from the paper.

24 unit tests passed. New tests independently check signed NS inputs, native
momentum equivalence, multi-step second-moment/RMS algebra on square/tall/wide
matrices, Nesterov on/off, zero/missing gradients, unchanged supplied gradients,
changing group LR/decay, exact own checkpoint resume and unchanged parameter
partition/class defaults. Accepted loss including real CPU DDP and schedule/
model tests pass. Required whole-repository Ruff check and format check pass;
69 frozen source/config/dependency/evaluation hashes match kickoff and model.py
again matches the original architecture.

The first GPU comparison to authors' unmodified code exceeded an initially
chosen 8% update-difference bound on the square matrix (10.48%). The failed
preflight script is preserved, and the numerical cause was investigated rather
than treating this as training success. Native fused vs authors' unfused NS
outputs differed 2.34-2.75% in Frobenius norm, with sign flips at 0.022-0.349%
of entries. Flipped entries have RMS 0.00029-0.00057 versus whole-update RMS
0.0153-0.0333. At the first adaptive step, division by sqrt((1-beta)*O^2)
approximates sign(O), amplifying these small entries. Even normalized vs
unnormalized momentum rounding can flip near-zero signs and perturb later NS
outputs. See NUMERICAL_DIAGNOSIS.json; no candidate code was altered to hide
the mismatch.

A separate reference check supplied exactly the same orthogonalized update
and fp32 variance to the authors' optimizer body. All second-moment tensors
then agreed elementwise; parameter-update differences were only 0.097-0.130%
from the authors' final bf16 cast (required <0.5%). Unmodified reference
comparisons remain recorded separately: up to 10.55% on the first square
update, falling to 5.32% after three updates; other shapes 1.63-5.69% across
these checks. The original 8% raw bound was replaced by a documented 20%
diagnostic bound plus this stricter algebra check. Do not describe the
unmodified algorithms as numerically identical or omit this limitation.

L40S full-model qualification confirmed 170,671,168 parameters, 96 matrix
states, bitwise native momentum for equal supplied gradients, and finite
nonnegative fp32 second moments. Additional state: 679,477,248 bytes (~648 MiB)
per replica. Three alternating timing rounds, each five warm+ten measured
steps: native optimizer median 40.9337 ms, AdaMuon 56.2119 ms, an additional
15.2782 ms per step. This optimizer-only cost must be offset by better learning
within one hour; no throughput gain is assumed. All candidate matrices remain
finite. Only synthetic gradients were used, with no training/held-out inputs.
Receipt: OPTIMIZER_PREFLIGHT.json.

Candidate scope: train.py, new nano_protein/adaptive_muon.py, new
tests/test_adaptive_muon.py and two R15 configs. Exact files/hashes and rollback
base in METHOD.json; source copies and candidate.patch retained. At completion
run `uv run --frozen python .dev/program2/verify_adamuon.py --require-complete`,
then `uv run --frozen python .dev/program2/collect.py --method r15_adamuon
--incumbent r10_sqrtloss --description 'AdaMuon v3 sign-stabilized, variance-adaptive RMS-aligned updates on accepted R10; native NS5 and fp32 second moments.'`.
Verifier checks every sampled four-rank batch, logged finite losses and fixed
LR, saved adaptive marker/RMS/eps, all 96 original momentum shapes and 96
finite nonnegative same-shape fp32 second moments/counters. Require both full
frozen evaluations; keep only if R10 mean 2.5943719316273928 minus candidate
mean exceeds candidate sample SD. On keep commit exactly these five files.
On discard/crash verify snapshots, restore ONLY train.py from bc06899 and
remove ONLY adaptive_muon.py, test_adaptive_muon.py and two R15 configs. Model.py
is already restored and must not be edited. Preserve all R10 and run artifacts.

R15 launched concurrently at 2026-09-06T02:05:14Z, suffix
20260906T020514Z, seed42/kn081 and seed43/kn056. Startup verification passed
at step180 for both: original sequence-mean losses 2.7979178428649902 /
2.819993734359741, weighted objectives 2.81007719039917 / 2.824773073196411.
All 19 logged records per seed matched exact four-rank sample replay, source
counts, cumulative tokens/residues/sequences and retained LPT rank loads.
Losses/gradients finite, global 554-step warmup correct. All eight L40S GPUs
active at 89-100%, about 25-27 GiB reported memory. Both nodes passed required
Ruff and frozen environment qualification. Receipt: STARTUP_VERIFICATION.json.
Interim losses are diagnostic only and do not determine acceptance.

Next useful check: 2026-09-06T03:14:14Z (launch+69 minutes). Once both
ROUND_STATE files are complete, run verify_adamuon.py --require-complete then
collect R15 against R10 with the command above. Incumbent remains bc06899,
mean 2.5943719316273928. Preserve the documented reference-precision limitations
when describing this candidate. On discard restore only train.py among existing
source files, then remove the four new files listed in METHOD.json.

## R15 discarded; R16 GEGLU feed-forward gates (2026-09-06)

R15 completed both full training/evaluation runs. Validation 2.608914740383625 /
2.6098790802061558; mean 2.6093969102948904, sample SD 0.0006818912278797226;
delta against R10 -0.015024978667497635: **discard**. Steps 9,921 / 9,874;
tokens 599.866850M / 597.337956M, about 4.5% fewer than R10; last-100 diagnostic
loss 2.658961796760559 / 2.663573079109192; P@L 0.11239782877038226 /
0.1076438517263567. Complete four-rank replay passed all 993/988 logged records,
and all 96 momentum and second-moment states, counters, fixed LRs and frozen
receipts were verified. The failure applies to the tested native-NS/fp32
AdaMuon variant and recipe. Restored only train.py from bc06899 and removed
only adaptive_muon.py, test_adaptive_muon.py and two R15 configs after snapshot
verification. R10 remains incumbent: mean 2.5943719316273928, commit
bc06899d1873489d88c0e57e7f9e30a18fb8b70b.

**Selected r16_geglu:** replace SiLU in each feed-forward multiplicative gate
with exact Gaussian-CDF GELU, changing SwiGLU to GEGLU in all 24 FFNs. Preserve
the fused gate/up 4096x768 and down 768x2048 matrices, hidden width2048 and
170,671,168 total parameters. All existing initialized tensor values and RNG
state remain bitwise identical to R10; the activation changes the initial
function. Feed-forward pre-normalization, attention, RoPE, residual scaling
and the existing GELU MLM head remain as R10.

Primary source: Shazeer, *GLU Variants Improve Transformer*,
https://arxiv.org/abs/2002.05202. It compares parameter/compute-matched GLU
variants on T5 denoising, using width768 and gated FFN width2048, matching our
FFN dimensions. GEGLU and SwiGLU were the strongest perplexity variants;
GEGLU's reported log-perplexity was 1.942 vs 1.944 after 65,536 steps and
1.633 vs 1.636 after 524,288 steps. The small early difference overlaps the
reported variation, and the study uses text/encoder-decoder models. This is
motivation for a protein-MLM experiment, not proof of an expected gain. GELU
uses x*Phi(x), from Hendrycks and Gimpel,
https://arxiv.org/abs/1606.08415; runtime uses torch F.gelu default exact mode.

Hypothesis: replacing the logistic gate with a Gaussian-CDF gate changes how
negative/near-zero FFN features and gradients are weighted, which may improve
masked-residue prediction at the same parameter and matrix-compute cost.
Native Muon operates on exactly the same 96 shapes as the incumbent, so this
avoids adding parameter blocks or adaptive-state overhead. The different
activation variance is part of the architecture change; no compensating
initialization or residual-scale tuning is added.

All research axes reviewed: algorithm returns to native Muon after both
Polar Express and AdaMuon failed the fixed-budget selection rule. The small
batching-only prototype had negligible timing benefit; further implementation
work would need a larger measured gain. Model activation is distinct from
R12's normalization scope and R14's added head gates, which lost throughput.
Data retains the accepted exact-sample rank balancing, original context512
and 15% mask-all corruption after prior density/count/context trials failed;
span masking remains a separate potential reconstruction-task ablation. Loss
retains R10's accepted globally normalized sqrt(masked-count) weighting and
original sequence-mean diagnostic; soft targets or confidence weighting would
change NLL calibration and need their own evidence. All peak LRs, decay,
momentum, batch size, seeds, 554-step warmup and one-hour stop are retained.

Implementation: ffn_gate_activation in ESMCConfig accepts silu (default) or
gelu and is serialized in checkpoints; ESMCFeedForward selects only the gate
activation. The trainer forwards that config option. Existing default output
is exact, and unknown activation names fail immediately. The frozen evaluator
already reconstructs the model from saved config, so its code, mask protocol,
features, probes and reductions remain frozen.

22 tests passed: independent explicit erf/CDF GEGLU forward and input/parameter
gradients for dense and packed-shaped inputs; original parameter/RNG and
budget equality; exact default SwiGLU behavior; real frozen checkpoint loader
preserving activation, logits and attention outputs; accepted loss/CPU DDP,
schedule and model tests. Required Ruff checks pass. 68 frozen source/config/
dependency/evaluator/runner hashes match kickoff.

L40S preflight confirmed all full-size initialized tensors and RNG match R10,
all 24 FFN gate/up gradients are finite and nonzero, all parameters remain
finite after native optimizer steps, and there are 96 Muon matrices. Initial
maximum logit difference 0.63037109375 confirms the changed activation function.
Packed vs dense maximum logit error 0.005859375, relative full-gradient error
0.0072335901, with original finite attention-feature layout. Full training-step
benchmark (forward/backward/clip/optimizer), batch64 synthetic variable-length
inputs, two alternating rounds of 3 warm+6 measured steps: SwiGLU median
348.8517 ms, GEGLU 348.9843 ms. The 0.038% difference is effectively equal
within this short diagnostic; actual tokens per hour determine runtime impact.
Only synthetic protein sequences were used. Receipt: MODEL_PREFLIGHT.json.

Candidate scope: model.py, train.py, new tests/test_geglu.py and two R16 configs;
exact list/hashes in METHOD.json, source copies and candidate.patch retained.
At completion run `uv run --frozen python .dev/program2/verify_geglu.py
--require-complete`, then `uv run --frozen python .dev/program2/collect.py
--method r16_geglu --incumbent r10_sqrtloss --description 'GEGLU replaces SwiGLU in all 24 FFNs with identical matrices and initialization on accepted R10.'`.
Verifier checks every four-rank sampled batch, source counts, rank balance,
finite losses/gradients, fixed LR, saved GEGLU flag and original 96 native
momentum states without adaptive or Polar Express metadata. Require both full
frozen evaluations; keep only if R10 mean 2.5943719316273928 minus candidate
mean exceeds candidate sample SD. On keep commit exactly these five files.
On discard/crash verify snapshots, restore ONLY model.py and train.py from
bc06899 and remove ONLY test_geglu.py and two R16 configs. Preserve accepted
R10 code/configs and all run/research artifacts.

R16 launched concurrently at 2026-09-06T03:20:28Z, suffix
20260906T032028Z, seed42/kn081 and seed43/kn056. Startup verification passed
at step150 for both: original sequence-mean losses 2.819675922393799 /
2.814507007598877, weighted objectives 2.8326756954193115 / 2.8220057487487793.
All 16 logged records per seed matched exact four-rank sampler replay, source
counts, cumulative tokens/residues/sequences and retained LPT rank loads.
Losses/gradients finite and global 554-step warmup correct. All eight L40S GPUs
active at 88-100%, about 25-27 GiB reported memory. Both nodes passed required
Ruff and frozen environment qualification. Receipt: STARTUP_VERIFICATION.json.
No interim loss is used for acceptance.

Next useful check: 2026-09-06T04:29:28Z (launch+69 minutes). After both
ROUND_STATE files are complete, run verify_geglu.py --require-complete and
collect R16 against R10 using the command above. Incumbent remains bc06899,
mean 2.5943719316273928. Restore both model.py and train.py on discard, plus
remove only the three new candidate files listed in METHOD.json.

## R16 discarded; R17 controlled Muon update scale (2026-09-06)

R16 completed both full runs and frozen evaluations. Validation
2.5903162509202957 / 2.6123678386211395; mean 2.6013420447707176, sample SD
0.015592827199196527; delta against R10 -0.006970113143324852: **discard**.
Steps 10,366 / 10,349; tokens 626.784418M / 626.093633M, close to R10;
last-100 diagnostic loss 2.6472403144836427 / 2.6450966095924375; P@L
0.11062270811494453 / 0.10235221509343505. Full replay passed 1,037/1,035
logged records, with the saved GEGLU flag, original 96 native momentum states
and all frozen receipts verified. Restored only model.py/train.py from bc06899
and removed only test_geglu.py and two R16 configs after snapshot verification.
R10 remains incumbent, mean 2.5943719316273928, commit
bc06899d1873489d88c0e57e7f9e30a18fb8b70b.

**Selected r17_muonscale:** test twice the amplitude of native Muon's hidden
matrix update, with identical decoupled shrinkage at every optimizer step.
Existing config support sets muon_lr_scale=2.0 and muon_weight_decay_scale=0.5.
Their product cancels in the decay term, while the orthogonal update doubles.
Muon peak LR becomes .000653198 and weight decay .0091856. AdamW fallback
keeps peak .000326599 and its original decay/no-decay groups. All groups warm
linearly for exactly 554 global optimizer steps and then stay at these fixed
peaks. This is a controlled optimizer-scale experiment with two coupled config
values; no new optimizer algorithm or source implementation is introduced.

Primary motivation: Liu et al., *Muon is Scalable for LLM Training*,
https://arxiv.org/abs/2502.16982, Appendix A, reports that update RMS targets
0.2 and 0.4 performed similarly and outperformed more extreme tested scales.
The native match_rms_adamw convention already targets approximately 0.2 through
shape scaling; doubling its amplitude probes the corresponding approximate
0.4 regime. This is not strict per-update RMS normalization, and the paper's
results on text models do not establish the optimum here. Keeping decay
unchanged isolates update amplitude from additional parameter shrinkage.

Measured incumbent training trajectory supports testing faster optimization:
1,000-step bins starting at 7k and 9k have sequence-mean losses 2.66878 ->
2.65213 (seed42) and 2.66213 -> 2.65000 (seed43); weighted objectives
2.65566 -> 2.63816 and 2.65294 -> 2.63736. Mean gradient norms decline to
.37535/.36070 in the 9k bin. These training diagnostics show continuing
learning near the one-hour limit; they do not prove a higher rate is better.
Larger updates may improve early progress or worsen the constant-rate noise
floor. The frozen paired evaluation, not training bins, decides acceptance.
All trajectory bins and source metrics are recorded in CONFIG_PREFLIGHT.json.

All research axes reviewed: algorithmic changes to Muon (Polar Express and
AdaMuon) failed despite strong numerical motivation; retain the accepted
native NS5 directions and momentum while directly testing their scale. Model
GEGLU just failed with near-equal throughput, while head norms/gates lost
throughput, so retain original ESMC SwiGLU/normalization/attention structure.
Data keeps original corpus/mixture, context512, 15% corruption and exact-sample
rank balancing after earlier masking/count/context candidates failed; span
corruption and alternative context sampling remain separate task-design
hypotheses. Loss keeps the accepted sqrt-count objective and original logged
sequence-mean NLL; calibration-changing soft targets or focal weighting have
less direct support for this frozen likelihood target. This round deliberately
chooses a scale ablation after that broader review, with no architecture,
corruption, loss, batch-size, warmup-length or evaluation change bundled.

20 existing tests passed, including Muon group scaling, fixed warmup/peak
schedule, model budgets and accepted loss with real CPU DDP. Required Ruff
checks pass. All implementation files exactly match accepted HEAD; 69 frozen
source/config/dependency/runner/evaluator hashes match kickoff (train.py matches
accepted R10). Only the two R17 configs are candidate files. The preflight
builds actual optimizer groups at steps1,2,553,554,555,10000, verifies every
LR*WD product is unchanged and AdamW rates match, and obtains bitwise exactly
2x Muon parameter updates on square/tall/wide synthetic matrices across three
steps each while preserving identical momentum buffers. No new tests or GPU
benchmark are needed for unchanged computation; actual training will establish
stability at the larger rate. Parameter count remains 170,671,168.

Candidate scope: only configs/program2/r17_muonscale_seed42.yaml and seed43.
Exact hashes/source receipts in METHOD.json, copies and patch retained. Check
after warmup at launch+5 minutes with `uv run --frozen python
.dev/program2/verify_muonscale.py --require-peak`; this requires >=600 steps
and exact doubled LR in all logged rows, including the constant phase. If
healthy, schedule the full completion check at launch+69 minutes without
starting another method. At final completion run the same verifier with
--require-complete, then `uv run --frozen python .dev/program2/collect.py
--method r17_muonscale --incumbent r10_sqrtloss --description 'Double native Muon update amplitude while preserving per-step weight shrinkage and AdamW rates on accepted R10.'`.

Require both full frozen evaluations, exact four-rank data replay and final
optimizer receipts. Keep only if R10 mean 2.5943719316273928 minus candidate
mean exceeds candidate sample SD. On keep commit exactly the two configs;
they define the new recipe while training/model source remains unchanged.
On discard/crash verify snapshots and remove ONLY the two R17 configs. No
tracked source file needs restoring for this candidate. Preserve accepted
R10 and all research/run artifacts.

R17 startup and post-warmup checkpoint verified 2026-09-06T04:42Z. Both runs
are state=running, all four L40S per node active. STARTUP_VERIFICATION.json
replayed 750 steps for each seed; PEAK_VERIFICATION.json replayed 790 steps
(seed42 kn081, 80 logged records) and 800 steps (seed43 kn056, 81 records).
All recorded loss/objective/gradient norms are finite, exact four-rank data
replay and source/config snapshots match. Every logged LR follows the
554-step warmup and doubled Muon peak thereafter. No divergence detected
at this checkpoint. The planned early check was performed in this same
invocation; schedule only the full completion check at 2026-09-06T05:44:59Z.
Do not collect or launch another method until both final evaluations complete.


R17 final decision 2026-09-06T05:48Z: DISCARD. Val losses 2.579673022031784 / 2.602123912423849; mean 2.5908984672278166, sample SD 0.0158751768399051, delta 0.003473464399576187 does not exceed SD. R10 bc06899 remains incumbent. Both final checkpoints/evaluations and exact four-rank replay verified; 10375/10339 steps, 627.332485/625.485368M model tokens, training loss 2.644584357738495/2.643554298877716, full P@L .10516094944375481/.10022758362063318. Checkpoint SHA42 fe13ead944d854b5b80d55da96654b5f0fa56033f30ee8a747de852594976826; SHA43 9ee301baa657f928c048660cd8ada27804e6b116e7ac83d77d5bd63365a935c0. Removed only two candidate configs after hash verification; source never changed. A better mean alone does not qualify under the prescribed variance rule.

## R18 proposal 2026-09-06: ESM-2-style token dropout on R10

One model input-design change: zero the learned mask-token embedding, then
rescale all visible embeddings per sequence by .85 * L / (L - M), where L is
valid token length including BOS/EOS and M counts visible mask tokens. Apply
the same transform in training, frozen MLM evaluation and attention/contact
inference (unmasked input scale .85). This is not additional stochastic
masking: original corrupted inputs/labels and all RNG streams remain intact.
Only token_dropout=true is added to the R10 recipe; no parameters are added,
all 170,671,168 tensors initialize bitwise identically. The unused mask row
remains a counted parameter and receives zero input gradients, while AdamW's
decoupled shrinkage still applies. Denominator is clamped for all-masked and
empty degenerate rows, which ordinary tokenizer-produced examples exclude.

Primary implementation: https://github.com/facebookresearch/esm/blob/main/esm/model/esm2.py
(lines79-88 in downloaded ESM2_REFERENCE.py). ESM-2 zeros mask embeddings and
scales by (1-.15*.8)/(1-observed_mask_fraction). R18 adapts its nominal visible
fraction from .88 to .85 because our frozen training tokenizer uses 15%
mask-all, not ESM-2's 80/10/10 corruption. As in the reference, observed length
includes special tokens; .85 is a nominal correction, not the exact expected
visible rate for every sequence, given noncanonical tokens and forced-one
mask selection. Our input transform follows the formula, but ESMC remains a
different architecture/optimizer/training regime. No claim of reproducing all
ESM-2 settings. Reference and MIT license are preserved with exact hashes in
METHOD.json; source was read, not imported or executed.

Hypothesis: learned shared mask vectors may bias early attention and the
residual stream at supervised positions. Zeroing them requires context to
construct those representations; visible-fraction scaling compensates for
sequence-to-sequence mask density differences without changing target weights.
At initialization first-layer masked Q/K vectors become zero, hence initially
uniform masked queries; learning can alter this through normalization gains/
biases and later layers. Could improve frozen NLL with minimal overhead, or
harm the current pre-norm ESMC optimization. Paired frozen NLL decides.

All research axes reviewed: retain native Muon and its original scale because
R13/R15 algorithm replacements and R17 higher scale failed the paired rule.
R12/R14 more elaborate model additions incurred throughput losses and R16
GEGLU failed even at equal cost; input token dropout offers a distinct,
protein-specific mechanism with no matrix additions. Data/corruption retains
context512, 15% mask-all and exact-sample LPT balancing. Considered 80/10/10 or
span corruption: protein scaling work uses 80/10/10 (Cheng et al.,
https://arxiv.org/html/2411.02142v1 Appendix C), but Wettig et al.
https://aclanthology.org/2023.eacl-main.217/ disentangle corruption difficulty
from prediction count and show corruption schemes are not universally best;
changing corruption introduces target-task mismatch with frozen mask-all NLL.
Defer that independent hypothesis. Loss retains sqrt-mask-count weighting,
which has the strongest accepted paired evidence. Soft-label/focal objectives
can bias exact likelihood and lack a direct improvement signal here. No data,
loss, optimizer, architecture-width/depth or schedule changes are bundled.

Qualification: 23 tests passed (3 token-dropout, 20 existing training loss/
budget/model tests including real CPU DDP). Independent per-row embedding and
gradient reference, degenerate masks, no-RNG behavior, unmasked inference and
frozen checkpoint loading passed. Full-size L40S check confirms bitwise initial
weights/RNG, all 24 FFN matrices have finite nonzero gradients, input mask row
gradient is exactly zero, canonical embedding gradients are nonzero, and the
same 96 native Muon matrix states apply. With flag disabled, logits/weights/
RNG match the immutable accepted model source bitwise. Packed-vs-dense BF16
max logit error .0078125 and relative gradient error .0066169463; attention
features keep original layer/head/token layout. Full synthetic step median
349.1442ms baseline versus 349.6186ms candidate (+.136%, near timing noise),
with alternating-order repeats. All qualification uses synthetic inputs,
without touching held-out data. Exact source/config snapshots and 68 frozen
source/evaluator/lock/config checks recorded before launch.

Candidate files: model.py, train.py, tests/test_token_dropout.py and two R18
configs, enumerated in candidates/r18_tokendrop/METHOD.json. Next completion:
`uv run --frozen python .dev/program2/verify_tokendrop.py --require-complete`,
then `uv run --frozen python .dev/program2/collect.py --method r18_tokendrop
--incumbent r10_sqrtloss --description 'Apply ESM-2-style zero mask embeddings with per-sequence visible-fraction rescaling adapted to 15% mask-all corruption on accepted R10.'`.
Require both complete final checkpoints/frozen evaluations, exact four-rank
sampler replay and saved zero mask-row Adam moments plus 96 Muon buffers.
Keep only when 2.5943719316273928 minus candidate mean exceeds its sample SD.
If discarded, restore only model.py/train.py from bc06899 and remove only
new test and two configs after snapshot validation; preserve all artifacts.

R18 launched both seeds at 2026-09-06T05:53:53Z; suffix20260906T055353Z.
Startup verified 2026-09-06T05:55Z: seeds42/kn081 and43/kn056 each reached
step150 with finite loss/objective/gradient and exact replay of all four-rank
sampled batches, cumulative token/residue/sequence counts, source counts and
LPT loads. All source/config snapshots match. Original554-step warmup verified
on all16 logged records per seed. Both ROUND_STATE=running; all eight L40S
active. STARTUP_VERIFICATION.json saved. Next completion check scheduled
2026-09-06T07:02:53Z (launch+69min), using the procedure above.


R18 final decision 2026-09-06T07:07Z: DISCARD. {"method_id": "r18_tokendrop", "status": "discard", "validation_loss_mean": 2.591290893033147, "validation_loss_std": 0.01628985968240379, "delta": 0.0030810385942459106, "incumbent_remains": "r10_sqrtloss", "restored": "Restored only model.py/train.py from bc06899 and removed only test_token_dropout.py and two R18 configs after complete verification."}
Both final checkpoints, frozen evaluations and exact four-rank data replay verified. Validation42/43: 2.579772222787142 / 2.602809563279152; steps10363/10337, tokens626.610769/625.361198M; train2.6475074005126955/2.643714277744293; full P@L .10803366055835385/.11281978530497391. SHA42 e56bdb7a7f5f5b21c07d2667f9109514240452fdb752ae00119ee0df81c25ad9; SHA43 584e00e9f081715a9275b7c7f6fdb99fb44a2e1d25e4ef4fb5a7b88b86ff7467. Mask-row Adam first/second moments were exactly zero as designed, original96 Muon buffers retained. The mean gain remains smaller than candidate variation, so R10 remains incumbent.

## R19 proposal 2026-09-06: training 80/10/10 replacements on R10

One denoising-task/data-presentation change: after the unmodified tokenizer
chooses its original Bernoulli15% targets (including forced-one fallback),
replace each selected position with mask/random/original with probabilities
.8/.1/.1. Keep all original target indices and labels in the accepted sqrt
mask-count objective. Random replacement draws uniformly from the 20 canonical
amino acids and may equal the original residue. Protected/unselected tokens,
attention masks, corpus, mixture, sampling, crops and LPT balancing remain
unchanged. A separate device generator seeded1900003+seed+rank ensures all
replacement RNG draws leave the original target-mask RNG stream untouched.
No token dropout, model change, optimizer change or loss-weight change.
Model/evaluator/tokenizer source matches the accepted R10 recipe exactly.

Literature: Cheng et al., Training Compute-Optimal Protein Language Models,
https://arxiv.org/html/2411.02142v1 Appendix C explicitly chooses 15% selection
with 80/10/10 replacement for its protein MLMs; its85M/154M masking sweeps found
10–20% most effective, but do not isolate 80/10/10 versus mask-only. This is
protein-specific precedent, not proof that the replacement rule itself helps.
Counterevidence: Wettig et al., Should You Mask 15% in MLM?,
https://aclanthology.org/2023.eacl-main.217.pdf Section7/Table4, finds same-token
and random replacements worse on most evaluated text fine-tuning tasks and
recommends mask-only there. Its decomposition distinguishes supervision count
from corruption difficulty, motivating a controlled comparison here. Uniform
canonical20 random sampling is our explicit protein adaptation; no noncanonical
or special token is injected and no distribution is inferred from held-out data.

Hypothesis: random-residue denoising may train predictions to be robust to
local substitutions, while unchanged-target supervision exercises normal
residue representations and provides a small auxiliary reconstruction signal.
At fixed15% selected targets, only ~12% eligible positions are explicit masks;
~1.5% are random and ~1.5% original, ignoring forced-one/special-token effects.
This may improve conditional representations or weaken learning on masked
sites. Because some targets are now copies, a lower training loss cannot be
interpreted as a quality gain across recipes. Frozen32-sequence/994-target
mask-all validation stays unchanged and alone controls acceptance; full20775
P@L remains diagnostic. No held-out data is used in design qualification.

All axes reviewed: native Muon geometry and original LR remain after R13/R15
and R17 failures. R18 mask-input transform and R16 gate activation failed even
at equal throughput, while R12/R14 added normalization/gating incurred cost;
retain original model and test corruption independently. Span masking remains
another task-design possibility, with a larger change in conditional context
and less direct protein evidence than the common80/10/10 recipe. Loss retains
accepted sqrt-count normalization; changing exponent would revisit a narrow
weighting axis, and calibration-altering soft/focal targets are less direct
for the fixed likelihood metric. The current corruption choice tests both
context denoising and supervision content without changing selected positions.

Qualification: 24 tests passed (4 corruption +20 existing model/budget/loss
including real CPU DDP). Independent per-position reference checks exact
outputs, labels, protected tokens and forced-one fallback. A200000-target
sample checks .8/.1/.1 rates and uniform canonical draws including original
matches. CPU tests and CUDA checks verify exact global RNG preservation over
multiple batches; CUDA qualified all8 seed/rank streams for16 batches each.
Full-size L40S model preserves identical initialization and original96 Muon
matrices with finite gradients in all24 FFNs. Synthetic full-step median
348.8268ms baseline vs348.9872ms candidate (+.046%, timing noise scale);
includes replacement helper but not distributed communication or periodic
category logging. Actual one-hour throughput decides its budget impact.
Unchanged model packed/dense paths qualify with maxlogiterror .0068359375,
relativegradienterror .0066458161. Required Ruff passes. Exact source/config
snapshots and69 frozen source/config/dependency/evaluator hashes verified.

Candidate files are exactly nano_protein/train.py, tests/test_bert_corruption.py
and two R19 configs, enumerated in candidates/r19_bertcorrupt/METHOD.json.
Logged rank0 pre-balance counts permit replay checks of selected-target laws,
.8/.1/.1 categories, uniform random amino acids and random-original matches.
Before collection run `uv run --frozen python .dev/program2/verify_bertcorrupt.py
--require-complete`, then `uv run --frozen python .dev/program2/collect.py
--method r19_bertcorrupt --incumbent r10_sqrtloss --description 'Replace the same selected MLM targets with 80% mask, 10% uniform canonical amino acid and 10% original residue using an independent RNG, on accepted R10.'`.
Require both final frozen evaluations, exact four-rank data replay and all
model/optimizer/data/lock receipts. Keep only if R10 mean2.5943719316273928
minus candidate mean exceeds candidate sampleSD. On discard restore only
train.py from bc06899 and remove only the new test and two configs after
snapshot verification; preserve all run/research artifacts.

R19 launched both seeds at 2026-09-06T07:12:39Z; suffix20260906T071239Z.
Startup verified 2026-09-06T07:14Z: seeds42/kn081 and43/kn056 each reached
step180 with finite loss/objective/gradient and exact four-rank sampler replay,
source counters, cumulative token/residue/sequence counts and LPT loads. All
source/config snapshots match; original554-step warmup verified on19 records
each. Logged selected targets42140/42715, eligible280064/287631, target-rate
z-scores .6901/-2.2436 consistent with unchanged15% plus forced-one. Category
mask/random/original totals33850/4144/4146 and34194/4227/4294, all categorical
checks pass along with uniform canonical replacement and original-match rates.
Both ROUND_STATE=running and all eight L40S active. STARTUP_VERIFICATION.json
saved. Next completion check2026-09-06T08:21:39Z (launch+69min).


R19 final decision 2026-09-06T08:25Z: DISCARD. {"method_id": "r19_bertcorrupt", "status": "discard", "validation_loss_mean": 2.6157104708254337, "validation_loss_std": 0.0029317400507738525, "delta": -0.021338539198040962, "incumbent_remains": "r10_sqrtloss", "restored": "Restored only train.py from bc06899 and removed only test_bert_corruption.py and two R19 configs after full verification."}
Both final checkpoints/frozen evaluations and exact four-rank replay verified. Validation42/43: 2.6136374175548553 / 2.617783524096012; steps10363/10318, tokens626.610769/624.213855M; train2.4895182061195373/2.493269908428192; full P@L .10253929271035707/.10605230938619957. SHA42 842234eced07c62a3e1568d0679c172b24e9b8ed37f1e6059f688d9b5e36ca03; SHA43 f38bc99eb9b5d650b013b01a6122675d240a681d55682a2fd30d82d4ecdb22b8. Final logged selected fractions .15006485/.15002145, branch and random-residue counts satisfy all preregistered checks. Training loss is lower partly because targets include copies; both frozen mask-only validation losses are worse. R10 mask-all remains incumbent.


R20 FFN normalization prototype: PREFLIGHT_NOT_LAUNCHED, not a completed method and no results.tsv entries. NormFormer-inspired affine LayerNorm after SwiGLU and before down projection added98304 parameters, total170769472. 23 tests and full-size L40S correctness/gradient checks passed; 48 new vectors learned and original tensors/RNG identical. However, step348.9416ms baseline vs413.3991ms candidate (+18.47%). Synthetic initial first4/last4 gradient ratios QKV5.447->44.168, gate-up5.780->9.822, down6.000->9.512; after18 repeated synthetic updates QKV22.477->331.799. The intended balancing mechanism was not observed, and computational cost is large. Repeated synthetic training fit became faster but is not evidence of held-out quality. Archive all files/reference/license/receipts under candidates/r20_ffnnorm; restore only its five changes. No hour-long launch or collector invocation for this prototype. Continue R20 with a distinct compute-saving method ID.

## R20 selected method: selective final FFN/head on R10

The unlaunched FFN norm prototype above is not this experiment. R20's launched
method ID is r20_lastsubset. One algorithmic implementation change: after full
final attention, select exactly supervised positions for the last block's
SwiGLU FFN, final norm and prediction head, then scatter results back into the
original dense output shape. All preceding layers and all final attention
queries/keys/values remain full. Unselected output entries are zero because
training ignores them; callers requesting full features cannot also request a
prediction mask. The frozen evaluator never passes this optional argument and
therefore retains the original complete model function. Parameter tensors,
architecture config, tokenizer, data/masks, loss and optimizer are unchanged.

Primary implementation precedent:
https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/models/bert.py
BertEncoder.forward subset_mask computes final-layer outputs only for selected
positions and retains full key/value context; BertForPreTraining supports
selected MLM prediction. That implementation also restricts final queries with
cross-attention. R20 deliberately uses a smaller optimization: full attention
including full queries, then selective FFN and head. It adapts the dependency
argument to the frozen native packed PyTorch kernels without new dependencies,
new attention kernels or architecture changes. Reference and BSD3 license are
archived with exact hashes. No reference code is imported or executed.

The last FFN and head act independently on each token. No following attention
can use the unselected FFN outputs, so removing those operations preserves the
mathematical selected logits and gradients. Sparse-mask, all-target and empty
cases are supported. This avoids spending final tokenwise work on roughly85%
of positions with ignored labels; all context paths still receive gradients
through attention. BF16 GEMM reductions can round differently with smaller
matrix row counts, so bitwise optimization trajectories are not claimed.

All axes reviewed: R19's random/original replacement task worsened both frozen
losses despite lower train loss, so retain mask-all. Prior context, mask-count
and density alternatives failed; retain exact-sample rank balancing and ctx512.
The accepted sqrt-mask-count objective remains strongest loss evidence, and
soft/focal target losses alter calibration. Keep native Muon geometry and
original LR after algorithm/scale candidates failed. NormFormer-inspired model
normalization was explored with a bounded preflight, but cost and gradient
measurements did not support that candidate; it is fully archived. Selective
computation instead targets budget efficiency using the accepted model/loss,
without assuming a new statistical objective will improve generalization.

Qualification: 23 tests passed, including CPU-double equality of all model
parameter gradients and selected logits/hidden states for sparse/all/empty
target sets (also with learned residual routing), and actual2-rank DDP with
one empty-target rank. Frozen checkpoint reload/default attention features and
prediction-mask API contracts pass. Default full evaluation logits/attention
features match immutable accepted source bitwise on CPU. Full-size L40S
initial supervised logits and objective both match exactly in this fixture;
all-parameter gradient relative difference .00144257 (~.144%, BF16 rounding).
All24 FFNs receive finite nonzero gradients and original96 native Muon
matrices remain. Dense-full vs packed-subset tiny comparison maxlogiterror
.005859375, relativegradienterror .00697765. Full synthetic step median
348.9346ms baseline vs342.9672ms candidate (1.71% less time,1.0174x throughput);
alternating-order runs, each3 warm+6 measured steps. Timing includes full
forward/backward/clip/optimizer and subset selection, but actual distributed
one-hour throughput and frozen validation decide acceptance. Parameter count
170671168. Required Ruff and68 frozen source/config/lock/evaluator hashes pass.

Candidate scope: model.py, train.py, tests/test_last_layer_subset.py and two
r20_lastsubset configs (five files in METHOD.json). After both complete, run
`uv run --frozen python .dev/program2/verify_lastsubset.py --require-complete`,
then `uv run --frozen python .dev/program2/collect.py --method r20_lastsubset
--incumbent r10_sqrtloss --description 'Compute the final FFN and prediction head only at supervised MLM positions, retaining all attention context and the accepted R10 objective.'`.
Require both final checkpoints/evaluations and all replay/config/data/optimizer
receipts. Keep only if2.5943719316273928 minus candidate mean exceeds candidate
sampleSD. On discard restore only model.py/train.py from bc06899, remove only
the new subset test and two configs after snapshot verification, and preserve
all research/run artifacts. Do not collect r20_ffnnorm: it was never launched.

R20 lastsubset launched both seeds at2026-09-06T08:37:24Z; suffix20260906T083724Z.
Startup verified2026-09-06T08:39Z: seeds42/kn081 and43/kn056 each atstep200,
finite original loss, objective and gradients; selective_final_ffn=true and
valid supervised-position counts in all21 logged records. Exact four-rank
sampler replay, source counts, token/residue/sequence totals and LPT loads
match accepted data presentation; all source/config snapshots match and
original554-step warmup is correct. Both ROUND_STATE=running and all8 L40S
active. STARTUP_VERIFICATION.json saved. Completion wake2026-09-06T09:46:24Z
(launch+69min). r20_ffnnorm remains unlaunched; only collect r20_lastsubset.


R20 lastsubset final decision2026-09-06T09:50Z: DISCARD. {"method_id": "r20_lastsubset", "status": "discard", "validation_loss_mean": 2.595495877787471, "validation_loss_std": 0.005452245398092816, "delta": -0.0011239461600780487, "incumbent_remains": "r10_sqrtloss", "restored": "Restored only model.py/train.py from bc06899 and removed only subset test and two R20 configs after completed verification."}
Both final checkpoints/frozen evaluations and exact four-rank replay verified. Validation42/43:2.5916405580937862/2.5993511974811554; steps10552/10494, tokens638.024468/634.866053M; train2.646087830066681/2.6462883591651916; fullP@L .10845685666234665/.10507654510404775. SHA42 1a98e9903b8b3c51326590252d0cef070eb73c7568df1b1baa5aada0266d5fb2; SHA43 a80aafff07dba8d0f45d02b7a2735b99ce3c561fbc1e5c143d15e41fb7a88dbf. Token throughput improved about1.6% over R10, but both fixed-seed losses are slightly worse, so the strict loss rule does not retain the speed change. All code/recipe restored; optimization remains archived for later research.

## R21 proposal2026-09-06: smaller batch on R10

One configuration change: per-rank micro_batch_size64->32, global batch256->128
on four GPUs with accumulation1. This reallocates the fixed hour between
optimizer updates and examples per update. The model, native Muon/AdamW recipe,
15% mask-all corruption, sqrt-mask-count global loss normalization, context512,
corpus/mixture, LPT balancing and frozen evaluation stay as accepted R10.
All source files match bc06899 exactly; R20's unaccepted speed change is absent.

Primary literature: Marek et al., Small Batch Size Training for Language
Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful,
https://arxiv.org/html/2507.07101v1 studies small-batch language model training
including Muon and describes improved per-FLOP efficiency/robustness in its
settings. It emphasizes scaling Adam beta2 to preserve a token half-life and
selecting a small batch that maintains throughput. Our modest2x batch ablation
holds all per-step optimizer settings fixed; it is not a reproduction of the
paper's rescaled-Adam or batch-one recipe. McCandlish et al.,
https://arxiv.org/abs/1812.06162 relates useful batch size to gradient noise and
motivates weighing statistical efficiency against hardware throughput. No
noise-scale estimate or optimal-batch claim is made for this protein model.

Hypothesis: more frequent updates using fresher parameters can improve early
optimization within the hour, even though each gradient averages fewer
sequences. Increasing update magnitude alone (R17) did not qualify; smaller
batches also change gradient sampling and the number of optimizer evaluations.
This could improve progress, increase variance/noise, or worsen the fixed-LR
noise floor. Exactly554 GLOBAL optimizer steps of warmup remain mandatory;
therefore warmup sees half as many examples. Adam/Muon moment half-lives in
examples decrease and decoupled weight shrinkage is applied more times per
hour. These are consequences of the single batch change; no LR, beta, momentum
or weight-decay compensation is bundled. All post-warmup group peaks stay fixed.

All axes reviewed: recent algorithmic efficiency R20 increased actual token
throughput~1.6% but failed frozen loss selection; retain accepted source. Model
norm/head/gate/input variants failed, and the FFN-normalization prototype had
18.5% overhead with worse measured gradient imbalance. Data/context/masking
variants, most recently80/10/10, did not outperform the frozen mask-all target.
Loss's sqrt-mask-count reduction remains the strongest accepted improvement;
focal/soft objectives would alter likelihood calibration and lack direct
support here. Rather than combining these changes, this round tests the
batch/update budget after that broader review. This is a deliberately
configuration-only round, not a restriction of research to hyperparameters.

Sampler caveat: the unchanged MixtureBatcher draws source choices for an entire
batch before crop offsets, so32-row calls interleave RNG differently from64-row
calls. Frozen corpus, mixture probabilities, per-source disjoint shuffled-row
rules and masking law are retained, but a bitwise R10 sample-prefix match is
not claimed. verify_batch32.py replays the exact32-row sampling process for all
four ranks, including source counts, LPT assignment and cumulative totals.
Expected per step:128 sequences, two special tokens each, and dynamically
measured full-context tokens. Training loss remains the final100 logged
rank0 sequence-mean NLL values, using the same metric definition.

Qualification: 20 existing tests passed (weighted loss including actual CPU
DDP, model and training budget tests); required Ruff passes. Full-size L40S
qualification uses identical initial models and an equal synthetic64-sequence
masked batch, with candidate updates alternating its32-row halves. Each timing
block measures six updates, so both halves occur three times. Actual optimizer
group LR/WD/scale settings match baseline at steps1,553,554,555,20000. All
candidate parameters/gradients remain finite, with original96 Muon matrices
and170671168 parameters. Alternating-order full-step medians348.9950ms batch64
versus160.3201ms batch32 imply2.1769x updates/sec and1.0884x tokens/sec in this
fixture. Kernel/memory behavior can make shape scaling non-linear; actual
distributed one-hour throughput must confirm this result. No held-out inputs
used. BATCH_PREFLIGHT.json preserves exact timings/protocol. All36 source files
match accepted HEAD and69 frozen config/source/dependency/evaluator hashes
match kickoff (train.py matches accepted R10).

Candidate scope: only two r21_batch32 configs listed in METHOD.json. Next:
`uv run --frozen python .dev/program2/verify_batch32.py --require-complete`, then
`uv run --frozen python .dev/program2/collect.py --method r21_batch32
--incumbent r10_sqrtloss --description 'Halve per-rank microbatch from64 to32 (global batch128), retaining accumulation1 and the accepted R10 model, loss and per-step optimizer recipe.'`.
Require both complete frozen evaluations, final checkpoint/optimizer/data/lock
receipts and exact four-rank replay. Keep only if2.5943719316273928 minus
candidate mean exceeds candidate sampleSD. On discard verify snapshots and
remove ONLY the two R21 configs; source already matches incumbent. Preserve
all research/run artifacts and do not collect unlaunched prototypes.

R21 paired launch: suffix20260906T095440Z, seed42 onkn081 and seed43 onkn056.
Startup verification at2026-09-06T10:01:42Z passed through steps1740/1760 with
175/177 finite logged records, exact four-rank32-row sampler/LPT replay,
128 sequences per update, and the fixed peak LR after exactly554 warmup
steps. All eight L40S GPUs are active. Observed distributed step times
roughly180-190ms are higher than the synthetic160.3ms benchmark; final
measured throughput and frozen evaluation will determine the tradeoff.
STARTUP_VERIFICATION.json preserves the full receipt. Next scheduled
completion check: 2026-09-06T11:03:40Z (07:03:40 Toronto, September6).

R21 complete and discarded (2026-09-06): seed42/43 validation
2.601979225873947 / 2.593340892344713; mean2.59766005910933,
sampleSD0.006108224216672435, delta-0.0032881274819374084.
Steps19417/19376, tokens586.979363/586.050840M, train_loss
2.643323941230774/2.652765939235687, full20775-chainP@L
0.11042849378849146/0.10023499823668175. Both synchronized3600s
stops, final checkpoints, fixed validation32seq994targets, all contact
shards/probes, lock/environment/config/data receipts and exact four-rank
32-row sampler replay passed. About1.87x more updates but6.4% fewer
tokens than R10: synthetic throughput did not transfer to distributed
training. Smaller batch alone did not improve the frozen objective;
variance remains material. Two candidate configs removed after hash
verification; source unchanged. Incumbent remains R10bc06899.

R22 hypothesis: reduce FFN capacity to increase training tokens in one hour.
Primary literature: Cheng et al., Training Compute-Optimal Protein Language
Models (https://arxiv.org/html/2411.02142v1, section3), fits the protein MLM
model/data allocation frontier, distinct from CLM. Their experiments use
1e18--1e21 FLOPs, >=20K updates, and cosine decay; our roughly6.4e17 nominal
6ND, fixed-LR, mixed-corpus Muon run is outside that tested regime. The paper
motivates an empirical capacity/data tradeoff, not a numerical optimum for
our hardware or an assertion that smaller is always better. Hoffmann et al.
(https://arxiv.org/abs/2203.15556) independently motivates joint model/token
allocation in text CLM, which is not directly transferable to protein MLM.
Counterpoint: Geva et al. (https://arxiv.org/abs/2012.14913) interprets text
FFNs as key-value memories; narrowing can remove learned pattern capacity.
We measure whether extra updates/examples repay that cost, not assume it.

Review across axes: native Muon plus R10 sqrt-mask-count loss remain accepted.
Alternative optimizers/polar updates, stronger Muon scaling, GEGLU, Q/K norm,
head gating and token dropout have not cleared the two-seed rule. Data-side
mask count, masking density, short-context curriculum and80/10/10 corruption
have not improved selection; corpus/mixture are frozen. Global token-mean
loss narrowly failed before R10; new focal or amino-acid soft targets risk
miscalibration against exact held-out NLL. Implementation-only final-layer
subset R20 bought1.6% tokens but failed loss selection; R21 smaller batch
bought updates at6.4% token cost and also failed. A larger capacity/compute
tradeoff remains untested. Considered lowering depth, but chose FFN width
so the full24-layer/12-head contact feature layout remains identical.

R22: every SwiGLU hidden dimension2048->1536, total142359616 parameters
(-28311552, -16.59%). Keep residual width768,24layers,12heads, LayerNorm,
RoPE10000, std.02 default initialization and residual divisor sqrt(24/36).
Keep microbatch64/rank, global256, accumulation1, context512,15% mask-all,
accepted LPT and sqrt-count loss, native96-matrix Muon plus AdamW, all LR/WD/
moment settings, exact554-step warmup and fixed peaks, synchronized3600s.
Resized tensors consume a different CPU initialization RNG stream, so shared
weights are not claimed bitwise equal between architectures. Native Muon's
shape-dependent adjustment follows its existing rule. More steps/hour also
means more per-step decay applications; do not bundle compensations. Exact
R10-shaped64-row sampler replay is required, preserving input/mask laws and
frozen data. The per-run seed is reset from scratch, with no weight transfer.

Implementation: optional ESMCConfig.ffn_hidden_dim (None preserves legacy),
positive-integer validation, identical FFN calculation in construction and
parameter accounting, one train.py option. Frozen load_checkpoint reconstructs
it without evaluator changes. Candidate scope is exactly model.py, train.py,
test_ffn_width.py and two configs, all archived under candidates/r22_ffn1536.
22 existing/new tests passed, including checkpoint logits/attention identity,
exact full-size meta-tensor parameter count and legacy defaults; Ruff passed.
All68 noncandidate frozen source/config/dependency/evaluator hashes verified.
Full-size L40S synthetic forward/backward/clip/optimizer and timing qualification
will establish feasibility before using the paired hour budget.

Next after completion: `uv run --frozen python .dev/program2/verify_ffn1536.py
--require-complete` then `uv run --frozen python .dev/program2/collect.py
--method r22_ffn1536 --incumbent r10_sqrtloss --description 'Narrow all24 SwiGLU FFN hidden dimensions2048->1536 (142359616 parameters), retaining the R10 attention architecture, batch256, loss and per-step optimizer recipe.'`.
Require complete frozen evaluations, checkpoint/data/env/lock/optimizer receipts
and replay, then keep iff2.5943719316273928 minus candidate mean exceeds its
sampleSD. On discard restore only model.py/train.py from bc06899 and remove
only the width test and two R22 configs after snapshot hash checks. Preserve
all artifacts; do not collect unlaunched prototypes.

R22 L40S qualification passed: baseline348.749632ms vs narrowed319.683481ms
per full synthetic update, +9.092% tokens/sec, measured in two alternating
rounds each3warm+6measured at equal batch64 varying-length <=512 inputs.
All142359616 parameters/gradients finite, all24 FFN pairs receive nonzero
gradients, and optimizer retains96 Muon matrices with new3072x768 and
768x1536 FFN shapes. Full-size contact output remains24 layers x12heads.
Tiny dense-vs-packed BF16 maxlogiterror0.005859375, relativegradienterror
0.00635003888; disabled-option tiny tensors/RNG/logits/attentions match
immutable accepted bc06899 source bitwise. Peak memory with both models
19673.46MiB. Native actual optimizer groups agree with R10 at schedule
steps1,553,554,555,20000. Synthetic cost is only a feasibility estimate;
R21 showed that distributed throughput can differ. MODEL_PREFLIGHT.json
preserves protocol and results. No held-out or training data used for
qualification. Launch both from scratch at the full3600s budget.

R22 paired launch suffix20260906T111309Z. At2026-09-06T11:15:40Z, seed42/kn081 and
seed43/kn056 startup verification passed steps210/220 (22/23 logged
records), finite losses/objectives/gradients, exact four-rank sampler and
LPT assignment/cumulative counters, and expected linear554-step warmup.
All eight L40S GPUs active; startup receipt saved. Next useful check is
2026-09-06T12:22:09Z (08:22:09 Toronto, September6), after one-hour training
and immediate frozen validation plus full contact evaluation.

R22 complete and KEPT (2026-09-06): validation seed42/43
2.5900170989334583 /2.59188773855567, mean2.590952418744564,
sampleSD0.0013227419620219622; delta0.0034195128828287125 exceeds SD.
New incumbent r22_ffn1536 commit19e9a97309ae2cc83f0e115280fbb5f03fbcf5ab, same seeds42/43.
Steps11178/11118; tokens675.938558/672.626676M; train_loss
2.6430711913108826/2.645797984600067; full20775-chainP@L
0.09627286725924138/0.10031101023208855. Actual mean throughput
+7.64% tokens versus R10. P@L decreased as a diagnostic and does not
affect selection. Both synchronized3600s stops, all checkpoint/optimizer
shapes/config/data/environment/lock receipts, frozen32seq994target validation,
all full contact shards/probes and four-rank64-row sampler replay passed.
Committed exactly the five candidate files; original uncommitted-run
results remain NA and DECISION.json binds them to accepted commit.

R23 hypothesis: improve output-logit conditioning with an auxiliary z-loss.
Primary PaLM (https://arxiv.org/html/2204.02311v5, section5) adds1e-4*log(Z)^2
to language-model NLL to keep log-normalizers near zero, reporting improved
training stability. T5X authors' implementation explicitly links this penalty
to limiting bfloat16 logit drift and encouraging normalized log-probabilities
(https://github.com/google-research/t5x/blob/main/t5x/losses.py); archived source
and Apache2 license hashes in METHOD.json. Our native PyTorch formula and
DDP reduction are independently implemented, not a JAX dependency/import.
Unlike PaLM's token-mean CLM, R23 applies the same penalty only to existing
masked targets and uses the accepted global sqrt-count sequence weighting.
No label smoothing, probability target changes or validation penalty.
Current runs are finite, with no established logit-instability failure;
this is a conditioning hypothesis rather than a claimed repair. The penalty
controls a common logit shift that pure cross-entropy does not identify;
it can nevertheless change parameter optimization and likelihood quality.

Review all axes: R22 narrow FFNs just improved mean beyond its own seedSD,
so keep that142.36M architecture. Further capacity tradeoffs remain possible;
prior GEGLU/head gates/extra norms/init/routing did not pass. Data alternatives
include span masking: SpanBERT motivated span reconstruction plus an auxiliary
boundary objective, but protein AMPLIFY authors explicitly report small-scale
span masking, label smoothing and class weighting did not look promising
(https://www.biorxiv.org/content/10.1101/2024.09.23.614603v2.full). Our15%-density,
80/10/10, fixed-mask-count and short-context trials also argue for keeping
corruption fixed while testing loss conditioning. Alternative Muon algorithms,
scaling, split projections and small batches have not cleared selection.
R10 sqrt-count normalization remains useful; R23 preserves it and adds a
separate small normalizer penalty instead of changing target weights again.

R23 uses accepted R22 from scratch, same seeds42/43, batch64/rank/global256,
24layers/12heads/FFN1536/142359616params, native96-matrix Muon plus AdamW,
constant peaks after exact554-step warmup, frozen mixture/corpus/tokenizer/
15% masks/evaluators and synchronized3600s. z_i = mean over selected targets
of logsumexp(all64 logits)^2; Zloss=sum_i sqrt(m_i)*z_i /sum_i sqrt(m_i)
globally over four ranks. Add1e-4*Zloss only to backward loss. Compute logsumexp
in fp32 (fp64 retained for oracle tests); ignore unselected positions and
empty sequences. Auxiliary numerator uses worldsize/globaldenominator to
cancel DDP averaging; an extra2scalar all-reduce obtains its metric/normalizer.
No model or sampler changes and no helper RNG consumption.

Metrics: loss remains rank0 sequence-mean MLM NLL; objective_loss remains
global sqrt-count MLM; logit_z_loss is unscaled global auxiliary, and
regularized_objective_loss=objective_loss+1e-4*logit_z_loss. Training summary
still means the final100 loss entries. The frozen evaluator and selection
use original validation NLL. Default coefficient0 performs no auxiliary
work. Nonzero coefficient requires sqrt_mask_count; invalid negative/nonfinite
coefficients rejected. Only train.py, test_logit_z_loss.py and two configs
are candidates; model.py matches new accepted19e9a97 byte-for-byte.

26 tests passed, including closed-form weighted-logit gradients, ignored-token
zeros, extremeBF16/empty target stability, shift invariance of original MLM,
RNG preservation and actual two-rank combined-objective DDP gradients with
unequal/empty rank targets. Required Ruff passed. All68 kickoff frozen
source/config/dependency/evaluator hashes plus accepted model verified.
Full-size synthetic L40S analytic gradients and paired full-step timing
qualification must pass before launching; no held-out data used.

Next: `uv run --frozen python .dev/program2/verify_zloss.py --require-complete`
then `uv run --frozen python .dev/program2/collect.py --method r23_zloss
--incumbent r22_ffn1536 --description 'Add1e-4 masked logit z-loss with global sqrt-mask-count sequence weighting to accepted R22; report original MLM loss separately.'`.
After full frozen receipt/replay checks, keep only if2.590952418744564 minus
candidate mean exceeds candidate sampleSD. On discard restore ONLY train.py
from19e9a97309ae2cc83f0e115280fbb5f03fbcf5ab and remove only the z-loss test
and two R23 configs after verifying snapshots. Retain accepted FFN model/
width test/configs and all run/research artifacts.

R23 full-size L40S qualification passed: all142359616 parameters/gradients
finite, native96 Muon matrices, identical initial weights/RNG/logits to
accepted R22; analytic fp32 gradient matches independent max-shifted
exp/log reference with zero ignored-position gradients and unchanged CUDA
mask RNG. Initial unscaledZloss18.56015205, contribution0.0018560152,
meanlogZ4.30812883 and unscaled gradient sum8.61625767 match2*meanlogZ.
Full-step median319.202182ms baseline vs319.375113ms z-loss (+0.054%,
within timing noise); paired distributed training will include the extra
all-reduce. After18 repeated synthetic updates, Zloss35.09568 baseline
and35.18349 candidate: no short-run reduction observed, and these inputs
are not validation evidence. Both remain finite; the small regularizer's
usefulness is decided by the full paired frozen evaluation. Peak memory
with two models17802.21MiB. LOSS_PREFLIGHT.json preserves exact protocol.

R23 launch suffix20260906T123228Z; startup at2026-09-06T12:34:53Z verified seed42/kn081
andseed43/kn056 through steps200/210 with21/22 logged records. Exact
four-rank64-row sampler/LPT/cumulative counters, linear554-step warmup,
finite MLM/z-loss/regularized objectives/gradients and exact metric identity
passed. Initial unscaledZloss19.0367851/18.6989918, latest43.1710358/43.6291313
(contributions0.0043171/0.0043629); no short-run normalizer reduction claimed.
All eight L40S GPUs active. STARTUP_VERIFICATION.json saved. Next useful
completion check2026-09-06T13:41:28Z (09:41:28 Toronto September6), following
one-hour training and immediate frozen validation plus full contact eval.

R23 complete and discarded (2026-09-06): seed42/43 validation
2.5989885218441486/2.600840497761965, mean2.5999145098030567,
sampleSD0.0013095447300819885, delta-0.00896209105849266.
Steps11175/11190, tokens675.752976/676.976299M, train_loss
2.6435548639297486/2.645685837268829, full20775-chainP@L
0.09909528677343807/0.09747377421593073. Both synchronized3600s
stops, exact data/LPT/LR replay, all MLM/z-loss identity records and
checkpoint/optimizer/config/data/env/lock/frozen validation/contact receipts
passed. UnscaledZloss first19.0368/18.6990 to last14.4351/16.2933, final100
means14.5412/16.5209; decreased within this run but no matched incumbent
Zloss series exists, so do not infer the penalty improved normalizers
relative to baseline. The regularization did not improve held-out NLL.
Restored only train.py from19e9a97 and removed only z-loss test+two configs
after snapshot hash checks. Accepted R22 model/width test/configs retained.

R24 hypothesis: probe the capacity/token tradeoff beyond accepted R22.
Direct evidence: R22 FFN2048->1536 saved16.6% parameters, processed7.64%
more tokens and passed the two-seed mean-improvement rule. R24 narrows
FFNs1536->1024, removing another28311552 parameters (19.89%) to114048064,
while holding24layers,12heads and768residual width fixed. This tests whether
extra training tokens outweigh the stronger capacity reduction, not an
assumption that smaller always wins. It is an architecture-size experiment
implemented solely by the existing accepted width option in two configs.

Primary literature: Cheng et al. (https://arxiv.org/html/2411.02142v1, section3)
profiles protein MLM model/data allocation under fixed FLOPs; its cosine-decay,
>=20K-step and >=1e18-FLOP regime does not numerically determine our fixed-LR
one-hour optimum. Liao et al. (https://arxiv.org/abs/2602.06471) revisits FFN
shape and attention/FFN capacity balance using deeper hourglass sub-MLPs;
its savings can be reallocated to wider model representations. That motivates
questioning conventional FFN expansion, but R24 implements neither hourglass
blocks nor attention widening. Mudarisov et al. (https://arxiv.org/abs/2608.02064)
reports benefits from geometry-guided layerwise width allocation at fixed
budgets; considered as a distinct future direction, not evidence that uniform
narrowing is optimal. R24's uniform reduction establishes a simple empirical
capacity boundary before introducing a learned/measured width profile.

All-axis review: loss-side R23 z-loss now failed both seed losses despite
finite training and within-run normalizer decrease; keep accepted sqrt-count
MLM. New label smoothing/class weighting risks changing NLL calibration and
AMPLIFY reported weak small-scale results. Data-side random replacements,
mask density/count and context curriculum already failed; span masking adds
prediction difficulty and lacks favorable direct evidence here, while corpus
and mixture are frozen. Algorithm alternatives (Muon scaling, AdaMuon,
Polar Express, splitQKV) and small-batch update frequency have not cleared
selection. Model-side width reduction has the strongest new local evidence;
other activation/norm/gating/residual modifications previously failed. R24
follows this model evidence while explicitly retaining algorithm/data/loss.

Exact scope: ffn_hidden_dim1024 plus derived parameter count114048064 in only
two seed configs. All37 implementation/test files match accepted19e9a97;
all68 kickoff frozen source/config/lock/evaluator hashes verified separately.
Batch64/rank/global256, context512, LPT,15% mask-all, sqrt-count loss,
96-matrix native Muon+AdamW, LR/WD/moments/defaultstd.02 initialization,
554-step global LR warmup then fixed peaks and synchronized3600s unchanged.
Matrix shapes now gate_up2048x768 and down768x1024 in24 blocks. Native Muon
shape adjustment follows the existing rule. Shape changes alter CPU RNG
consumption during initialization, so no bitwise common-tensor claim or
pretrained weight transfer. More updates/hour inherently means more WD
applications; no bundled compensation. Data replay uses the same64-row
sampler stream as R22; frozen contact feature channels remain24x12.

No new persistent unit tests for this configuration-only change. Existing
22 tests cover option validation, budgets, checkpoint compatibility, loss and
DDP, while full-size GPU qualification checks actual114048064 parameter count,
finite gradients/all24 FFNs, checkpoint loading with1024 and original attention
layout, plus dense/packed numerical agreement and measured full-step cost.
No training or held-out data is used in the qualification.

After completion: `uv run --frozen python .dev/program2/verify_ffn1024.py
--require-complete` then `uv run --frozen python .dev/program2/collect.py
--method r24_ffn1024 --incumbent r22_ffn1536 --description 'Narrow all24 SwiGLU FFN hidden dimensions1536->1024 (114048064 parameters), retaining the accepted R22 attention architecture, batch256, sqrt-count MLM and optimizer recipe.'`.
Require both complete frozen evaluations, full checkpoint/optimizer/data/env/
lock/feature/probe receipts and exact four-rank replay. Keep only if
2.590952418744564 minus candidate mean exceeds candidate sampleSD. On discard
verify archived config hashes and remove ONLY the two R24 configs; source
already matches accepted19e9a97. Preserve all artifacts and incumbent files.

R24 L40S qualification passed: measured full-step319.245231ms FFN1536
versus285.568090ms FFN1024, +11.793% synthetic tokens/sec in two
alternating rounds (each3warm+6measured updates, identical64 synthetic
masked sequences <=512). All114048064 parameters/gradients finite and
all24 FFN gate/up+down pairs have nonzero gradients;96 native Muon matrices,
original24x12 attention features. Tiny model with FFN1024 roundtrips through
frozen load_checkpoint with exact parameter tensors; dense/packed BF16
maxlogiterror0.0078125, relativegradienterror0.00969686325. Actual optimizer
group LR/WD/scales match R22 at1,553,554,555,20000 using the unchanged
shape-adjustment rule. Peak memory with both models17492.92MiB. Existing22
tests passed, all37 sources match accepted19e9a97, and required Ruff passed.
MODEL_PREFLIGHT.json preserves the qualification; actual paired throughput
and frozen NLL determine whether the capacity tradeoff is favorable.

R24 paired launch suffix20260906T134921Z; startup at2026-09-06T13:52:26Z verified
seed42/kn081 andseed43/kn056 through step220 each (23 logged records).
Exact four-rank64-row sampler/LPT/cumulative counters, expected554-step
linear warmup, finite losses/objectives/gradients and all eight active
L40S GPUs confirmed. STARTUP_VERIFICATION.json saved. Next useful check
2026-09-06T14:58:21Z (10:58:21 Toronto September6) after the synchronized
one-hour training budget and immediate frozen validation/full contact eval.


## R24 complete and discarded, 2026-09-06
Both full-hour runs and frozen validation/full20775-chain contact evaluations
passed all receipt checks and complete four-rank data replay. Seed42/43 losses
2.5950704645365477 /2.6011903136968613, mean2.5981303891167045,
sampleSD0.00432738684109655, delta-0.0071779703721404076 versus R22.
Steps12279/12226, tokens742.498977/739.623456M: about9.9% more tokens than
R22, but worse frozen NLL. Train-final100 means2.646404857635498/
2.6433655643463134 and diagnostic P@L0.09951391298248805/0.10551984077584066.
Uniform capacity reduction has reached an unfavorable tradeoff at hidden1024.
Only two archived hash-verified R24 configs removed; source stays19e9a97.
R22 remains incumbent mean2.590952418744564. All artifacts retained.

## R25: training-profile-guided FFN width allocation
R22 remains the accepted base19e9a97 (mean2.590952418744564). R24 shows that
more throughput alone does not justify uniform hidden1024: it loses capacity
needed under this hour budget. Considered all axes before this round:
- Algorithm: R13/R15 alternative polar/adaptive Muon and R06 splitQKV failed;
  grouped Muon preflight had only1.0175x speedup. Keep accepted optimizer/LPT.
- Data: R07 fixed mask counts, R09 short-context curriculum, R11 mask20,
  R18 input token-dropout and R19 mixed replacements gave no qualifying win.
  Preserve corpus/mixture and current15% masking; use train-only diagnostics.
- Loss: sqrt-count weighting remains accepted; R23 z-loss hurt NLL despite
  within-run log-normalizer reduction. No auxiliary objective this round.
- Model: redistribute existing FFN capacity rather than uniformly shrinking
  again; fixed24x12 attention features and142359616 parameters isolate allocation.

Primary literature consulted:
[Tapered Language Models](https://arxiv.org/html/2606.23670v1), June2026,
tests early-wide/late-narrow FFNs under a fixed total width budget in text
language models. Its setting differs from bidirectional protein MLM. We
initially proposed first12 widths2048,last12 widths1024, then checked whether
our learned representations supported that taper before constructing it.
[Geometry-Guided Layerwise FFN Width Allocation in Transformers](https://arxiv.org/html/2608.02064v1), August2026,
uses forward geometry to allocate a fixed FFN budget. Its spherical angular
shift is a simple work proxy; equal-exponent allocation normalizes a power
of layer work, followed by hardware-aligned rounding. We adapt that idea,
without intrinsic-dimension estimation, GW/topology, or causal sensitivity.
No quantitative gain from the text studies is assumed here.

Own measurement: FFN_DEPTH_PROFILE_20260906.json and profile_ffn_depth.py use
both accepted R22 checkpoints, same64 training sequences (4x16,ctx512),
sampler seed20260906, independent mask generators250000..250003. Exactly15495
canonical tokens per checkpoint; BOS/EOS/padding/noncanonical excluded. The
hook compares normalized z and z+FFN(z)/sqrt(24/36) at each post-attention
residual. It does not implement full LayerNorm centering/gains. No held-out
examples or labels are used. Checkpoint hashes and batch digests retained.
First-layer angular shifts0.52460/0.49248 dominate, but late layers exceed
most early layers2-12 in both seeds. Thus monotone early-half taper is NOT
supported and was never launched or entered into the results table.

Fixed selection rule chosen before calculating rounded widths: average work
across seeds42/43, take square roots (exponent0.5), normalize to total36864;
floor to multiples256 then distribute remaining units to largest fractional
remainders, layer index breaks ties. Minimum1024 is inactive. No exponent
sweep. WIDTH_ALLOCATION.json stores all means, ideals, provenance and result:
[3072] + [1280]*6 + [1536]*17. This transfers1536 hidden units from layers2-7
to layer1. A64-example proxy may be noisy; large forward change need not
mean useful/causal capacity demand, and roles may shift on retraining.
The whole-hour paired frozen evaluation decides whether this idea works.

Implementation: optional validated immutable ffn_hidden_dims selects each
block's width, forbids simultaneous scalar+list, preserves legacy defaults,
and is recorded in model_config for the unchanged frozen loader. Exact count
sums all gate/up/down sizes. Total142359616 params, original768 residual,
24 layers,12 heads, all norm/attention/init/residual formulas preserved.
Native Muon96 matrices and existing shape-adjustment rule remain; matrix
shapes and corresponding scaling differ where FFN width changes. Target
models initialize from scratch; no checkpoint weights are transferred and
no cross-architecture bitwise common-tensor assumption. Batch256,ctx512,
554-step global warmup then fixed group peaks,3600s synchronized wallguard,
original rank0 logged MLM loss and accepted global sqrt-count objective stay.

25 CPU tests passed including exact parameter shapes/counts, malformed-list
rejection, scalar/uniform-list RNG+tensor equivalence, gradients and frozen
loader outputs/attention features. Separate immutable19e9a97 source comparison
verified bitwise tiny CPU RNG/weights/logits/attentions for both legacy default
and scalar-width configurations. Required whole-worktree Ruff passed.

R25 full-size L40S qualification passed: baseline319.587051ms versus
candidate319.089728ms (+0.156% synthetic throughput, effectively equal),
actual142359616 params, finite all parameter gradients, all24 FFNs active,
96 native Muon matrices,24x12 full attention features. Tiny variable-width
frozen checkpoint roundtrip exact; packed/dense maxlogiterror0.0078125,
relativegradienterror0.006643018964945442. GroupLR/WD/scales agree at
steps1,553,554,555,20000. Peak memory with two full models17809.41MiB.
68 frozen source/config/evaluation/dependency hashes and35 unaffected
accepted-source files verified before launch; five candidate files archived.

After completion: `uv run --frozen python .dev/program2/verify_ffnprofile.py
--require-complete` then `uv run --frozen python .dev/program2/collect.py
--method r25_ffnprofile --incumbent r22_ffn1536 --description 'Redistribute the same total FFN width using a training-only angular-work profile: layer1=3072, layers2-7=1280, layers8-24=1536; retain142359616 parameters and accepted R22 attention, batch256, sqrt-count MLM and optimizer recipe.'`.
Require complete frozen evaluation and checkpoint/config/data/environment/
lock/probe receipts, all-rank sampler replay and every layer's saved FFN shape.
Keep only if2.590952418744564 minus mean exceeds candidate sampleSD.
On discard verify archived source/config hashes; restore ONLY model.py/train.py
from19e9a97, remove ONLY test_ffn_profile.py and two R25 configs. Preserve all
R22 files and all candidate artifacts. No commits until a candidate qualifies.

R25 paired launch suffix20260906T151455Z; startup at2026-09-06T15:17:26Z verified seeds42/kn081
and43/kn056 through step170 each,18 logged records. Finite losses/objectives/
gradients, exact four-rank64-row sampler/LPT/cumulative counters and554-step
warmup confirmed. All eight L40S GPUs active. STARTUP_VERIFICATION.json saved.
Next useful check2026-09-06T16:23:55Z (12:23:55 Toronto September6), allowing
one-hour training and immediate frozen validation/full-contact evaluation.


## R25 complete and discarded, 2026-09-06
Both full-hour runs and frozen validation/full20775-chain contact evaluations
passed receipts, all-rank replay and per-layer FFN/optimizer shape checks.
Seed42/43 NLL2.595098178833723/2.59324898943305; mean2.5941735841333866,
sampleSD0.001307574364914104, delta-0.0032211653888225555 versus R22.
Steps11240/11199; tokens679.697388/677.516190M. Train-final100 means
2.6437209129333494/2.644370632171631; diagnostic P@L0.10638096269821629/
0.10379307662886174. Similar throughput but neither seed improves its paired
R22 NLL. The measured forward angular-work proxy did not identify a useful
allocation under this short training regime. It is not a causal capacity test.
Five candidate files hash-verified and restored/removed only as declared;
all artifacts retained. R22 stays incumbent19e9a97 mean2.590952418744564.

## R26: fixed first-layer value residual in attention
All axes reviewed after R25 failed. Algorithm: accepted native Muon/LPT
remains preferable to tested splitQKV, alternative polar/adaptive updates
and the negligible grouped-Muon preflight gain. Data/loss: considered broad
random masking levels inspired by protein diffusion. The DSM primary study
(https://arxiv.org/html/2506.08293v1, June2025, section4.2) reports worse
reconstruction at5-15% masking than ESM2 despite stronger high-corruption
performance. This weakens its fit to our frozen15% evaluator, especially
with R11 mask20 not qualifying and R19 mixed replacements clearly worse.
Keep accepted15% masking and sqrt-count objective; R23 z-loss also failed.
Model: R24 uniform shrinkage and R25 forward-work capacity allocation failed.
Test a different information path at unchanged model size and matrix shapes.

Primary basis: Zhou et al., Value Residual Learning, ACL2025,
https://aclanthology.org/2025.acl-long.1375.pdf, sections3.2/4.2.
Eq5 mixes the first layer's values with each layer's current values before
attention. Identity-ResFormer fixes both coefficients at0.5, while other
variants learn or vary them. Its text-language experiments use different
data, AdamW, context lengths and a decaying LR schedule; their gains do not
predict a quantitative improvement for protein MLM with Muon and fixed LR.
Our bounded hypothesis is that direct access to initial residue information
through attention values can help later layers predict masked residues.
This is distinct from R08 adding/rescaling embeddings in the residual stream:
current Q/K construct the attention distribution, which aggregates mixed V.

Candidate r26_valueres uses V'_l=0.5*(V_l+V_1) for layers2-24; layer1 unchanged.
Original pre-attention LayerNorm computes V1. It is not detached or persisted
between batches. Cache and reuse the first fused QKV projection once per
forward; no extra projection, weights, optimizer states or random draws.
Each layer retains its own Q/K and current V projection. Both dense evaluation
and packed training use the same flat-width value mixture before head reshape.
Checkpoint recomputation receives values/projection explicitly, preserving
shared autograd. The first value projection receives gradients from later
layers, while every current V also retains its gradient through coefficient0.5.
No claim of bitwise initial outputs except first layer; weights and RNG are
bitwise unchanged. Actual attention probabilities still produce all24x12
frozen contact feature channels. No evaluator/probe changes.

R22 base19e9a97, uniform FFN1536,142359616 parameters, batch256,ctx512,
native96-matrix Muon plus AdamW,554 global warmup steps and fixed group peaks,
3600s wallguard, data/mixture/sampler/masks/tokenizer/validation unchanged.
Only five candidate files: model.py,train.py,test_value_residual.py and two
seed configs. Same seeds42/43 and nodeskn081/kn056; train from scratch.

26 CPU tests passed. New tests compare explicit attention-equation outputs,
all attention features and gradients; detaching only the reference path
changes the first V gradient as expected. QKV hooks verify exactly one
projection per layer and no state leakage between forwards. Checkpointed
and ordinary CPU gradients/logits agree exactly, frozen checkpoint loader
preserves new option and features, all initial weights/RNG match, and a
one-layer model is unchanged. Meta count exactly142359616. Separate immutable
19e9a97 legacy comparison passed tensors/RNG/outputs/features/all gradients
for both ordinary and checkpointed paths. Required whole-worktree Ruff passed.

R26 full-size L40S qualification passed: baseline319.041497ms versus
candidate328.488311ms (+2.961% step time, -2.876% synthetic throughput).
This bounded cost is allowed; actual same-hour NLL must justify it. All
142359616 parameters/gradients finite and all24 FFN pairs active,96 native
Muon matrices, original24x12 attention features. Initial fullmodel tensors
match R22 bitwise. Tiny dense/packed maxlogiterror0.005859375,
relativegradienterror0.007123398275540764. Packed checkpointed versus
ordinary packed logits and all gradients are bitwise equal. Frozen loader
roundtrip preserves value_residual and exact tiny tensors. Actual optimizer
group LR/WD/scales match at1,553,554,555,20000. Peak memory with two models
17793.58MiB. 68 frozen source/config/evaluation/lock hashes and35 unchanged
accepted-source files verified; all five candidate files archived/hash-bound.

After completion: `uv run --frozen python .dev/program2/verify_valueres.py
--require-complete` then `uv run --frozen python .dev/program2/collect.py
--method r26_valueres --incumbent r22_ffn1536 --description 'Identity value residual: mix first-layer and current attention values equally in layers2-24 with intact cross-layer gradients and cached first QKV; retain accepted R22 shapes,142359616 parameters, batch256, sqrt-count MLM and optimizer recipe.'`.
Require completed frozen evaluations and all checkpoint/config/data/env/lock/
probe receipts, four-rank training replay, saved option and native optimizer
state shapes. Keep only if2.590952418744564 minus mean exceeds candidate SD.
On discard hash-verify candidate source/configs, restore ONLY model.py/train.py
from19e9a97, remove ONLY test_value_residual.py and two R26 configs. Preserve
accepted R22 files and all archived artifacts. No commit before a qualifying win.

R26 paired launch suffix20260906T163327Z; startup at2026-09-06T16:36:14Z verified seeds42/kn081
and43/kn056 through step190 each,20 logged records. Finite losses/objectives/
gradients, exact four-rank64-row sampler/LPT/cumulative counters and554-step
warmup confirmed. All eight L40S GPUs active. STARTUP_VERIFICATION.json saved.
Next useful check2026-09-06T17:42:27Z (13:42:27 Toronto September6), allowing
one-hour training and immediate frozen validation/full-contact evaluation.


## R26 complete and discarded, 2026-09-06
Both full-hour runs and frozen validation/full20775-chain contact evaluations
passed receipts, complete four-rank data replay, saved value_residual option
and native optimizer state checks. Seed42/43 NLL2.5971374958753586/
2.6035686023533344; mean2.6003530491143465, sampleSD0.004547479001109455,
delta-0.009400630369782448 versus R22. Steps10917/10862; tokens660.137984/
657.152066M, roughly2.3% fewer than R22. Train-final100 means2.6403492975234983/
2.642250292301178; diagnostic P@L0.11181480359646992/0.11107025250545471.
The improved contact diagnostic and training NLL do not override worse frozen
validation MLM. Fixed equal value mixing failed this selection criterion.
Five candidate files hash-verified and restored/removed only as declared;
all artifacts retained. R22 stays incumbent19e9a97 mean2.590952418744564.

## R27: cautious weight decay on Muon matrices
Review after R26: architecture changes (R24/R25 FFN allocation and R26 value
residual) did not improve frozen MLM despite some contact-score gains. Keep
accepted R22 model. Data: R07 fixed mask counts, R09 short contexts, R11 mask20,
R18 token dropout, R19 mixed replacement and the DSM low-mask counterevidence
favor preserving current masking/context/mixture. Loss: sqrt-count remains
accepted; z-loss failed. Algorithm: revisit regularization within the accepted
Muon geometry, rather than changing its polar map (R13/R15 failed) or splitting
QKV (R06 incurred throughput loss). No scalar LR or WD coefficient sweep.

Primary sources consulted:
- Chen et al., Cautious Weight Decay, arXiv2510.12402v1 (October2025),
  https://arxiv.org/html/2510.12402v1, Algorithm1 and AppendixB.4.
  Decay is multiplied by indicator(weight * optimizer_direction >= 0),
  before subtracting the update. The paper tests AdamW/Lion/Muon in text
  pretraining and vision and reports gains without baseline retuning. Its
  theoretical stationary-point discussion is not a guarantee for our finite
  stochastic protein run. This round adapts its decay mask only to Muon;
  original EMA/Nesterov and shape-scaled update rules remain from locked Torch.
- Wang et al., Muon Outperforms Adam in Tail-End Associative Memory Learning,
  https://arxiv.org/abs/2509.26030. It identifies VO attention and FFN weights
  as major beneficiaries. This suggested selective optimizer assignment, but
  fused QKV entangles Q/K with V and R06 showed a cost to splitting projections.
  Defer that branch; preserve all96 Muon matrices for this single-change round.

Own implementation: CautiousMuon inherits native Muon state initialization,
validation, groups and serialization; reproduces its momentum-buffer lerp and
Nesterov lerp, calling frozen _zeropower_via_newtonschulz and _adjust_lr.
Use old FP32 weights and native BF16 orthogonalized direction for the mask,
including equality at zero as the paper specifies. Apply
p <- p - lr*wd*p*mask, then p <- p - adjusted_lr*direction. The decay LR is
native group LR, not the shape-adjusted update LR. No mask normalization,
no gradient masking or direction reconstruction from rounded parameter deltas.
The update direction and sole momentum state are unchanged for supplied grads.
CWD affects96 hidden matrices only; fused AdamW embedding/head/scalars stay
exactly as accepted. Same WD coefficient means less total decay, so a future
plain reduced-WD ablation would be needed to attribute any gain specifically
to sign selection. No extra persistent optimizer tensors. Preflight measures
the elementwise mask/update overhead under the same full-hour budget.

R22 base19e9a97,142359616 params, original24x12 features, batch256,ctx512,
15% masking and accepted sqrt-count objective unchanged. Exact554-step global
warmup followed by fixed peaks; synchronized3600s guard and all frozen data/
evaluators/tokenizer/dep.lock unchanged. Seeds42/43 onkn081/kn056 from scratch.
Candidate files: train.py optimizer-class selector, cautious_muon.py and its
retained BSD notice, test_cautious_muon.py, and two configs. Native Torch source
and hashes archived in TORCH_REFERENCE.json; private helpers are tied to our
frozen2.13.0+cu126 environment. No dependency changes or copied external model.

26 CPU tests passed. New cases cover native bitwise equivalence with WD0 for
square/tall/wide matrices, both Nesterov settings and a zero gradient step;
fixed direction fixtures test positive/opposed/zero products and pre-update
weight signs even when an update crosses zero. State resume, no-gradient
parameter skipping and closure behavior pass. Config routing selects CWD only
for Muon; supplied-gradient AdamW parameters and states remain bitwise equal.
Whole-worktree Ruff check and formatting passed.

R27 full L40S qualification passed. Production QKV2304x768, output768x768,
FFNgate3072x768 and down768x1536 tested with three supplied-gradient updates:
maximum FP64 cautious-reference error7.430070336766903e-9, native momentum
buffers exactly equal and zero-decay updates bitwise equal at every shape.
Synthetic independent weights/gradients activate decay on approximately50%
of coordinates; this is not a measurement of campaign training-mask density.
Full-model baseline319.172744ms versus candidate322.422060ms (+1.018%
step time, -1.008% synthetic throughput). All142359616 params/gradients finite,
24 FFN pairs active,96 Muon matrices, original24x12 contact-feature layout.
Tiny frozen loader roundtrip exact, dense/packed maxlogiterror0.00537109375,
relativegradienterror0.00624101404. GroupLR/WD/scales agree at
1,553,554,555,20000; decay support changes but configured coefficient does not.
Peak memory with two models17799.99MiB. 68 frozen source/config/eval/lock hashes
and36 unchanged accepted-source files verified. Six candidate files archived;
39 source files plus retained license are hash-bound in METHOD.json.

After completion: `uv run --frozen python .dev/program2/verify_cwd.py
--require-complete` then `uv run --frozen python .dev/program2/collect.py
--method r27_cwd --incumbent r22_ffn1536 --description 'Muon-only cautious weight decay: gate decay by pre-update weight times native orthogonalized update >=0; retain all R22 momentum/NS/shape scaling, AdamW groups,142359616-parameter model, batch256 and sqrt-count MLM.'`.
Require completed frozen evaluations and all checkpoint/config/data/env/lock/
probe receipts, four-rank replay, saved cautious flag and native momentum-only
state shapes. Keep only if2.590952418744564 minus mean exceeds candidate SD.
On discard hash-verify candidate sources/configs, restore ONLY train.py from
19e9a97, remove ONLY cautious_muon.py/its LICENSE, test_cautious_muon.py and
two R27 configs. Model.py already remains accepted R22. Preserve all artifacts.

R27 paired launch suffix20260906T175504Z; startup at2026-09-06T17:57:28Z verified seeds42/kn081
and43/kn056 through steps160/170,17/18 logged records. Finite losses/objectives/
gradients, exact four-rank64-row sampler/LPT/cumulative counters and554-step
warmup confirmed. All eight L40S GPUs active. STARTUP_VERIFICATION.json saved.
Next useful check2026-09-06T19:04:04Z (15:04:04 Toronto September6), allowing
one-hour training and immediate frozen validation/full-contact evaluation.


## R27 complete and discarded, 2026-09-06
Both full-hour runs and frozen validation/full20775-chain contact evaluations
passed all receipts, four-rank replay, saved cautious flags and native momentum
state checks. Seed42/43 losses2.5987412109971046/2.5994090661406517;
mean2.599075138568878, sampleSD0.0004722449008524399, delta-0.008122719824314117
versus R22. Steps11014/10959; tokens665.997433/663.023793M. Train-final100
means2.6438844442367553/2.646044704914093; P@L0.09249408040267994/
0.09830363228310703. Both seed losses worsen: Muon-only selective decay
fails this fixed-budget criterion. Only six declared candidate files restored/
removed after hash verification; all artifacts retained. R22 remains19e9a97,
mean2.590952418744564. No post-hoc decay coefficient adjustment is retained.

## R28 EMA of weights, 2026-09-06
All-axis review after R27: algorithmically, selective Muon decay worsened both
seeds; retain native Muon/AdamW and test averaging noisy constant-LR iterates.
The model branch has now rejected narrower FFN1024, fixed-budget depth allocation,
and value residuals; retain R22's uniform1536 FFNs and142359616 parameters.
Data/batching retain15% masks,ctx512,batch256 and deterministic LPT after earlier
masking/corruption/context/small-batch failures. DSM's low-mask counterevidence
also argues against transferring high-mask diffusion results directly here.
The accepted sqrt-count MLM loss remains; R23 z-loss did not improve selection.
This round changes the estimator used for final evaluation, with no new loss,
optimizer update, data sampling rule or model parameterization.

Primary inspiration: Meterez et al., Anytime Pretraining: Horizon-Free
Learning-Rate Schedules with Weight Averaging (2026),
https://arxiv.org/html/2602.03702v1. Their150M/300M text experiments pair weight
averaging with constant or inverse-square-root rates and compare with tuned
cosine schedules. This motivates averaging within our mandatory constant-rate
regime; it does not establish benefit for our protein objective or short budget.
Their substantially longer token horizons differ from ours. Morales-Brotons,
Vogels and Hendrikx, Exponential Moving Average of Weights in Deep Learning:
Dynamics and Benefits, https://arxiv.org/abs/2411.18704, provides complementary
vision evidence on EMA and reduced reliance on LR decay. Neither is evidence
that our chosen0.999 decay is optimal; choose it once before the paired run.

Own hypothesis: the final constant-rate iterate may contain optimization noise;
a trailing weight average may improve held-out MLM within this same hour. Lag
behind a still-improving online model may instead hurt in this undertrained
regime. Fix EMA decay0.999 (effective window about1000 updates, half-life693).
Initialize from the post-update online weights at step554, then perform
shadow <- shadow + .001*(online-shadow) after every optimizer update. All
floating state tensors averaged; integer persistent buffers copied if present.
There is no BatchNorm recalibration in ESMC. Rank0 owns the nontrainable shadow;
its arithmetic is before CUDA synchronization and the global time guard, so
all costs count against3600s. No updates after the final optimizer step.

Train from scratch, seeds42/kn081 and43/kn056, four L40S each. Exactly554-step
linear warmup, all configured peaks thereafter; no cooldown or online weight
interpolation. Final checkpoint model field contains the one predeclared EMA;
online_model retains original weights alongside their optimizer/RNG state and
explicit ownership metadata. Frozen evaluator reads model as before. Do not
choose between online and EMA using held-out results. Training loss diagnostic
remains final100 logged online sequence-mean losses. Same corpus/mixture/mask
RNG, trainable parameter count,24x12 contact features, evaluation/probes/lock.
Five candidate files listed in METHOD.json; all other accepted source unchanged.

26 CPU tests passed: closed-form EMA recurrence and exact start boundary,
integer buffers, shadow storage/no-grad/RNG preservation, full tiny online SGD
trajectory and optimizer equality, frozen evaluator checkpoint roundtrip with
averaged logits/attention features, default checkpoint compatibility and invalid
state/step rejection. Whole-worktree Ruff check and formatting passed.

The first full-size preflight incorrectly required bitwise equality between
independently executed GPU trajectories and failed; preserve its script/log.
An EMA-disabled control initialized identical full models on the same fixed
synthetic input and matched execution order. It also diverged: maxabs parameter
difference0.001538705 after9 updates and0.003127273 after18, L2 differences
0.377108838 and0.978508807. No unique kernel cause established. Direct full-size
snapshots around both EMA initialization and lerp updates showed every online
parameter, gradient, optimizer state and CPU/CUDA RNG bitwise unchanged, with
all shadow storage distinct. This directly tests isolation; it does not claim
bitwise independent GPU trajectories. EMA_ISOLATION_PREFLIGHT.json records it.

Full-size L40S qualification passed:319.582966ms baseline versus322.289328ms
with EMA (+0.846841% synthetic step cost, -0.839731% throughput). Shadow costs
569438464 bytes (~543.06MiB), no trainable parameters. All online parameters,
gradients and EMA tensors finite, all24 FFN pairs active,96 native Muon matrices,
24x12 contact features preserved. Frozen loader tiny roundtrip exact; dense vs
packed maxlogiterror0.00537109375 and relativegradienterror0.00624101404.
Actual LR groups agree at1,553,554,555,20000. Synthetic timing averages from
step1 to include steady-state cost; campaign starts at554. No held-out data
used for implementation/preflight.68 frozen source/config/eval/lock hashes and
36 unchanged accepted-source files verified;39 source hashes archived.

After completion: `uv run --frozen python .dev/program2/verify_ema.py
--require-complete` then `uv run --frozen python .dev/program2/collect.py
--method r28_ema --incumbent r22_ffn1536 --description 'Evaluate predeclared EMA weights: initialize after warmup step554, decay0.999 every later update within the hour; preserve online R22 model/Muon+AdamW, batch256 and sqrt-count MLM, retain online weights with optimizer state.'`.
Require final EMA metadata through last step, finite distinct EMA/online weights,
96 native Muon state shapes, complete four-rank data/LR replay and all frozen
checkpoint/evaluation/config/data/lock/probe receipts. Keep only if incumbent
mean2.590952418744564 minus paired mean exceeds candidate sample SD. On discard,
hash-verify exactly five METHOD.json candidate files, restore only train.py
from19e9a97 and remove only weight_average.py, test_weight_average.py and two
R28 configs. All accepted model sources remain intact; preserve all artifacts.

R28 paired launch suffix20260906T192318Z; startup verified at2026-09-06T19:28:03Z: seeds42/kn081
and43/kn056 both through step620,63 records each. Finite losses/objectives/
gradients, exact four-rank64-row sampler/LPT/cumulative counters and554-step
warmup followed by fixed peak verified. EMA last_step620,num_updates67 on
rank0 of each run; all eight L40S GPUs active. STARTUP_VERIFICATION.json saved.
Next useful check2026-09-06T20:32:18Z (16:32:18 Toronto September6), allowing one-hour
training plus immediate frozen validation and full-contact evaluation.

## R28 complete and discarded, 2026-09-06
Both full-hour runs and frozen evaluations passed all checkpoint, EMA metadata,
optimizer, config/data/lock/probe receipts and four-rank replay. Seed42/43 losses
2.5787397138774395/2.5909698754549026; mean2.584854794666171, sampleSD
0.008648030186431356, delta0.0060976240783929825 versus R22. Both paired seed
losses improve, but mean gain does not exceed candidate SD, so discard strictly.
Steps11060/11100, tokens668.780038/671.533810M; train-final100 means
2.643080554008484/2.6457286143302916; P@L0.10907910386529514/
0.10563076491962989. EMA/online L2 distances75.331043537/75.477240983,
10507/10547 averaging updates. No online held-out evaluation or checkpoint
selection performed. EMA remains a promising rejected branch; no post-hoc
decay sweep retained. All five declared files restored/removed after hashes
verified; artifacts retained. Incumbent R22 remains19e9a97,mean2.590952418744564.

## R29 tied input/output embeddings, 2026-09-06
All-axis review: R28 EMA improved the mean and both paired seed losses but
failed the predeclared gain>candidate-SD criterion. Retain it as a rejected
algorithmic branch, without an immediate decay sweep or selecting an online
checkpoint. R27 selective decay failed; native Muon/AdamW remains accepted.
For model design, FFN1536 remains supported over narrower or layer-reallocated
widths and value residuals. Explore parameter sharing between the vocabulary
input and output roles, leaving transformer computation intact. Data retain
15% masking,512 context, global256 batch and LPT after the earlier corruption,
context and small-batch failures; no corpus/mixture changes. Loss remains
accepted sqrt-count MLM after R23 z-loss failure. Sharing imposes a model
constraint that changes gradient flow without adding a loss or LR schedule.

Primary literature: Press and Wolf, Using the Output Embedding to Improve
Language Models, EACL2017, https://aclanthology.org/E17-2025/. Their language
model experiments and update analysis motivate coupling input and output
embedding learning; those text results do not establish protein gains here.
Protein precedent is direct in the official ESM-2 model:
https://github.com/facebookresearch/esm/blob/main/esm/model/esm2.py passes the
input embedding weight to its MLM head; the implementation in
https://github.com/facebookresearch/esm/blob/main/esm/modules.py uses that
shared matrix for the final vocabulary projection. This borrows the sharing
idea only; retain our ESMC transformer, head preprocessing and output bias.
No ESM-2 pretrained weights, masking rescale or other settings are imported.

Own hypothesis: joining dense output-classifier and input-lookup gradients may
regularize amino-acid representations and improve sample efficiency. Conversely,
sharing can restrict useful differences between the roles, and a64-token
vocabulary has little parameter redundancy compared with large text vocabularies.
Parameter saving is only49152 (~0.035% ofR22), not a material compute hypothesis.
R29 sets head_out.weight = embedding.weight after constructing/initializing all
original tensors. Thus common initial tensors and final RNG are exact; the
output matrix deliberately adopts the input draw. Separate output bias remains.
One shared trainable parameter receives the sum of both gradient paths, is
counted/clipped once, and receives one native AdamW decay/update; its state is
not duplicated. All96 transformer matrices remain native Muon. Same LR/WD,
554-step linear warmup and fixed peaks thereafter; no EMA or added state.

Trainable count142310464. Still24layers,768width,12heads,FFN1536 and24x12
contact features. Both seeds42/43 run from scratch onkn081/kn056 withfour L40S
each, full synchronized3600s guard and immediate frozen final-checkpoint eval.
Same train corpus/mixture/tokenizer, masked-target RNG and loss reductions,
validation/probes/features/evaluators and dependency lock. Exactly five changed
files: optional model alias/count/config, train.py option forwarding, new test,
and two configs. Frozen evaluator reconstructs aliasing from model_config;
state_dict retains both key names referencing the same stored tensor.

32 whole-suite CPU tests passed. New tests cover initialization/RNG/common
weights, default legacy config,142310464 exact meta-model count, shared
parameter after dtype conversion, full tiny chain rule (shared gradient equals
input+output untied gradients), unique optimizer placement, three exact native
AdamW updates/states without duplicate decay, and frozen checkpoint sharing,
logits and attention features. Immutable accepted R22 source compared with new
untied default: every initialized tensor,RNG,logits,attention and gradient
bitwise equal on CPU math model. LEGACY_PREFLIGHT.json records provenance.
Whole-worktree Ruff check and formatting passed.

Full-size L40S qualification passed. Baseline319.394687ms versus tied319.330784ms
(~0.02% throughput difference; effectively equivalent synthetic step cost).
All142310464 parameters/gradients finite,24 FFN pairs active,96 Muon matrices;
common initialized tensors bitwise equal. Shared embedding remains aliased on
CUDA and appears once in optimizer groups; exactly one64x768 AdamW state after
18 full updates, all AdamW states finite. Original24x12 attention features
preserved. Tiny frozen loader retains sharing; dense/packed maxlogiterror
0.00634765625, relativegradienterror0.00747474013. LR groups agree at
1,553,554,555,20000. No campaign or held-out data used for qualification.
68 frozen source/config/eval/lock hashes and35 unchanged accepted-source files
verified;38 source hashes and all five candidate files archived in METHOD.json.

After completion: `uv run --frozen python .dev/program2/verify_tied.py
--require-complete` then `uv run --frozen python .dev/program2/collect.py
--method r29_tied --incumbent r22_ffn1536 --description 'Tie input embedding and output vocabulary projection after unchanged initialization; one AdamW-owned64x768 parameter receives both gradients, separate output bias and R22 FFN1536/model/optimizer/batch256/sqrt-count MLM remain.'`.
Require actual saved alias storage,142310464 unique finite parameters, exactly
one64x768 AdamW state with no duplicate parameter IDs,96 native Muon states,
complete four-rank replay and frozen checkpoint/config/data/lock/evaluation/
probe receipts. Keep only if2.590952418744564 minus candidate mean exceeds
candidate sample SD. On discard hash-verify all five declared files, restore
ONLY model.py/train.py from19e9a97, remove ONLY test_tied_embeddings.py and two
R29 configs; preserve all artifacts and accepted FFN configs/test.

R29 paired launch suffix20260906T204223Z; startup at2026-09-06T20:44:37Z verified seeds42/kn081
and43/kn056 throughstep160 each,17 records each. Finite losses/objectives/
gradients, exact four-rank64-row sampler/LPT/cumulative counters and554-step
warmup confirmed. All eight L40S GPUs active. STARTUP_VERIFICATION.json saved.
Next useful check2026-09-06T21:51:23Z (17:51:23 Toronto September6), allowing the full
hour plus immediate frozen validation and full-contact evaluation.

## R29 complete and kept, 2026-09-06
Both full-hour runs and frozen validation/full20775-chain contact evaluations
passed all checkpoint/config/data/environment/lock/probe receipts, complete
four-rank replay and exact tied storage/unique native optimizer state checks.
Seed42/43 losses2.584416225552559/2.5767199900001287; mean2.580568107776344,
sampleSD0.005442060348732356, delta0.010384310968220234 versus R22: KEEP.
Steps11160/11129, tokens674.846645/673.284777M; train-final100 means
2.6410908794403074/2.645231456756592; P@L0.09017804768897014/
0.10035763870166096. Both seed losses improve. Only five declared candidate
files committed after source/config hash verification; new accepted commit
fc0a1253a8919615714fd7f95bbe1a7a00235172. Run rows retain NA for originally uncommitted
source; DECISION.json maps exact accepted commit. New incumbent is r29_tied,
mean2.580568107776344, seeds42/43; future selection and restoration must use
this base, retaining model tying and its tests/configs.

## R30 EMA on accepted tied embeddings, 2026-09-06
All-axis review: R29 weight tying improves both seeds and qualifies, establishing
new incumbent mean2.580568107776344 atfc0a1253a8919615714fd7f95bbe1a7a00235172.
Model sharing is therefore retained with24layers/12heads/FFN1536. Algorithmically,
R28 EMA also improved both seed losses on untied R22 but failed gain>candidate
SD; test whether its improvement transfers to the accepted tied model. This is
one averaging change relative to R29, retaining the exact R28 decay0.999 and
startstep554. No window sweep or assumption that the two gains are additive.
Data/batching retain15% masks,512 context,batch256,LPT and frozen mixture after
prior masking/corruption/context/small-batch failures. Loss stays accepted
sqrt-count MLM after z-loss failure; tying and averaging introduce no new loss.

Primary motivation remains Meterez et al., Anytime Pretraining: Horizon-Free
Learning-Rate Schedules with Weight Averaging,
https://arxiv.org/html/2602.03702v1. Their150M/300M text experiments compare
constant-rate averaging with cosine at substantially longer token horizons;
their EMA window varies over time, unlike our fixed0.999 rule. This motivates
an evaluation estimator and does not establish benefit for short protein runs.
Complementary EMA evidence from Morales-Brotons et al.,
https://arxiv.org/abs/2411.18704, is in a different vision setting. Our immediate
empirical motivation is the rejected R28 branch, not new decay tuning.

Hypothesis: trailing averaging reduces final-iterate noise beyond the effect
of sharing representations; it could instead lag the still-improving model or
interact poorly with shared gradients. Both seeds train from scratch. Preserve
all R29 initialization, native Muon/AdamW updates and gradients. Rank0 copies
post-update weights atstep554, then averages every subsequent online iterate
withdecay0.999; effective window~1000 updates, half-life~693. The final EMA is
predeclared for frozen evaluation, without an online/EMA held-out comparison.
Online training diagnostic remains the final100 logged sequence-mean losses.

Shared parameters require one EMA operation per unique tensor. state_dict with
keep_vars preserves exact parameter identity; canonical owners get one detached
clone and aliases reference that same shadow. Subsequent updates validate state
shapes/dtypes/devices and alias relationships, then lerp each unique floating
tensor once; copy nonfloating buffers if any. This preserves tied storage and
avoids applying EMA twice to the shared embedding. No BatchNorm recalibration.
All shadow allocation/arithmetic precedes synchronization/time guard and counts
against3600s. Final checkpoint model=EMA, online_model=online with optimizer/RNG
ownership recorded. Both saved models preserve shared input/output storage.
Trainable count remains142310464; shadow is nontrainable and costs569241856B.

Same554-step linear warmup then fixed peak in every group, samefour L40S/run,
seeds42/kn081 and43/kn056, full synchronized hour and immediate frozen final
validation/full-contact evaluations. Corpus/mixture/tokenizer/dependency lock,
validation/probes/features/seeds/reductions and all evaluator sources unchanged.
Only five candidate files: train.py timing/checkpoint changes, EMA module/test,
and two configs. Accepted tied model, tying test and R29 configs remain intact.

38 whole-suite CPU tests passed. Six EMA cases cover exact recurrence/start
boundary, integer buffers, exact tied online trajectory and optimizer-state
isolation, frozen evaluator checkpoint logits/features and default checkpoint
compatibility, invalid decay/step/state, one shared-weight average rather than
two and alias preservation across serialization, and rejection of alias splits
or merges. Whole-worktree Ruff check and formatting passed.

Full-size L40S qualification passed. At both initialization and first lerp,
snapshots of every online parameter, gradient, optimizer state and CPU/CUDA RNG
are bitwise unchanged; all shadow storage is distinct from online and the tied
embedding shadow remains shared. All142310464 parameters/gradients and unique
EMA tensors finite,24 FFN pairs active,96 Muon matrices,24x12 contact features.
Tiny frozen loader retains tying; dense/packed maxlogiterror0.00634765625,
relativegradienterror0.00747474013. Baseline318.843188ms versusEMA321.943725ms
(+0.97243% synthetic step cost, -0.96307% throughput). Isolation snapshots occur
only in warmup, excluded from measured timings; campaign EMA starts554 while
preflight starts1 to measure steady-state cost. No independent GPU bitwise
trajectory claim: R28 controls already established baseline variability without
identifying a unique kernel cause.68 frozen source/config/eval/lock hashes and
37 unchanged accepted-source files verified;40 source hashes archived.

After completion: `uv run --frozen python .dev/program2/verify_tiedema.py
--require-complete` then `uv run --frozen python .dev/program2/collect.py
--method r30_tiedema --incumbent r29_tied --description 'Add predeclared EMA to accepted tied embeddings: decay0.999 from post-update step554 through final step within the hour, preserve one shadow per shared parameter; evaluate EMA only and retain online weights with optimizer state.'`.
Require EMA metadata through the final step, both EMA/online tied storage,
142310464 unique finite model parameters, exactly one64x768 AdamW state and96
native Muon states, complete four-rank replay and all frozen evaluation/receipt
checks. NEW threshold:2.580568107776344 minus mean must exceed candidate SD.
On discard hash-verify five declared files, restore ONLY train.py from
fc0a1253a8919615714fd7f95bbe1a7a00235172, remove ONLY weight_average.py,
test_weight_average.py and two R30 configs. Preserve accepted R29 model.py,
test_tied_embeddings.py and R29 configs; preserve all artifacts.

R30 paired launch suffix20260906T220018Z; startup at2026-09-06T22:05:05Z verified seeds42/kn081
and43/kn056 throughstep620 each,63 records each. Finite losses/objectives/
gradients, exact four-rank64-row sampler/LPT/cumulative counters and554-step
warmup followed by fixed peaks verified. EMA last_step620,num_updates67 on
rank0 of each run; all eight L40S GPUs active. STARTUP_VERIFICATION.json saved.
Next useful check2026-09-06T23:09:18Z (19:09:18 Toronto September6), allowing the full
hour plus immediate frozen validation and full-contact evaluation.
