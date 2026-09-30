# Benchmarking Agentic AutoResearch Systems

NanoProteinLM benchmarks an agent's ability to discover better protein-model training recipes.

This page details the [AutoResearch protocol in the README](../README.md#autoresearch-protocol). A benchmark specifies the permitted design space, a fixed number of search rounds, the compute available in each round and the final evaluation budget. Publish these settings before search and use them for every method being compared. Each method chooses how to propose recipes, use previous results and select its final recipe within those limits.

```mermaid
flowchart LR
    B["Published design space<br/>R rounds × fixed compute per round"] --> A["AutoResearch method"]
    A --> C["Selected training recipe"]
    C --> E["Owner-run evaluation<br/>Fixed token budget and metrics"]
```

The protocol applies to any AutoResearch algorithm. Our implementation of Karpathy-style sequential search, including its pipeline, training-seed policy and improvement criteria, is described in [AUTORESEARCH_BASELINE.md](AUTORESEARCH_BASELINE.md).

## Preparation

Use the `autoresearch-v1` clean starter for evaluation v3. It contains the updated evaluator and task contracts while preserving the plain ESMC model and training baseline. The historical `autoresearch-v0` release uses different evaluation populations; remeasure baselines for every new v3 campaign.

Benchmark attempts start from `autoresearch-v1`: a single root commit containing the plain ESMC implementation and updated evaluation, with no research history. Allocate four matching H100 GPUs or four matching L40S GPUs on Linux with a working CUDA driver. A search round provides 20 minutes on H100 with FlashAttention-3 or one hour on L40S with FlashAttention-2; these budgets are roughly equivalent. Declare one hardware profile before search and fix the GPU model and backend across every method in a comparison.

The organizer prepares the workspace on the allocated node before giving it to an agent. Install the declared clean starter into a new directory outside existing Git checkouts, keep it isolated from research history, and run setup:

```bash
git clone --depth 1 --single-branch --no-tags --branch autoresearch-v1 \
  https://github.com/Lumin-Science/Nano-ProteinLM.git nano-protein-autoresearch
cd nano-protein-autoresearch
git remote remove origin
bash scripts/setup.sh --contact-v3-archive /path/to/contact-evaluation-v3.tar.gz
```

Record the starter commit and evaluator hashes with the organizer records. A published starter should pin the same source for every participant and contain no other branches, tags or research Git objects. Remove its remote before handing it to the agent. Do not reuse a research clone as a clean search workspace.

`scripts/setup.sh` needs uv `>=0.11.31,<0.12`; it installs the locked Python 3.11 environment, downloads 30 training shards and the original validation/source assets, installs the organizer-provided v3 contact archive, then prepares and verifies the fixed mask caches. Allow roughly 20 GB for data plus space for dependencies, checkpoints and run outputs. Use `--training-shards N` or `--training-samples N` to change the corpus size, and provision enough data for the selected source mixture and budget. Data and outputs default to `data/` and `outputs/` inside the workspace. Each task run checks the four GPUs and records them in its `ENVIRONMENT.json`.

```bash
# After preparation, one invocation consumes one search round:
bash tasks/171m-validation-loss_ar.sh configs/autoresearch/esmc-171m.yaml trial-001 42
```

Start the agent in this directory with a fresh conversation, the selected task and its own AutoResearch method. The task information-access rules prohibit inspecting other branches or searching for this repository and its previous findings online. The organizer enforces that policy, keeps other research workspaces inaccessible and uses a trusted evaluator; a Git branch and written rules alone cannot block online access.

## Design Space

**Immutable settings**:

- **Model and training settings:** use only the provided training corpus, without adding datasets. Keep the tokenizer, Stage-1 context of 512 tokens, global batch of 256 sequences and schedule of 500 linear warmup steps followed by constant learning rate, with no decay. Actual trainable parameters must stay within ±5% of the original 171M model, with no unused parameters added to satisfy the bound. Train from scratch without pretrained models.
- **Evaluation and execution:** preserve the hardware, round allowance and per-round training budget. Keep the evaluation code, data, masking, contact-probe procedure and metric definitions unchanged; do not train on held-out evaluation data. Preserve the published dependencies and source-data verification records.

**Design space**:
* Everything outside the immutable contract is open to research, including data selection and source mixture within the provided corpus, model architecture, training loss, optimizer, learning rate, weight decay and training implementation. [DATA.md](DATA.md) describes the available corpus, and [EVALUATION.md](EVALUATION.md) specifies the scientific measurements.

## Search Budget

A budgeted round consists of one training run and its evaluation. Each method receives **72 rounds**, with **20 minutes on 4×H100** or **1 hour on 4×L40S** per round. These are roughly equivalent search budgets. The totals are **24 node-hours / 96 H100 GPU-hours**, or **72 node-hours / 288 L40S GPU-hours**. Fix one hardware profile across methods in a comparison. Setup, checkpoint saving and evaluation add to elapsed runtime and are reported separately.

| Budget item | Protocol setting |
|---|---|
| Round allowance | **72** |
| Hardware per round | **4×H100 with FA3** or **4×L40S with FA2**; fixed profile across methods in a comparison |
| Training time per round | **20 minutes / 4/3 H100 GPU-hours**, or **1 hour / 4 L40S GPU-hours** |
| Total search training allowance | **24 node-hours / 96 H100 GPU-hours**, or **72 node-hours / 288 L40S GPU-hours** |
| Global batch | **256 sequences**, for example 64 per GPU on four GPUs |
| Learning-rate schedule | 500 linear warmup steps, then constant peak learning rate with no decay |
| Measurement after each round | Final checkpoint; MLM and P@L on the same fixed 8,192 chains |
| Outside the training clock | Environment/data setup, final checkpoint saving and evaluation; report their time separately |

Methods may spend rounds exploring new recipes or repeating earlier recipes. Two baseline runs of the untouched starting recipe, with seeds 42 and 43 on the allocated hardware, are free and calibrate the setup; every other training run, including a seed repeat or a further reference measurement, consumes a round. Retain failed attempts and their consumed compute; declare any infrastructure-failure replacement policy before the benchmark. A method's internal iteration may contain several budgeted rounds.

The training clock includes batch loading and synchronization. Prepare the inputs before timing a run and keep data placement consistent across methods. Random batch reads from shared network storage can stall training, so before the clock starts the task command copies the prepared training data to node-local storage: `/tmp`, or the folder named by `NANOPROTEIN_STAGE_DIR`. It reuses that copy while the data manifest is unchanged. Record the code revision, resolved recipe, data receipts, seed, actual steps and non-padding model tokens for each run, together with its metrics and elapsed training time.

Search results use the fixed evaluation described below. Proposal generation, repeated-seed comparisons, candidate retention and stopping within the round allowance are choices made by the AutoResearch method.

## Hill-climbing evaluation

The default reward is sequence-mean MLM negative log-likelihood on the fixed 8,192 contact chains. Every search checkpoint also reports P@L on those exact same chains. MLM remains the candidate-selection signal; P@L adds no non-regression gate. The older P@L task filename is a compatibility alias for this same MLM-selected paired evaluation.

| Measurement | Search setting |
| --- | --- |
| Default reward | MLM NLL, lower is better |
| MLM | Fixed 8,192 chains, context 512, one fixed mask seed 20260821 |
| Contact P@L | Same 8,192 chains, one probe attempt with seed 20260819 |
| Scored checkpoint | Final checkpoint at the round's training-time limit |
| Required receipt | `profile=search`, complete MLM and P@L populations |

Mask preparation is outside evaluation timing. Search baseline runs must be measured again with the updated evaluator; old 12,288-protein scores cannot serve as the new baseline. [EVALUATION.md](EVALUATION.md) specifies frozen IDs, deterministic crops, mask caches and contact-probe details. Each repeat training run still consumes one search round.

## Final evaluation

After search, each method submits its selected recipe for owner-run final evaluation. Freeze the recipe before this test. Train it and the reference, [configs/test-100k/esmc-171m.yaml](../configs/test-100k/esmc-171m.yaml), from scratch to **24,200,224,761 non-padding model tokens each**, using **one common training seed**: 42 unless another seed is declared before the comparison. Final training uses **global batch 1,024** and **1,000 linear warmup steps followed by constant learning rate, with no decay**, so the checkpoint can continue into Stage 2 training. Learning rate, weight decay and every other setting come from the submitted recipe. Both recipes use the provided corpus and retain their selected data mixtures.

| Measurement | Final evaluation setting |
|---|---|
| Training budget | **24,200,224,761 non-padding model tokens per recipe**, including BOS/EOS |
| Batch and schedule | **Global batch 1,024**; 1,000 linear warmup steps, then constant learning rate; learning rate and weight decay from the recipe |
| Training seeds | **1 per recipe**, 42 unless declared otherwise, matched between the selected recipe and reference |
| Hardware | Not fixed, because the token target defines the budget; the reference takes about 12 hours on four H100 GPUs |
| MLM validation | **26,062 contact chains × five fixed masks**, plus **original 12,288 proteins × five fixed masks**, context 512 |
| Contact P@L | **All 26,062 non-probe eligible chains**, five probe attempts |
| Scored checkpoint | Final checkpoint at the token target |
| Reported results | Three means and sample SDs: contact P@L, contact-population MLM, original-validation MLM; no separate single-mask score |

Final evaluation uses `--profile scaleup`. Its three SDs quantify variation across five probe attempts or five masks, keeping checkpoint, protein populations and crops fixed. They do not quantify training-seed variability. The original 20 probes remain separate from the 26,062 scored chains. Training exclusion of the extra 5,287 recovered chains is not yet verified; retain this qualification in reports.

The final training budget is separate from the 72-round search allowance. Prepare the required portion of the provided corpus before training and record each recipe's data selection, source exposure and any reuse. Keep the evaluation data and code fixed across recipes.

The reference takes roughly **12 hours on four H100 GPUs**, or about **48 H100 GPU-hours**. Actual time depends on the recipe and hardware; the token target determines completion. Score the checkpoint at the first optimizer update reaching that target and report its actual token count and overrun.

### Final-evaluation command

Prepare enough of the provided corpus for the recipe's source mixture at this token target; `bash scripts/setup.sh --training-samples 103424000 --contact-v3-archive /path/to/contact-evaluation-v3.tar.gz` in a fresh `DATA_ROOT` covers the default mixture at 100,000 updates of batch 1,024 ([data sizing](DATA.md#sizing-a-training-download)). Run the procedure once per recipe with a fresh run name and the same seed. The example uses the round-2 recipe on four H100 GPUs. On other GPUs, use `--attention-backend flash` and raise the 16-hour `--walltime-seconds` guard as needed. If a recipe needs a different micro-batch size for memory, keep the global batch at 1,024 and record the layout.

```bash
set -a
if [ -f .env ]; then source .env; fi
source .env.example
set +a
recipe=configs/test-100k/nanop-best-171m-round2.yaml
run_dir="$OUTPUT_ROOT/final-nanop-best-171m-round2-seed42"
mkdir -p "$OUTPUT_ROOT"
mkdir "$run_dir"
uv run --frozen python -m nanoprotein.check_environment \
  --require-gpus 4 --attention-backend flash3 \
  --output "$run_dir/ENVIRONMENT.json"
uv run --frozen python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m nanoprotein.train --config "$recipe" --seed 42 \
  --data-root "$DATA_ROOT/training" --output-root "$run_dir" \
  --max-steps none --max-model-tokens 24200224761 --schedule-steps 100000 \
  --walltime-seconds 57600 --attention-backend flash3 --warmup-steps 1000 \
  --micro-batch-size 64 --gradient-accumulation 4 \
  --checkpoint-interval 0 --periodic-evaluation-interval 0
uv run --frozen python -m nanoprotein.evaluate \
  --checkpoint "$run_dir/checkpoint-final.pt" --data-root "$DATA_ROOT/training" \
  --output-root "$run_dir/evaluation" \
  --profile scaleup \
  --contact-root "$DATA_ROOT/evaluation/contact-v3" --external-src "$DATA_ROOT/evaluation/source" \
  --prepared-root "$DATA_ROOT/evaluation/prepared-v3"
```

Require `stop_reason=max_model_tokens` and `model_token_budget_reached=true` in `TRAINING_COMPLETE.json`. The count includes BOS/EOS and excludes padding. An early wall-time stop is incomplete: continue it to the token target with `--resume` before scoring ([training and continuation](USAGE.md#training)). Report the actual tokens and overrun, the three mean/SD metrics from `evaluation/EVALUATION.json`.

This test measures whether search improvements carry over to longer training. It uses evaluation assets also available during search, so it does not establish performance on a blind holdout. Training a larger model requires its own agreed model size and comparison budget.

[LEADERBOARD.md](LEADERBOARD.md) collects recorded measurements with each study's hardware, seeds and evaluation sample counts. [AUTORESEARCH_BASELINE.md](AUTORESEARCH_BASELINE.md) documents our sequential-search method and experiment history.
