# 171M validation loss

Improve protein-model training recipes for masked-language modeling. This task defines the scientific objective, design boundaries and measurement for a budgeted run. It can be used by any search method.

## Background

The starting recipe is `configs/autoresearch/esmc-171m.yaml`: a plain ESMC model with 24 transformer layers, width 768, 12 attention heads and 170,671,168 trainable parameters, trained with AdamW at base learning rate 5e-4, weight decay 0.01 and global batch 256. The benchmark workspace must contain only this independent starting commit and changes made during your own search.

Training uses the provided processed `LuminScience/LuminBench-Nano-ESMC` corpus at revision `bd38448d50d8f426d7b9bd4410b53159ea001259`, containing UniRef90, MGnify and OMG/IMG proteins. Use its verified training splits; the original validation proteins and evaluated sequence fragments from the 20 probes and original 20,775 contact chains are included in the protected union under the declared identity/coverage exclusion rule. This does not exclude every longer parent sequence. Homology exclusion of the additional 5,287 final-evaluation chains is not yet verified; report this qualification.

## Score and budget

- **Score:** held-out sequence-mean MLM negative log-likelihood, **lower is better**. Read `validation_mlm.sequence_mean_nll` from the run's `evaluation/EVALUATION.json`. Report contact P@L alongside MLM on the same fixed 8,192 chains; P@L does not enter this task's selection gate.
- **Training:** **20 minutes (1,200 seconds) on 4×H100 with FlashAttention-3**, or **1 hour (3,600 seconds) on 4×L40S with FlashAttention-2**, starting from scratch. These are roughly equivalent search budgets. Declare one profile before search and keep the GPU model and backend fixed across methods in a comparison. Use global batch 256 and 500 linear warmup steps followed by constant peak learning rate, with no decay. The training clock includes batch loading, computation and synchronization; setup, final checkpoint saving and evaluation are outside it.
- **Search allowance:** 72 rounds: **24 node-hours / 96 H100 GPU-hours** with the H100 profile, or **72 node-hours / 288 L40S GPU-hours** with the L40S profile. Two baseline runs of the untouched starting recipe, with seeds 42 and 43 on the allocated hardware, are free; every other run, including candidates, seed repeats and further reference measurements, consumes one round. Retain failed attempts and their consumed compute; the organizer declares any infrastructure-failure replacement policy before search.
- **MLM evaluation:** the fixed 8,192 contact-evaluation chains, with their evaluated fragments at context 512 and fixed mask seed 20260821. Compute masked-token NLL within each chain, then average equally over chains. Preserve the ID list, masking and scoring code. Both MLM and P@L must complete, and the evaluation receipt must declare `profile=search`. See [EVALUATION.md](../docs/EVALUATION.md) for exact population and mask bindings.
- **Checkpoint:** score the final checkpoint at the training-time limit. A run must complete training and both MLM and P@L evaluations to supply a task score. Report the actual training duration and any overrun from the final optimizer update.

## Design boundaries

Use only the provided training corpus; data selection and source mixture may change within it. Keep the ESMC tokenizer, maximum context of 512 tokens, global batch of 256 sequences and 500-step linear-warmup/constant-LR schedule fixed. Actual trainable parameters must stay within ±5% of 170,671,168. Train from scratch: do not use pretrained weights, train on held-out evaluation sequences or labels, or add unused parameters to satisfy the size bound.

Preserve the declared hardware, compute budget, evaluation code and data, dependency lock and input-verification records. The task measurement scripts are protected during search. Architecture, training loss, optimizer, learning rate, weight decay and training implementation may change within these limits. Review the resolved configuration and completed run records against these boundaries; a successful command alone does not establish compliance.

## Measurement command

The organizer prepares the environment and data before agent access. Use Linux with a working CUDA driver and an allocation exposing exactly four matching H100 or four matching L40S GPUs. Use the `autoresearch-v1` clean starter: one root commit with the updated evaluator and untouched plain model/training baseline. Record its commit and evaluator hashes; do not reuse historical campaign scores. Inside that prepared clean starter:

```bash
# Inside the organizer-provided evaluation-v3 clean starter:
bash scripts/setup.sh
```

The organizer-provided starter must contain one root commit and no other branches or research history; create a campaign branch before committing search changes. An ordinary branch checkout in a research clone retains old Git objects and is not a clean benchmark workspace.

Data and outputs default to `data/` and `outputs/` inside the workspace; set `DATA_ROOT` and `OUTPUT_ROOT` in `.env` to change them. Training stores are under `$DATA_ROOT/training`; frozen contact assets and evaluator sources are under `$DATA_ROOT/evaluation/contact-v3`, `$DATA_ROOT/evaluation/prepared-v3` and `$DATA_ROOT/evaluation/source`. Setup defaults to 30 training shards containing 29,979,351 proteins and includes the original validation and contact source assets; v3 additionally downloads the pinned expanded contact archive and prepares the fixed MLM masks. Allow roughly 20 GB for data plus space for dependencies, checkpoints and outputs. Set `--training-shards N` during preparation to change the initial corpus size. Provision enough records from each source for the chosen mixture and budget; retain `DATA_COVERAGE.json` and report source exposure and any permitted data reuse.

Run the measurement from the prepared workspace, supplying a recipe, a fresh run name and an explicit training seed. The example seed 42 is a caller choice, not a task-imposed replication policy:

```bash
bash tasks/171m-validation-loss_ar.sh configs/autoresearch/esmc-171m.yaml experiment-001 42
```

The command qualifies the GPU model and attention backend, selects the matching 1,200-second or 3,600-second training limit, snapshots the recipe, trains a checkpoint and evaluates paired MLM validation loss and contact P@L through the standard APIs. It writes `recipe.yaml`, the effective `config.yaml`, `run_contract.json`, `TRAINING_COMPLETE.json`, `checkpoint-final.pt` and `evaluation/EVALUATION.json` under `$OUTPUT_ROOT/experiment-001/`. Verify that the training and evaluation receipts identify the same final checkpoint and the full evaluation populations. The command performs no replication, score aggregation or acceptance decision; every additional invocation consumes another budgeted round.

Record the release and code revisions, resolved recipe, training seed, GPU model and backend, data receipts, actual optimizer steps, non-padding model tokens, source exposure, metrics and elapsed training/evaluation times with the run ledger. Proposal generation, seed allocation and candidate selection belong to the search method.

## Final evaluation

After search, freeze the selected recipe. The owner trains that recipe and the reference, `configs/test-100k/esmc-171m.yaml`, from scratch to **24,200,224,761 non-padding model tokens each**, using **one common training seed**: 42 unless another seed is declared before the comparison. Final training uses **global batch 1,024** and **1,000 linear warmup steps followed by constant learning rate, with no decay**; learning rate, weight decay and every other setting come from the submitted recipe. Hardware is not fixed because the token target defines the budget; the reference takes about 12 hours on four H100 GPUs. Count BOS/EOS tokens and exclude padding. Stop at the first optimizer update reaching the token target, and report the actual token count and overrun. This final evaluation budget is separate from the 72 search rounds; do not launch it as part of an agent's search unless the owner requests it. [AUTORESEARCH.md](../docs/AUTORESEARCH.md#final-evaluation) gives the command.

Prepare enough of the provided corpus for each final recipe's selected source mixture and token budget. Retain tokenizer, context and parameter-size boundaries. Use `--profile scaleup` to report exactly three metrics, each as mean and sample SD: P@L over all 26,062 non-probe chains with five probe attempts; MLM on those 26,062 chains with five fixed masks; MLM on the original 12,288 validation proteins with five fixed masks. Keep the original crop fixed across masks. Do not report a separate single-mask final score. These SDs quantify probe/mask variation, not training-seed uncertainty. Search proteins overlap final evaluation, and training exclusion of the extra 5,287 chains is unverified. [EVALUATION.md](../docs/EVALUATION.md) defines the complete contract.

## Information-access rules

Start from the organizer-provided evaluation-v3 clean starter in a fresh workspace and a fresh agent context. Use only this starting code, the supplied data and observations produced within your allocated search budget. Keep the release commit (`git rev-parse HEAD` before search) with the run ledger.

Do not inspect, check out, fetch or restore `main`, other repository branches, tags, earlier commits, reflogs, Git objects, backups or other research workspaces. Local commits created during your own search are allowed. Do not add a remote to recover excluded material.

Do not search online for `https://github.com/Lumin-Science/Nano-ProteinLM`, its source code, mirrors, forks, issues, pull requests, reports or previous experimental findings. Do not obtain those findings through another agent, person, cached page or saved conversation. The organizer may prepare the declared clean starter and its pinned dependencies and datasets; this exception does not permit browsing the research repository during search.

General language/library documentation and scientific references are allowed only under the external-information policy declared by the organizer for every method. Record any accidental exposure to excluded material and notify the organizer before continuing. Do not use exposed findings to select a recipe. The organizer determines whether the attempt remains comparable.
