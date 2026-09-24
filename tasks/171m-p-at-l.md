# 171M contact P@L

Improve protein-model training recipes for long-range contact prediction. This task defines the scientific objective, design boundaries and measurement for a budgeted run. It can be used by any search method.

## Background

The starting recipe is a plain ESMC model with 24 transformer layers, width 768, 12 attention heads and 170,671,168 trainable parameters, using AdamW. Start from `configs/default.yaml` in the released `autoresearch` branch. The benchmark workspace must contain only this independent starting commit and changes made during your own search.

Training uses the provided processed `LuminScience/LuminBench-Nano-ESMC` corpus at revision `bd38448d50d8f426d7b9bd4410b53159ea001259`, containing UniRef90, MGnify and OMG/IMG proteins. Use its verified training splits; all validation sequences and contact-evaluation data are held out from model training.

## Score and budget

- **Score:** contact P@L, **higher is better**. Read `contact.precision_at_l` from the run's `evaluation/EVALUATION.json`. MLM validation loss is a diagnostic.
- **Training:** **20 minutes (1,200 seconds) on 4×H100 with FlashAttention-3**, or **1 hour (3,600 seconds) on 4×L40S with FlashAttention-2**, starting from scratch. These are roughly equivalent search budgets. Declare one profile before search and keep the GPU model and backend fixed across methods in a comparison. Use 554 linear warmup steps followed by constant peak learning rate. The training clock includes batch loading, computation and synchronization; setup, final checkpoint saving and evaluation are outside it.
- **Search allowance:** 72 rounds: **24 node-hours / 96 H100 GPU-hours** with the H100 profile, or **72 node-hours / 288 L40S GPU-hours** with the L40S profile. Each reference measurement, candidate run or seed repeat consumes one round. Retain failed attempts and their consumed compute; the organizer declares any infrastructure-failure replacement policy before search.
- **MLM evaluation:** 4,096 fixed validation proteins, evaluated as 1,024 batches of four at context 512, with sampling and masking seed 20260821. Compute masked-token negative log-likelihood within each protein, then average equally over proteins. Preserve the frozen validation source mixture and masking procedure.
- **Contact evaluation:** all 20,775 frozen chains, one fitted probe per checkpoint using the fixed probe split and fitting procedure, and 32 workers across the four GPUs. P@L is precision among the top L predicted long-range contacts, where L is the evaluated chain length, averaged equally over chains. Values are fractions from 0 to 1. Report the 5,000-replicate chain-bootstrap 95% interval from `contact.precision_at_l_uncertainty`; this measures variation over chains, not training seeds.
- **Checkpoint:** score the final checkpoint at the training-time limit. A run must complete training and both evaluations to supply a task score. Report the actual training duration and any overrun from the final optimizer update.

## Design boundaries

Use only the provided training corpus; data selection and source mixture may change within it. Keep the ESMC tokenizer, maximum context of 512 tokens and 554-step linear-warmup/constant-LR schedule fixed. Actual trainable parameters must stay within ±5% of 170,671,168. Train from scratch: do not use pretrained weights, train on held-out evaluation sequences or labels, or add unused parameters to satisfy the size bound.

Preserve the declared hardware, compute budget, evaluation code and data, dependency lock and input-verification records. The task measurement scripts are protected during search. Architecture, training loss, optimizer settings, batch size and training implementation may change within these limits. Review the resolved configuration and completed run records against these boundaries; a successful command alone does not establish compliance.

## Measurement command

The organizer prepares the environment and data before agent access. Use Linux with a working CUDA driver and an allocation exposing exactly four matching H100 or four matching L40S GPUs. From an organizer checkout containing the preparation helper, create a fresh destination outside existing Git checkouts:

```bash
bash runs/setup_autoresearch.sh ../nano-protein-autoresearch
cd ../nano-protein-autoresearch
```

If repository access uses an SSH key, add `--repository git@github.com:Lumin-Science/Nano-ProteinLM.git` to the setup command. Use `--revision FULL_COMMIT_SHA` to pin the same released starting point for every participant. The helper clones only `autoresearch`, checks the release manifest, removes the remote, installs the locked environment, qualifies the GPUs and prepares the data. An ordinary branch checkout retains old Git objects and is not a clean benchmark workspace.

The prepared workspace sets `DATA_ROOT` and `OUTPUT_ROOT` in `.env`, defaulting to `data/` and `outputs/` inside that workspace. Training stores are under `$DATA_ROOT/training`; frozen contact assets and evaluator sources are under `$DATA_ROOT/evaluation/contact` and `$DATA_ROOT/evaluation/source`. Setup defaults to 30 training shards containing 29,979,351 proteins and includes all MLM validation and contact assets. Allow roughly 20 GB for data plus space for dependencies, checkpoints and outputs. Set `--training-shards N` during preparation to change the initial corpus size. Provision enough records from each source for the chosen mixture and budget; retain `DATA_COVERAGE.json` and report source exposure and any permitted data reuse.

Run the measurement from the prepared workspace, supplying a recipe, a fresh run name and an explicit training seed. The example seed 42 is a caller choice, not a task-imposed replication policy:

```bash
bash tasks/171m-p-at-l_ar.sh configs/default.yaml experiment-p-at-l-001 42
```

The command qualifies the GPU model and attention backend, selects the matching 1,200-second or 3,600-second training limit, snapshots the recipe, trains a checkpoint and evaluates it through the standard APIs. It writes `recipe.yaml`, the effective `config.yaml`, `run_contract.json`, `TRAINING_COMPLETE.json`, `checkpoint-final.pt` and `evaluation/EVALUATION.json` under `$OUTPUT_ROOT/experiment-p-at-l-001/`. Verify that the training and evaluation receipts identify the same final checkpoint and the full evaluation populations. The command performs no replication, score aggregation or acceptance decision; every additional invocation consumes another budgeted round.

Record the release and code revisions, resolved recipe, training seed, GPU model and backend, data receipts, actual optimizer steps, non-padding model tokens, source exposure, metrics and elapsed training/evaluation times with the run ledger. Proposal generation, seed allocation and candidate selection belong to the search method.

## Final evaluation

After search, freeze the selected recipe. The owner trains that recipe and the plain starting reference from scratch for **24 billion non-padding model tokens each**, on the same declared four-GPU hardware, using **one common training seed declared before the comparison**. Count BOS/EOS tokens and exclude padding. Stop at the first optimizer update reaching the token target, and report the actual token count and overrun. This final evaluation budget is separate from the 72 search rounds; do not launch it as part of an agent's search unless the owner requests it.

Prepare enough of the provided corpus for each final recipe's selected source mixture and token budget. Retain the same tokenizer, context, schedule, parameter-size and evaluation boundaries. Score each final checkpoint with the same 4,096-protein MLM evaluation and full 20,775-chain contact evaluation specified above. Report both metrics and the 5,000-replicate chain-bootstrap 95% interval for P@L. One training seed does not estimate training-seed variability. Because search uses these same evaluation assets, this test measures transfer to longer training rather than performance on a blind holdout.

## Information-access rules

Start from the released `autoresearch` branch in a fresh workspace and a fresh agent context. Use only this starting code, the supplied data and observations produced within your allocated search budget. Keep the release commit and provisioning receipt with the run ledger.

Do not inspect, check out, fetch or restore `main`, other repository branches, tags, earlier commits, reflogs, Git objects, backups or other research workspaces. Local commits created during your own search are allowed. Do not add a remote to recover excluded material.

Do not search online for `https://github.com/Lumin-Science/Nano-ProteinLM`, its source code, mirrors, forks, issues, pull requests, reports or previous experimental findings. Do not obtain those findings through another agent, person, cached page or saved conversation. The organizer may download the released `autoresearch` branch and its pinned dependencies and datasets during preparation; this exception does not permit browsing the research repository during search.

General language/library documentation and scientific references are allowed only under the external-information policy declared by the organizer for every method. Record any accidental exposure to excluded material and notify the organizer before continuing. Do not use exposed findings to select a recipe. The organizer determines whether the attempt remains comparable.
