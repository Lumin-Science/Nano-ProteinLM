# Research task definition

| Field | Definition |
|---|---|
| **Task** | Compute-bounded protein representation learning with ESMC-300M |
| **Scientific lead** | Muchen Li |
| **Version** | 1.0 |
| **Status** | Pilot-ready |

This page is the human-readable task contract. Machine-readable data manifests,
evaluation ledgers, run receipts, code, and environment locks remain the
authoritative artifacts when a summary and an executable contract disagree.

## Scientific purpose and recurrence

### Scientific purpose

The task develops the recurring capability to train a compact, sequence-only
protein encoder whose frozen embeddings expose useful structural and functional
information. Improving this capability would strengthen protein annotation,
remote-homology recognition, localization, fitness prediction, and contact
inference without requiring a separately fine-tuned encoder for each use case.

The scientific claim is deliberately narrower than “solve protein biology.” It
asks whether architecture, optimization, and systems improvements can extract
more useful representation quality from the same data interface, model class,
and compute clock.

### Recurring instances

An instance occurs whenever a new model, optimizer, training schedule, batching
strategy, kernel, or efficiency technique is proposed. Representative
instances are the original checkpoint-compatible ESMC-300M recipe, the current
P@L-selected 300M setting, and future AutoResearch experiments.

The target is stable enough for benchmark evaluation because every candidate
uses a versioned corpus release, tokenizer, masking objective, hardware count,
training clock, evaluation split ledger, probe family, and metric definition.
The scientific target does not change in response to candidate results.

### Scope and exclusions

Included work trains an ESMC-300M-class encoder from public protein sequences
and evaluates frozen representations. Candidates may improve model internals,
optimization, schedules, batching, kernels, compilation, checkpointing, and
other training-efficiency components.

Excluded claims and interventions are:

- structure-conditioned or multimodal encoder inputs;
- task-specific encoder fine-tuning or test-time label access;
- private or candidate-specific training corpora;
- changing evaluation examples, probes, metrics, or aggregation after seeing a
  candidate result;
- silently increasing the GPU count or synchronized training clock; and
- claiming parity with the original ESMC training corpus or paper protocol.

## Problem definition and data

### Canonical form

> Given a verified, evaluation-decontaminated corpus of protein sequences,
> produce an ESMC-300M-class encoder checkpoint within the fixed compute budget
> that improves frozen protein representation quality.

### Input and output

The scientific input is an amino-acid sequence. The training implementation
encodes normalized sequences into ESMC token IDs, adds boundary tokens, applies
the frozen masked-language-model corruption rule, and packs crops to the
configured context length. Data are stored locally as source/split mmap stores
materialized from checksum-verified Parquet shards.

The primary output is a checkpoint for a sequence-to-representation mapping:

- a residue embedding matrix with one vector per valid residue; and
- a protein embedding formed by the frozen windowing and mean-pooling rule.

Required operational outputs include the config, environment receipt, run
contract, metrics, completion receipt, and content hash of the final
checkpoint. These artifacts support downstream frozen probes and
attention-based contact prediction.

Exact model shapes are defined in [`ARCHITECTURE.md`](ARCHITECTURE.md). Data
schemas, preprocessing, source versions, mixture, deduplication, and splits are
defined in [`DATA.md`](DATA.md).

### Data-access mode

The task uses **publicly accessible data with mixed upstream terms**. The
supported release is hosted at
[`LuminScience/LuminBench-Nano-ESMC`](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC)
and is pinned by immutable revision in production runs. It combines UniRef90,
MGnify, and OMG/IMG representatives after normalization, global exact
ownership, length filtering, train/validation separation, and exact plus
MMseqs2 evaluation decontamination.

Development sees only the frozen training and validation interfaces. Final
evaluation uses fixed public-source reconstructions whose exact roles and
original publications are recorded in
[`EVALUATION_DATASETS.md`](EVALUATION_DATASETS.md). Public availability does
not authorize adaptive use of final test labels.

## Agent interface and evaluation

### Authorized resources and mutable components

| Component | Contract |
|---|---|
| Public resources | The pinned training release, released ESMC baselines, named evaluation sources, public literature, and packages in `uv.lock` |
| Mutable | Model internals, optimizer, training loss, learning-rate schedule, batching, kernels, compilation, precision strategy, packing, and checkpointing |
| Fixed | Corpus revision and mixture, tokenizer, dependency lock, GPU count, synchronized clock, evaluation rows and masking seed, probe definitions, metrics, and reductions |
| Candidate record | One tested working-tree change, fresh output directory, config, data/environment receipts, checkpoint, and result row; only improvements become commits |

The exact AutoResearch permissions are narrower where required and are
authoritative in [`program.md` on the `auto-research` branch](https://github.com/Lumin-Science/LuminBench-Nano-ESMC/blob/auto-research/program.md).

### Development feedback

Each candidate returns full-manifest long-range contact precision at L as its
development-selection score, plus final-window training loss and frozen
held-out validation loss. The run log also records training time, realized
steps, model tokens, parameter count, peak memory, and evaluation time. These
diagnostics explain learning and resource use but do not replace P@L selection.

One result is associated with one tested change and fresh output root. Only a
strict P@L improvement is committed; failed and discarded attempts use `NA` as
their commit and remain in the untracked result ledger.
The most plausible adaptive-overfitting route is repeated selection on the same
contact population; the safeguards are an immutable evaluator, single-concept
candidate changes, a complete query log, separate trusted guardrails, and
controlled repeated evaluation before promotion.

### Experimental budget

| Phase | Accelerator and clock | Included in comparison |
|---|---|---|
| AutoResearch candidate | Four matched GPUs; exactly 3,600 seconds of synchronized training-loop time | Training only; startup, checkpoint writing, and evaluation are logged separately; no timing dry run |
| Public production reference | Four GPUs; 14,400-second training-loop limit | The original ESMC-300M speedrun configuration |
| Development evaluation | Same four-GPU allocation where practical | Frozen full contact evaluator; execution may be accelerated only after exact parity |

The environment receipt records the actual GPU, Python, Torch, CUDA, BF16, and
FlashAttention qualification. No candidate receives a hidden allowance through
unreported memory, storage, or extra accelerators.

### Final evaluation and success rule

The primary promotion metric is mean long-range contact precision at L over the
frozen 20,775-chain population. The trusted P-CORE-Q4 panel, held-out MLM, raw
quarantined metrics, throughput, memory, and artifact-integrity checks are
reported as guardrails or diagnostics according to
[`EVALUATION.md`](EVALUATION.md).

A development candidate must strictly exceed the incumbent P@L under the same
contract to be retained. Scientific release additionally requires controlled
rebuild, repeated training runs, uncertainty reporting, checkpoint verification,
and review of every trusted-task regression. Failure to reproduce the gain, a
data/evaluation integrity failure, or an unexplained trusted regression
falsifies the intended improvement claim.

## Baselines, reproduction, and release

### Baselines

The comparison ladder contains:

1. metric-specific chance or prevalence readouts;
2. the original checkpoint-compatible local ESMC-300M recipe;
3. the current compute-matched scientific workflow; and
4. released ESMC-300M, ESMC-600M, and ESMC-6B checkpoints as strong public
   references.

Exact revisions, resources, repeated results, values, and fairness caveats are
kept in [`BASELINES.md`](BASELINES.md).

### Submission and reproduction

An acceptable submission includes source code, config, `uv.lock`, immutable
data revision and materialization receipt, environment receipt, training
contract, metrics, completion receipt, final checkpoint, and checkpoint hash.
The acceptance path is a fresh locked rebuild followed by receipt validation
and repeated frozen evaluation. The one-command training path and the verified
Killarney pilot are documented in [`REPRODUCTION.md`](REPRODUCTION.md).

### Openness and governance

Code is MIT licensed. The project-authored dataset compilation is CC BY-SA 4.0,
while upstream protein records retain their source-specific terms. Evaluation
datasets retain their original terms. Credentials, controlled caches, and
private infrastructure details are not release artifacts. Agent involvement,
code revision, data revision, environment, resources, metrics, and hashes are
recorded for each accepted run.

### Release sign-off

| Role | State before task release |
|---|---|
| Scientific lead | Task purpose, scope, and scientific claim require final sign-off |
| ML/task owner | Executable contract, budget, and mutable-component boundary require final sign-off |
| Evaluation reviewer | Split ledger, metric implementation, uncertainty, and quarantine decisions require final sign-off |
| Pilot-run owner | Fresh-environment data/training pilot passed; production-length repetition remains required |

The task remains **pilot-ready**, rather than fully released, until these roles
approve the same version and the unresolved repeated-run and evaluator risks in
[`RELEASE.md`](RELEASE.md).
