# LuminBench Nano ESMC

LuminBench Nano ESMC asks a practical scientific question: **how much useful
protein biology can a compact, sequence-only ESMC encoder learn under a fixed
and reproducible compute budget?** The repository provides the public training
data, model implementation, frozen evaluations, baselines, and an end-to-end
`uv` speedrun needed to answer that question.

## Research task card

| Field | Definition |
|---|---|
| **Task** | Compute-bounded protein representation learning with ESMC-300M |
| **Scientific lead** | Muchen Li |
| **Version** | 1.0 |
| **Status** | Pilot-ready |

### Scientific purpose

The recurring capability is to train a compact protein language model that
turns amino-acid sequences into residue- and protein-level embeddings useful to
frozen downstream readouts. A materially better result would improve
sequence-only protein annotation and structural reasoning without adding task-specific
encoder fine-tuning or increasing the experimental compute budget.

The task recurs whenever the model architecture, optimizer, training schedule,
or systems implementation changes. It is benchmarkable because the corpus,
tokenizer, compute clock, evaluation populations, probes, and metrics are
versioned and held fixed across candidate runs.

### Canonical problem

> Given a verified, evaluation-decontaminated corpus of protein sequences,
> produce an ESMC-300M-class encoder checkpoint within the fixed compute budget
> that improves frozen protein representation quality.

Inputs are public amino-acid sequences from UniRef90, MGnify, and OMG/IMG.
Outputs are a checkpoint, embeddings produced from that checkpoint, and the
data/environment/training receipts required to reproduce it. The task excludes
structure-conditioned inputs, task-specific encoder fine-tuning, private
training data, evaluation-set adaptation, and changes to the frozen evaluator.

### Agent interface and evaluation

Candidate work may change the model, optimizer, training loss, schedule,
batching, kernels, and other training-efficiency components. It may not change
the protected data release, tokenizer, locked dependencies, compute clock, or
evaluation definitions.

AutoResearch development compares one-hour experiments under four GPUs and
commits only improvements. The public production reference uses the same task
with a four-hour training budget. Development selection maximizes full
long-range contact precision at L while also reporting training and held-out
validation loss; final reporting includes the trusted P-CORE-Q4 representation
panel. A result is promotable only when it beats the incumbent under the frozen
protocol, leaves no trusted-task regression unexplained, and passes controlled
reproduction.

### Baselines and release

Comparisons include metric-specific null readouts, the original
checkpoint-compatible ESMC-300M recipe, the current compute-matched workflow,
and released ESMC-300M, 600M, and 6B checkpoints evaluated through the same
local protocol. Exact versions, results, resource use, and fairness caveats are
in [the baseline contract](docs/BASELINES.md).

Every accepted run must retain its config, lockfile digest, immutable data
revision, environment receipt, training contract, metrics, and checkpoint
hash. Dataset and code artifacts are public; model promotion still requires
the review and sign-off described in [the release contract](docs/RELEASE.md).

The complete task definition, including feedback policy, experimental budgets,
success rule, reproduction requirements, governance, and unresolved sign-offs,
is in [`docs/TASK.md`](docs/TASK.md).

## AutoResearch-Codex Round 1

| Model | Exact P@L | Delta vs. original | Train loss | Validation loss | Steps | Model tokens (M) | Parameters (M) | Peak VRAM (MiB) | Train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original ESMC | 0.0924624073 | — | 2.723379 | 2.698895 | 5,883 | 357.792 | 332.997 | 35,579.898 | 3,600.602 |
| **AutoResearch-Codex-Round1** | **0.0960432677** | **+0.0035808604 (+3.87%)** | **2.720052** | 2.703555 | 5,682 | 345.564 | 332.823 | 37,698.239 | 3,600.004 |

These results use Stage 1 training only, four NVIDIA L40S GPUs, and one hour of
synchronized training time per model.

Relative to original ESMC, `AutoResearch-Codex-Round1` adds learned
residual/input routing, parameter-free transformer RMSNorm, depth-scaled
attention-output and FFN-down initialization, and a linear learning-rate
cooldown over the final 20% of training that ends at 0.1× the peak learning
rate.

## One-command training

From a fresh clone, the only project-level tool required is
`uv >=0.11.31,<0.12` (plus a supported NVIDIA driver and four visible GPUs):

```bash
git clone https://github.com/Lumin-Science/LuminBench-Nano-ESMC.git
cd LuminBench-Nano-ESMC
bash speedrun.sh
```

The script creates the locked environment, downloads and verifies the required
public shard prefix, materializes local training data, qualifies CUDA, and
trains ESMC-300M. Environments, caches, data, receipts, and checkpoints default
to `.exps/`. Copy [`.env.example`](.env.example) to `.env` to change the
artifact root or run settings. See [the reproduction guide](docs/REPRODUCTION.md)
for smoke runs, receipts, hardware verification, and full evaluation.

## Documentation

| Topic | Canonical document |
|---|---|
| Full research-task contract | [`docs/TASK.md`](docs/TASK.md) |
| Model and mutable architecture | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Training corpus, provenance, and download | [`docs/DATA.md`](docs/DATA.md) |
| Evaluation metrics and execution | [`docs/EVALUATION.md`](docs/EVALUATION.md) |
| Evaluation datasets and original publications | [`docs/EVALUATION_DATASETS.md`](docs/EVALUATION_DATASETS.md) |
| Baseline versions and results | [`docs/BASELINES.md`](docs/BASELINES.md) |
| Fresh-environment reproduction | [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) |
| Publication and governance | [`docs/RELEASE.md`](docs/RELEASE.md) |
| AutoResearch loop and end-to-end round command | [`program.md` on `auto-research`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC/blob/auto-research/program.md) |

`main` contains the supported task. Active experiments, reports, plans, and
research receipts are isolated under `dev/` and on the `auto-research` branch.

Source and issues: [Lumin-Science/LuminBench-Nano-ESMC](https://github.com/Lumin-Science/LuminBench-Nano-ESMC).
