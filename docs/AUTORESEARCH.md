# AutoResearch

LuminBench Nano ESMC asks a practical scientific question: **how much useful
protein biology can a compact, sequence-only ESMC encoder learn under a fixed
and reproducible compute budget?** The repository provides the public training
data, model implementation, frozen evaluations, baselines, and end-to-end
training path needed to answer that question.

## Research task

| Field | Definition |
|---|---|
| **Task** | Compute-bounded protein representation learning with ESMC-300M |
| **Scientific lead** | Muchen Li |
| **Version** | 1.0 |
| **Status** | Pilot-ready |

The goal is to train a compact protein language model whose residue- and
protein-level embeddings expose useful structural and functional information
to frozen downstream readouts.

> Given a verified, evaluation-decontaminated corpus of protein sequences,
> produce an ESMC-300M-class encoder checkpoint within the fixed compute budget
> that improves frozen protein representation quality.

Inputs are public amino-acid sequences from UniRef90, MGnify, and OMG/IMG.
Outputs are a checkpoint, embeddings produced from that checkpoint, and the
data, environment, and training receipts needed to verify the run. The task
excludes structure-conditioned inputs, task-specific encoder fine-tuning,
private training data, evaluation-set adaptation, and changes to the frozen
evaluator.

## Experiment contract

Candidates may change the model, optimizer, training loss, learning-rate
schedule, batching, kernels, compilation, precision strategy, packing, and
checkpointing. The following remain fixed across comparisons:

- the pinned training corpus, mixture, tokenizer, and dependency lock;
- four matched GPUs and 3,600 seconds of synchronized training-loop time;
- held-out validation data and the frozen contact evaluator; and
- evaluation sequences, probes, masking seed, metrics, and reductions.

Every completed experiment reports full long-range contact P@L, final-window
training loss, frozen held-out validation loss, realized steps, model tokens,
parameter count, peak memory, and elapsed evaluation time. P@L is the primary
selection metric. A candidate is kept only when its P@L is strictly higher than
the incumbent under the same contract.

The public production reference uses the same task with a four-hour training
budget. Promotion beyond development additionally requires a controlled clean
rebuild, repeated runs, checkpoint verification, uncertainty reporting, and
review of trusted-task regressions.

The exact retained one-hour preset is
[`configs/autoresearch_300m_4xa100_1h.yaml`](../configs/autoresearch_300m_4xa100_1h.yaml).
It is opt-in; the public speedrun remains on the original ESMC-compatible
configuration unless `CONFIG` is set explicitly.

## AutoResearch-Codex Round 1

| Model | P@L | Delta vs. original | Train loss | Validation loss | Steps | Model tokens (M) | Parameters (M) | Peak VRAM (GB) | Train (h) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original ESMC | 0.0925 | — | 2.723 | 2.699 | 5,883 | 358 | 333 | 35 | 1 |
| **AutoResearch-Codex-Round1** | **0.0960** | **+0.0036 (+3.87%)** | **2.720** | 2.704 | 5,682 | 346 | 333 | 37 | 1 |

These results use Stage 1 training only, four NVIDIA L40S GPUs, and one hour of
synchronized training time per model. Relative to original ESMC, the retained
candidate adds learned residual/input routing, parameter-free transformer
RMSNorm, depth-scaled attention-output and FFN-down initialization, and a
linear learning-rate cooldown over the final 20% of training that ends at
0.1× the peak learning rate.

Subsequent decontaminated experiments and the current incumbent are recorded in
[`BASELINES.md`](BASELINES.md). Detailed experimental instructions live in
[`program.md` on the `auto-research` branch](https://github.com/Lumin-Science/LuminBench-Nano-ESMC/blob/auto-research/program.md).

## Run the supported training path

From a fresh clone, the project-level prerequisite is `uv >=0.11.31,<0.12`,
plus a supported NVIDIA driver and four visible BF16-capable GPUs:

```bash
git clone https://github.com/Lumin-Science/LuminBench-Nano-ESMC.git
cd LuminBench-Nano-ESMC
bash runs/speedrun.sh
```

The script creates the locked environment, downloads and verifies the required
public shard prefix, materializes on-disk training data, qualifies CUDA, and
trains ESMC-300M. See [`USAGE.md`](USAGE.md) for the concise usage guide.

## Reference documentation

- [`DATA.md`](DATA.md): training corpus, provenance, and download behavior.
- [`EVALUATION.md`](EVALUATION.md): datasets, splits, metrics, and execution.
- [`BASELINES.md`](BASELINES.md): baseline versions and results.
- [`USAGE.md`](USAGE.md): training and evaluation commands.
