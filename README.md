# LuminBench Nano-ESMC

Nano-ESMC is a small, end-to-end implementation of ESMC-style protein
language-model training. It carries the whole recipe: public, decontaminated
protein sequences, a locked training environment, checkpoint receipts, and a
frozen evaluation.

It is built for two purposes.

1. **A reproduction you can run and change.** Protein LM pretraining is mostly
   published as a paper plus a released checkpoint. This repository is the
   training run itself, small enough to read in an afternoon and to execute on
   four GPUs, so the pretraining recipe is open to ordinary research rather
   than locked inside an industrial pipeline.
2. **A benchmark for agentic autoresearch.** The corpus, tokenizer, compute
   budget, and evaluation are pinned, so an automated agent can search for a
   better training recipe and its results can be compared against the baseline
   and against other agents on equal terms.

This is early work. The training path, the data release, and the contact
evaluator run end to end, but the evaluation suite is still growing and the
autoresearch loop has only a few rounds behind it. If you work on protein
language models or on agentic autoresearch, we would like the help: new
evaluation tasks, recipe candidates, reproductions on other hardware, or
arguments that something here is measuring the wrong thing. Open an issue or a
pull request.

## Data

We reconstructed the training-data recipe described in the
[ESMC paper](https://doi.org/10.64898/2026.06.03.729735), using the same three
source roles and a similar quality-filtering, deduplication, and 70%-identity
clustering pipeline. After evaluation decontamination, this produces a public
training dataset of 665,970,495 proteins.

The release is screened against the complete protected evaluation union,
including every RCSB Protein Data Bank chain used for contact P@L. Exact
matches and homologs at 30% or greater sequence identity with at least 80%
bidirectional coverage are removed before validation and training records are
selected.

| Source | Processed training records | Source reference |
|---|---:|---|
| UniRef90 | 74,175,974 | [UniRef clusters](https://doi.org/10.1093/bioinformatics/btu739) |
| MGnify | 328,949,335 | [MGnify in 2023](https://doi.org/10.1093/nar/gkac1080) |
| OMG/IMG (JGI-role surrogate) | 262,845,186 | [The OMG dataset](https://doi.org/10.1101/2024.08.14.607850) |
| **Total** | **665,970,495** | |

The main gap relative to ESMC is the JGI arm. The paper's exact July 2023 JGI
snapshot is not available as a reproducible public download, so this release
uses public OMG/IMG data as its surrogate and remains roughly 1.68 billion
70%-identity representatives below ESMC in that source role. Closing that gap
with a public, redistributable JGI-scale source is future work.

Construction is documented in full in [`docs/DATA.md`](docs/DATA.md).

- 🤗 [Hugging Face](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC/tree/bd38448d50d8f426d7b9bd4410b53159ea001259).

## Evaluation

What we are trying to produce is a good residue-level representation of a
protein. Almost everything built on a protein LM reads those representations
rather than the weights: ESMFold, for example, predicts structure from the
representations of a pretrained ESM encoder. The encoder therefore sets a
ceiling on what the models above it can do, which makes it worth measuring
directly instead of trusting training loss as a stand-in.

Long-range contact precision is the most informative cheap proxy we have for
that. Recovering which residue pairs are in physical contact while far apart in
sequence is close to the core of what a folding model needs from its encoder,
and it can be read out of a frozen model with a small probe. Following the ESMC
paper, we use full long-range contact precision at L (P@L) over a frozen
20,775-chain population as the headline metric. Training and validation MLM
losses are reported alongside it as diagnostics.

Our evaluation and the paper's both start from the RCSB Protein Data Bank at
the 2024-02-28 snapshot date, contain 20,775 chains, and follow the same
published construction rules. Biohub does not publish its ordered chain
manifest or raw-file digests, so item-for-item identity cannot be verified; the
two columns below are protocol-matched measurements, not a claim of an
identical evaluation set. All structural labels for our P@L evaluation come
from that frozen RCSB source, and those chains are in the protected union used
to decontaminate the training corpus.
[RCSB PDB](https://www.rcsb.org/); [ESMC](https://doi.org/10.64898/2026.06.03.729735).

| Released model | ESMC paper P@L-LR (95% CI) | Our full 20,775-chain P@L |
|---|---:|---:|
| ESMC-300M | 0.552 ± 0.002 | 0.5387 |
| ESMC-600M | 0.589 ± 0.002 | 0.5803 |
| ESMC-6B | 0.725 ± 0.002 | 0.7126 |

Contact precision is one view of one property, so we are also building a
broader frozen-representation suite, P-CORE, covering remote homology,
secondary structure, subcellular localization, and mutational fitness. It is a
reporting panel today, not a selection metric.

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for dataset lineage, exact
splits, probe definitions, P-CORE results, confidence intervals, and execution.

## Baselines

## AutoResearch

The reason the data and the evaluation are frozen is that we want to hand this
repository to agentic systems and ask them to do the research: propose a change
to the training recipe, run it, measure it, and keep it only if it worked. A
protein LM recipe is a large search space of architecture, optimizer, loss,
schedule, and systems choices, most of it explored by hand today. The question
we put to the agent is:

> Train a protein sequence encoder from scratch under a fixed compute
> budget and improve the biological information exposed by its frozen
> representations.

**Scope.** A candidate may change the model, optimizer, loss, schedule,
batching, kernels, and other training-efficiency components. The processed
corpus, tokenizer, dependency lock, hardware class, training clock, evaluation
examples, probes, and metrics stay fixed.

**Budget.** One round is one hour of synchronized training time on four NVIDIA
L40S GPUs, starting from scratch — roughly ten minutes on eight H100s. That is
short enough for an agent to run many rounds per day and long enough that the
model learns something measurable.

**Evaluation.** Selecting on a single metric at a single compute budget is easy
to game, and we assume an agent will find the cheap wins. One hour of training
is short enough that a recipe can win by front-loading progress in ways that
cost capacity later, and optimizing P@L alone rewards changes that suit this
particular probe rather than the representation. So the search budget is not
the acceptance budget: a retained recipe is re-run at about 36× the compute —
1.5 days on four L40S — and is only credited if the gain survives the scale-up.
Those runs are in progress and the table is not published yet.

| Model | P@L | Delta vs. original | Train loss | Validation loss | Steps | Model tokens (M) | Parameters (M) | Peak VRAM (GB) | Train (h) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original ESMC | 0.0925 | — | 2.723 | 2.699 | 5,883 | 358 | 333 | 35 | 1 |
| **AutoResearch-Codex-Round1** | **0.0960** | **+0.0036 (+3.87%)** | **2.720** | 2.704 | 5,682 | 346 | 333 | 37 | 1 |

Round 1's retained recipe adds learned residual/input routing, parameter-free
transformer RMSNorm, depth-scaled attention-output and FFN-down initialization,
and a final-20% linear learning-rate cooldown ending at 0.1× peak.

### ESMC-171M AutoResearch

The 171M campaign transfers the retained architecture and Muon optimizer to a
24-layer, width-768 model, then evaluates each recipe from scratch for one hour
on four L40S GPUs. We re-ran the starting baseline and all three retained
changes with matched seeds 42, 43, and 44. R02 won on every seed and is now the
retained 171M AutoResearch preset:
[`configs/autoresearch_171m_4xl40s_1h.yaml`](configs/autoresearch_171m_4xl40s_1h.yaml).

| Recipe | Seed 42 P@L | Seed 43 P@L | Seed 44 P@L | Mean P@L ± SD | Delta vs. baseline |
|---|---:|---:|---:|---:|---:|
| Starting baseline | 0.1056 | 0.1008 | 0.0925 | 0.0996 ± 0.0067 | — |
| R01: differential Muon LR | 0.1068 | 0.1032 | 0.0955 | 0.1019 ± 0.0058 | +0.0022 (+2.22%) |
| **R02: R01 + RoPE base 20,000** | **0.1081** | **0.1058** | **0.1091** | **0.1077 ± 0.0017** | **+0.0080 (+8.06%)** |
| R03: R02 + attention Muon LR 1.0 | 0.1055 | 0.1024 | 0.1054 | 0.1044 ± 0.0018 | +0.0048 (+4.82%) |

The promoted R02 recipe uses Muon LR scales of 0.9 for attention and 0.75 for
FFNs, with RoPE base 20,000. It has 170.56M trainable parameters and processed
561.65M model tokens on average within the fixed one-hour training window.

The starting baseline in this table already uses Muon and the retained
architecture; it is distinct from `configs/esmc-171m-original.yaml`. Validation
loss mean and standard deviation are not published in the campaign record.

See [`docs/AUTORESEARCH.md`](docs/AUTORESEARCH.md) for the experiment contract
and [`docs/BASELINES.md`](docs/BASELINES.md) for detailed baseline context.

## Usage

From a fresh clone with `uv >=0.11.31,<0.12`, a supported NVIDIA driver, and
four visible BF16-capable GPUs:

```bash
git clone https://github.com/Lumin-Science/LuminBench-Nano-ESMC.git
cd LuminBench-Nano-ESMC
bash runs/speedrun.sh
```

The speedrun creates the locked environment, downloads and verifies the
required processed-data shards, qualifies CUDA, trains ESMC-300M, and verifies
the final artifacts. Configuration, smoke-run, and evaluation commands are in
[`docs/USAGE.md`](docs/USAGE.md).

Source and issues:
[Lumin-Science/LuminBench-Nano-ESMC](https://github.com/Lumin-Science/LuminBench-Nano-ESMC).

## Citation

If you use Nano-ESMC, please cite this repository and the original
[ESMC paper](https://doi.org/10.64898/2026.06.03.729735):

```bibtex
@software{lumin_science_nano_esmc_2026,
  author = {Muchen Li},
  title = {Nano-Protein-LM: A Minimal Reproduction of ESMC Language-Model Training},
  year = {2026},
  url = {https://github.com/Lumin-Science/LuminBench-Nano-ESMC}
}

@article{candido2026language,
  author = {Candido, Salvatore and Hayes, Thomas and Rao, Roshan and others},
  title = {Language Modeling Materializes a World Model of Protein Biology},
  journal = {bioRxiv},
  year = {2026},
  doi = {10.64898/2026.06.03.729735}
}
```
