# LuminBench Nano-ESMC

Nano-ESMC implements ESMC-style protein language-model training on four GPUs,
with public, decontaminated sequences, pinned dependencies, saved run records,
and fixed evaluations.

The repository supports two uses:

1. **Reproduce and modify protein LM pretraining.** Train from scratch and
   inspect the data pipeline, model, optimizer, and evaluation code.
2. **Benchmark agentic autoresearch.** Let an agent test training recipes under
   a fixed budget, select improvements across repeated seeds, and check whether
   they carry over to longer training runs.

The evaluation suite is still growing. Contributions of evaluation tasks,
training recipes, and reproductions on other hardware are welcome through
issues and pull requests.

## Data

We reconstructed the training-data recipe described in the
[ESMC paper](https://doi.org/10.64898/2026.06.03.729735), using the same three
source roles and a similar quality-filtering, deduplication, and 70%-identity
clustering pipeline. After evaluation decontamination, this produces a public
training dataset of 665,970,495 proteins.

The release is screened against all protected evaluation datasets,
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

Download the pinned release from
[Hugging Face](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC/tree/bd38448d50d8f426d7b9bd4410b53159ea001259).

## Evaluation

We measure held-out masked language-model (MLM) loss and the structural
information available from a frozen encoder. **Validation loss is the default
autoresearch reward; lower is better.** Training loss and contact precision are
reported alongside it.

For structure, we follow the ESMC paper's long-range contact precision at L
(P@L) over a fixed 20,775-chain population. A small probe predicts which
residues are in physical contact despite being far apart in sequence. This
checks a property of the learned representation that MLM loss alone does not
establish.

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

We are also building P-CORE, a suite of frozen-representation evaluations for
remote homology, secondary structure, subcellular localization, and mutational
fitness. These results do not affect autoresearch selection.

See [docs/EVALUATION.md](docs/EVALUATION.md) for dataset lineage, splits, probe
definitions, P-CORE results, confidence intervals, and execution.

## Baselines

Run these from the repository root after cloning it as described in
[Usage](#usage). Both settings start from the 24-layer, width-768 ESMC-171M
AdamW baseline.

See the [config index](configs/README.md) for current presets, historical
recipes, and superseded pilots.

Research setting: one hour on four L40S GPUs per seed, global batch 256.

```bash
for seed in 42 43; do
  CONFIG="$PWD/configs/program2/baseline_seed${seed}.yaml" \
  NUM_GPUS=4 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  WALLTIME_SECONDS=3600 \
  RUN_NAME="baseline-171m-1h-seed${seed}" \
    bash runs/speedrun.sh
done
```

Scale-up setting: 100,000 steps on four H100 GPUs, global batch 1,024.
The 16-hour wall-time guard leaves room to finish the step budget.

```bash
CONFIG="$PWD/configs/esmc-171m-default-h100-fa3-b1024-stage1-100k.yaml" \
NUM_GPUS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WALLTIME_SECONDS=57600 \
RUN_NAME=baseline-171m-100k \
  bash runs/speedrun.sh
```

These commands train and save checkpoints. Use a fresh run name for repeats;
evaluation requires the prepared datasets and commands in
[docs/EVALUATION.md](docs/EVALUATION.md). The exact research evaluation is
specified in [program.md](program.md).

## AutoResearch

The authoritative task definition is [program.md](program.md), organized as:
**Background** (research question and established codebase), **Autoresearch**
(protocol and boundaries), and **Test of Progress**. It includes complete
train/evaluate commands, score extraction, fixed settings and success criteria.

Research selects lower mean MLM validation loss across N=2 training seeds after
one hour on four L40S GPUs per seed. The new Test of Progress uses 24,200,224,761
non-padding model tokens per seed on four H100s and requires both lower mean
MLM loss and higher mean P@L. The completed leaderboard below retains its original
100k-step, single-seed protocol. Raw research and verification scores remain
separate because their budgets and MLM evaluation sizes differ.

**38-round history.** The updated run log contains the AdamW baseline and 38
candidate rounds: 78 one-hour runs across seeds 42 and 43, with five kept
changes. R30–R38 were all discarded, so R29 remains the best accepted recipe
in this history, which predates the ±5% parameter rule.
The five numbered changes are **1: Muon → 2: batch balance → 3: sqrt loss →
4: FFN width 1536 → 5: tied embeddings**. Mean validation
loss falls from **2.63868 to 2.58057 (2.20%)**. R29's accepted configs are
available for [seed 42](configs/program2/r29_tied_seed42.yaml) and
[seed 43](configs/program2/r29_tied_seed43.yaml).

Each point below is a method's mean validation loss with thin **±1 sample SD**
error bars. Numbered markers **1–5** identify the accepted changes; the x-axis
retains their original search-round numbers. The green line follows the
current best accepted recipe; lower means that fail the acceptance rule do
not advance it. Change **4** is marked separately because it changes model size
and is excluded from the fixed-size tests below. Source:
[run log through R38](reports/program2/runs-through-r38.tsv), with
[import details and the original R29 audit](reports/program2/README.md).
Regenerate the [SVG](reports/program2/validation-loss.svg) and PNG with
[the plotting script](scripts/plot_autoresearch_history.py).

![Validation loss across 38 search rounds. Numbered accepted changes are 1 Muon, 2 batch balance, 3 sqrt loss, 4 FFN 1536 and 5 tied embeddings. Change 4 is excluded from the fixed-size Test Leaderboard.](reports/program2/validation-loss.png)

| Research metric | Baseline | 1: + Muon | 2: + batch balance | 3: + sqrt loss | 4: + FFN 1536* | 5: + tied embeddings* |
|---|---:|---:|---:|---:|---:|---:|
| Validation loss ↓ | 2.63868 ± 0.01303 | 2.61807 ± 0.00945 | 2.60415 ± 0.00650 | 2.59437 ± 0.00578 | 2.59095 ± 0.00132 | **2.58057 ± 0.00544** |
| P@L (%) ↑ | 9.648 ± 0.598 | 9.795 ± 0.189 | 9.370 ± 0.270 | **10.533 ± 0.366** | 9.829 ± 0.286 | 9.527 ± 0.720 |

Entries are **mean ± sample SD across seeds 42 and 43**, recomputed from the
[research run log](reports/program2/runs-through-r38.tsv). P@L is in percent;
its SD is in percentage points. Selection uses validation loss, so an accepted
change need not improve P@L. Each research seed receives one hour on four
L40S GPUs, batch 256, and 32 fixed MLM validation sequences.

*Research change 4 reduces the model from 170.67M to 142.36M parameters;
research change 5 inherits that narrower FFN and has 142.31M parameters.
These historical results are preserved. The fixed-size Test Leaderboard
**skips change 4** and tests tied embeddings directly on change 3, retaining
FFN width 2,048.

## Test Leaderboard

The completed fixed-size tests follow **Baseline → 1 → 2 → 3 → 5**. Each recipe
starts from scratch for **100,000 Stage-1 steps on four H100s**, batch **1,024**
and **102.4M sampled sequences**, with FFN width **2,048** and the same full
evaluations. **Setting 4 (FFN narrowing) is excluded because it changes model
size.** The original six-recipe leaderboard, including previous RoPE20k R02,
is preserved in the [archive](docs/archive/TEST_LEADERBOARD_20260908.md).

| Recipe | Validation loss ↓ | P@L ↑ | P@L 95% CI | Training time |
|---|---:|---:|---:|---:|
| Baseline: ESMC-like AdamW | 2.47436 | 26.505% | 26.295–26.719% | 12h 01m |
| 1: + Muon (R02 recipe)† | 2.43781 | 30.165% | 29.936–30.394% | 12h 58m |
| 2: + batch balance | 2.43872 | 30.715% | 30.487–30.948% | 12h 34m |
| **3: + sqrt loss** | **2.41872** | **32.682%** | **32.447–32.920%** | **12h 35m** |
| 5: + tied embeddings | 2.42304 | 31.884% | 31.651–32.123% | 12h 33m |

†The completed Muon test uses **R02 with RoPE10k**, including parameter-free
RMSNorm, learned residual/input routing, depth-scaled initialization and
R02's optimizer-group multipliers. It is not a Muon-only ablation of the
research baseline. Settings 2, 3 and 5 inherit that recipe. Setting 5 is the
already completed `r29_tied` test, previously numbered **Setting 4** in the
archived four-setting campaign; only its display label changes.

**Setting 3 is best on both metrics:** validation loss is **2.25% lower** and
P@L is **6.18 percentage points higher** than the AdamW baseline, with **4.71%
longer training**. It combines hybrid Muon/AdamW, parameter-free transformer
RMSNorm, learned residual/input routing, depth-scaled initialization, rank
balancing, and square-root masked-target weighting. RoPE remains 10k, FFN
width remains 2048, and its embeddings are untied.

The next scale-up comparison uses **eight H100s, batch 2,048, 100,000 steps,
and full evaluation every 10,000 steps** on Nibi. The [AdamW baseline
learning curve](reports/nibi-baseline-b2048-100k-eval10k-20260908/README.md)
is followed by [Setting 3](reports/nibi-setting3-b2048-100k-eval10k-20260908/README.md).
Both preserve their full final optimizer checkpoints for continuation. These
runs have twice the sequence exposure of the four-GPU table above.

All use base LR 5e-4, base WD 0.01, and a 1,000-step warmup; Muon retains R02's
per-group LR/WD multipliers. Validation uses the same 4,096 held-out sequences
and P@L uses the same 20,775 chains. Intervals are 5,000-resample chain-bootstrap
95% CIs, not training-seed uncertainty; each recipe has one training seed.
Training times exclude evaluation and are approximate to the minute.

See **[best recipe versus baseline: differences, figures, and worked examples](docs/BEST_RECIPE_VS_BASELINE.md)**,
the [complete results and adjacent comparisons](reports/fir-r02-rope10k-100k-20260906/README.md),
and the [historical baseline/R02 records](reports/fir-171m-100k-20260906/README.md).
The narrower-FFN change is excluded from this test track; its historical
implementation is retained in [TODO](TODO.md).

## Usage

From a fresh clone with `uv >=0.11.31,<0.12`, a supported NVIDIA driver, and
four visible BF16-capable GPUs:

```bash
git clone https://github.com/Lumin-Science/LuminBench-Nano-ESMC.git
cd LuminBench-Nano-ESMC
```

Then run one of the [baseline commands](#baselines) above. The speedrun creates
the locked environment, downloads and verifies the required processed-data
shards, qualifies CUDA, trains the selected model, and checks the saved
artifacts. Configuration, smoke-run, and evaluation commands are in
[docs/USAGE.md](docs/USAGE.md).

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
