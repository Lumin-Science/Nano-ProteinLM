# LuminBench Nano-ESMC

LuminBench Nano-ESMC is a minimal, end-to-end reproduction of ESMC-based
protein language-model training. It provides the complete recipe from public,
decontaminated protein sequences through a locked training environment,
checkpoint receipts, and frozen evaluation.

The same recipe is formalized for hill climbing: hold the data, compute budget,
and evaluation fixed; change one model or training idea at a time; and retain
only measured improvements.

## Data

We reconstructed the training-data recipe described in the
[ESMC paper](https://doi.org/10.64898/2026.06.03.729735), using the same three
source roles and a similar quality-filtering, deduplication, and 70%-identity
clustering pipeline. After evaluation decontamination, this produces a public
training dataset of **665,970,495 proteins**.

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
snapshot is not directly available as a reproducible public download, so this
release uses public OMG/IMG data as its surrogate and remains approximately
1.68 billion 70%-identity representatives below ESMC in that source role.
Closing this gap with a public, redistributable JGI-scale source is future work.

- [Download the immutable processed dataset on Hugging Face](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC/tree/bd38448d50d8f426d7b9bd4410b53159ea001259).
- Read the full [data contract and provenance](docs/DATA.md).

## Evaluation

**The main hill-climbing axis is full long-range contact precision at L
(P@L)** over the frozen 20,775-chain population. Training and validation MLM
losses are diagnostics; they do not override a P@L regression.

| Released model | ESMC paper P@L-LR (95% CI) | Our full 20,775-chain P@L |
|---|---:|---:|
| ESMC-300M | 0.552 ± 0.002 | 0.5387 |
| ESMC-600M | 0.589 ± 0.002 | 0.5803 |
| ESMC-6B | 0.725 ± 0.002 | running; exact result pending |

Both evaluations start from the RCSB Protein Data Bank with the 2024-02-28
snapshot date, contain 20,775 chains, and follow the same published construction
rules. Biohub does not publish its ordered chain manifest or raw-file digests,
so exact item-for-item identity cannot be verified. All structural labels for
our P@L evaluation come from this frozen RCSB PDB source, and these chains are
included in the protected union used to decontaminate the training corpus.
[RCSB PDB](https://www.rcsb.org/); [ESMC](https://doi.org/10.64898/2026.06.03.729735).

We also developed **P-CORE**, a broader frozen-representation evaluation suite.
The currently reported P-CORE score combines four trusted tasks:

- Remote homology, balanced accuracy — [DeepSF](https://doi.org/10.1093/bioinformatics/btx780), [TAPE](https://proceedings.neurips.cc/paper/2019/hash/37f65c068b7723cd7809ee2d31d7861c-Abstract.html).
- Secondary structure, residue macro-F1 — [NetSurfP-2.0](https://doi.org/10.1002/prot.25674), [CB513](https://pubmed.ncbi.nlm.nih.gov/10081963/).
- DeepLoc2 localization, macro average precision — [DeepLoc 2.0](https://doi.org/10.1093/nar/gkac278).
- FLIP2 Hydrophobic Core low-to-high fitness, Spearman correlation — [FLIP2](https://doi.org/10.64898/2026.02.23.707496).

The evaluation program separately reports:

- Held-out MLM negative log-likelihood and perplexity — [data provenance](docs/DATA.md).
- Enzyme Commission macro average precision, quarantined — [DeepFRI](https://doi.org/10.1038/s41467-021-23303-9), [TorchProtein](https://doi.org/10.5281/zenodo.6622158).
- Human PPI average precision, quarantined — [Pan et al.](https://doi.org/10.1021/pr100618t), [PEER](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e467582d42d9c13fa9603df16f31de6d-Abstract-Datasets_and_Benchmarks.html).

The source-qualified P-CORE v0.5 alpha additionally covers:

- CATH 4.4 remote retrieval — [CATH 4.4](https://doi.org/10.1093/nar/gkae1087).
- PRING Human PPI, provisional — [PRING](https://github.com/SophieSarceau/PRING).
- FLIP2 engineering shift — [FLIP2](https://doi.org/10.64898/2026.02.23.707496).
- MegaScale stability — [MegaScale](https://doi.org/10.1038/s41586-023-06328-6).
- CAID3 disorder — [CAID3](https://doi.org/10.1002/prot.70045).
- CAFA5 molecular function, blocked because only 110 test proteins survived
  its preregistered identity screen — [CAFA5](https://doi.org/10.64898/2026.04.27.716980).

None of the alpha or quarantined tasks contributes to the current four-task
P-CORE score. Public development outcomes and open evaluation work are tracked
in [`.dev/LOG.md`](.dev/LOG.md).

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for dataset lineage, exact
splits, probe definitions, representation-panel results, confidence intervals,
and execution.

## AutoResearch

> Train an ESMC-300M-class sequence encoder from scratch under a fixed compute
> budget and improve the biological information exposed by its frozen
> representations.

Candidates may change the model, optimizer, loss, schedule, batching, kernels,
and other training-efficiency components. The processed corpus, tokenizer,
dependency lock, hardware budget, training clock, evaluation examples, probes,
and metrics remain fixed.

The AutoResearch baseline trains Stage 1 from scratch for exactly one hour on
four NVIDIA L40S GPUs. The corpus, tokenizer, dependency lock, hardware class,
training clock, and full contact evaluator stay fixed. Each round tests one
focused change and keeps it only when P@L strictly improves; training and
validation loss remain required diagnostics.

| Model | P@L | Delta vs. original | Train loss | Validation loss | Steps | Model tokens (M) | Parameters (M) | Peak VRAM (GB) | Train (h) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original ESMC | 0.0925 | — | 2.723 | 2.699 | 5,883 | 358 | 333 | 35 | 1 |
| **AutoResearch-Codex-Round1** | **0.0960** | **+0.0036 (+3.87%)** | **2.720** | 2.704 | 5,682 | 346 | 333 | 37 | 1 |

The retained recipe adds learned residual/input routing, parameter-free
transformer RMSNorm, depth-scaled attention-output and FFN-down initialization,
and a final-20% linear learning-rate cooldown ending at 0.1× peak.

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
  author = {{Lumin Science}},
  title = {LuminBench Nano-ESMC: A Minimal Reproduction of ESMC Language-Model Training},
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
