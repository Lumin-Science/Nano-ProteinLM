# Baseline contract and results

This page separates baseline evidence from the evaluation definition. Every
number below is meaningful only with the frozen datasets, probes, metrics, and
aggregation in [`EVALUATION.md`](EVALUATION.md) and
[`EVALUATION_DATASETS.md`](EVALUATION_DATASETS.md).

## Comparison ladder

### Simple comparators

Each downstream task retains its chance, prevalence, or frequency readout.
P-CORE converts raw performance to null-normalized task skill before taking the
trusted four-task geometric mean. These comparators validate metric direction
and prevent a superficially nonzero raw score from being mistaken for useful
representation quality; they are not trainable model baselines.

### Original local workflow

`configs/esmc-300m-original.yaml` is the checkpoint-compatible ESMC-300M
training recipe: 30 layers, width 960, 15 attention heads, and 332,997,184
parameters. It uses four GPUs, context length 512, 64 sequences per GPU, and a
14,400-second training-loop limit. This is the default public speedrun and the
simple model-training baseline for future production comparisons.

Historical local campaigns establish compute scaling but used the earlier
9-million-protein materialization that is no longer a supported training
artifact. Their results are retained as scientific context, not presented as a
fresh reproduction of the current public release.

| Historical run | GPUs × training time | Steps | Sequences | Peak memory | P-CORE-Q4 | Full contact P@L |
|---|---:|---:|---:|---:|---:|---:|
| Local ESMC-300M Stage 1 | 4 × 4 h | 20,971 | 5.369 M | 35.65 GiB | 25.4609 | 0.1013 |
| Local ESMC-300M Stage 1 | 4 × 16 h | 84,000 | 21.504 M | 37.23 GiB | 30.6609 | 0.1636 |

The two runs use different seeds. Paired evaluation intervals compare their
fixed predictions and do not estimate training-seed uncertainty. Full details
remain in `dev/report/`.

### Current compute-matched workflow

`configs/esmc-300m-current-best.yaml` is the opt-in AutoResearch incumbent. It
keeps the ESMC-300M class while adding learned residual/input routing,
parameter-free transformer RMSNorm, depth-scaled residual initialization, and
a final-20% learning-rate cooldown. It has 332,823,484 parameters.

Under the current four-GPU, one-hour, decontaminated AutoResearch contract, the
incumbent progression was:

| Candidate | Full contact P@L | Status |
|---|---:|---|
| Previous incumbent re-baseline | 0.0896190356 | comparison baseline |
| Depth-scaled residual initialization | 0.0981775134 | retained |
| Final-20% learning-rate cooldown | 0.0987620524 | current incumbent |

These are development-selection results, not multi-seed release claims.
Architecture provenance and rejected candidates are documented in
[`ARCHITECTURE.md`](ARCHITECTURE.md) and on the `auto-research` branch.

## Released ESMC references under the local protocol

The strong public references are the original Biohub ESMC checkpoints, pinned
to revisions `a59b831…` (300M), `a7e8201…` (600M), and `45b0fa5…` (6B). The
values below were recomputed through this repository's exact embeddings,
splits, probes, and metrics; they were not copied from the ESMC paper.

| Task / metric | Trust | ESMC-300M | ESMC-600M | ESMC-6B |
|---|---|---:|---:|---:|
| Remote homology / balanced accuracy | trusted | 0.1159 | 0.1152 | 0.1186 |
| Secondary structure / residue macro-F1 | trusted | 0.8315 | 0.8407 | 0.8780 |
| Enzyme Commission / macro average precision | quarantined | 0.7174 | 0.7096 | 0.0237 |
| DeepLoc2 / macro average precision | trusted | 0.6442 | 0.6568 | 0.6942 |
| Human PPI / average precision | quarantined | 0.8155 | 0.8026 | 0.8345 |
| FLIP2 Hydro low-to-high / Spearman correlation | trusted | 0.4132 | 0.4276 | 0.4616 |
| **P-CORE-Q4 v0.3** / trusted four-task geometric mean | **selection** | **37.4847** | **38.1553** | **40.6011** |
| Legacy P-CORE v0.2 / six-task geometric mean | diagnostic | 45.5503 | 45.6899 | 25.7677 |

Contact P@L uses attention maps rather than final-layer frozen embeddings. The
diagnostic population is one fixed SHA-ranked 1,024-chain sample; the full
population contains 20,775 chains.

| Model | Local 1,024-chain P@L | Local full 20,775-chain P@L |
|---|---:|---:|
| ESMC-300M | 0.5340 [0.5229, 0.5445] | 0.5387 |
| ESMC-600M | 0.5778 [0.5668, 0.5881] | 0.5803 |
| ESMC-6B | 0.7097 [0.6997, 0.7195] | not completed |

The missing 6B full-manifest value is not estimated from the diagnostic sample.
The [ESMC publication](https://doi.org/10.64898/2026.06.03.729735) reports a
different P@L-LR protocol and values of 0.552, 0.589, and 0.725 for the three
scales. Dataset and protocol differences prohibit substituting either column
for the other.

## Fairness and acceptance caveats

- Local undertrained checkpoints and released pretrained checkpoints do not
  consume matched pretraining compute; the released models are capability
  references, not compute-matched competitors.
- Architecture candidates are compared only within the fixed one-hour
  AutoResearch contract. Historical four- and sixteen-hour results are not
  eligible in that selection loop.
- EC and Human PPI remain visible but quarantined because their local
  reconstructions fail the preregistered discrimination checks.
- One checkpoint evaluation is not a release claim. Promotion requires a clean
  rebuild, repeated seeds, uncertainty, integrity receipts, and review of every
  trusted-task regression.
