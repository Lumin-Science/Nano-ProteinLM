# Baseline contract and results

This page preserves historical baseline workflows. The current starting recipe
is [Setting 3](../../../configs/default.yaml), with direct commands in
[Usage](../../../docs/USAGE.md) and the current [task definition](../../../tasks/171m-validation-loss.md).

The 171M recipes target small-budget experiments using the paper's 170M
scaling backbone ([Appendix A.1.4.1, Table S4](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1.full.pdf#page=29)).
For the original-size ESMC architectures, use the
[300M/600M reference configs](../../../configs/esmc/README.md).

This page separates baseline evidence from the evaluation definition. Every
number below is meaningful only with the frozen datasets, probes, metrics, and
aggregation in [`EVALUATION.md`](../../../docs/EVALUATION.md).

## Comparison ladder

### Simple comparators

Each downstream task retains its chance, prevalence, or frequency readout.
These comparators validate metric direction and prevent a superficially nonzero
raw score from being mistaken for useful representation quality; they are not
trainable model baselines. Null normalization and representation-panel
aggregation are defined in [`EVALUATION.md`](../../../docs/EVALUATION.md).

### Original reference workflow

`configs/archive/esmc-300m-original.yaml` is the checkpoint-compatible ESMC-300M
training recipe: 30 layers, width 960, 15 attention heads, and 332,997,184
parameters. It uses four GPUs, context length 512, 64 sequences per GPU, and a
14,400-second training-loop limit. This was the original public speedrun default.

Historical project campaigns establish compute scaling but used the earlier
9-million-protein materialization that is no longer a supported training
artifact. Their results are retained as scientific context, not presented as a
fresh reproduction of the current public release.

| Historical run | GPUs × training time | Steps | Sequences | Peak memory | Full contact P@L |
|---|---:|---:|---:|---:|---:|
| Project ESMC-300M Stage 1 | 4 × 4 h | 20,971 | 5.369 M | 35.65 GiB | 0.1013 |
| Project ESMC-300M Stage 1 | 4 × 16 h | 84,000 | 21.504 M | 37.23 GiB | 0.1636 |

The two runs use different seeds. Paired evaluation intervals compare their
fixed predictions and do not estimate training-seed uncertainty. This table is
the public record for these historical comparisons.

### Historical compute-matched workflow

`configs/archive/esmc-300m-current-best.yaml` is the opt-in, exact one-hour
AutoResearch incumbent; `configs/archive/autoresearch_300m_4xa100_1h.yaml` is its
campaign-named alias. That campaign used the original ESMC recipe as its
reference. Its incumbent keeps the ESMC-300M class while adding learned
residual/input routing, parameter-free transformer RMSNorm, depth-scaled
residual initialization, and a final-20% learning-rate cooldown. It has
332,823,484 parameters.

Under that campaign’s four-GPU, one-hour AutoResearch contract, the
incumbent progression was:

| Candidate | Full contact P@L | Status |
|---|---:|---|
| Previous incumbent re-baseline | 0.0896190356 | comparison baseline |
| Depth-scaled residual initialization | 0.0981775134 | retained |
| Final-20% learning-rate cooldown | 0.0987620524 | current incumbent |

These are development-selection results, not multi-seed release claims.
The experiment contract and retained changes are summarized in
[`AUTORESEARCH.md`](../../../docs/AUTORESEARCH.md); rejected candidates remain on the
`auto-research` branch.

### Validated ESMC-171M preset

[`configs/archive/autoresearch_171m_4xl40s_1h.yaml`](../../configs/archive/autoresearch_171m_4xl40s_1h.yaml)
records the R02 winner from `autoresearch-171m`, promoted in `f3293e4` and
documented in `9ec883b`. It has 170,559,856 parameters: 24 layers, width 768,
and 12 attention heads. It combines learned residual/input routing,
parameter-free RMSNorm, depth-scaled residual initialization, and RoPE base
20,000 with Muon for transformer matrices and AdamW for the remaining weights.
Muon learning-rate scales are 0.9 for attention and 0.75 for FFNs; its
weight-decay scale is 0.75. The base learning rate is 0.000326599 and weight
decay is 0.0183712. Training uses a 554-step warmup followed by constant LR,
Stage 1 context 512, and 64 sequences per GPU for one hour on four L40S GPUs.

Across matched seeds 42, 43, and 44, R02 recorded contact P@L of 0.1081,
0.1058, and 0.1091, with a reported mean ± sample SD of **0.1077 ± 0.0017**.
The reported improvement is +8.06% over the campaign's starting baseline,
which already used Muon and the retained architecture. This is not a
comparison against `configs/esmc-171m-original.yaml`.
The retained recipe is documented in the [historical recipe comparison](171m-adamw-muon-recipes.md).

The checked-in campaign record does not report validation-loss mean or SD,
and the per-seed validation-loss receipts are not included. The P@L standard
deviation must not be interpreted as validation-loss uncertainty.

## Released ESMC reference checkpoints

The strong public references are the original Biohub ESMC checkpoints, pinned
to revisions `a59b831…` (300M), `a7e8201…` (600M), and `45b0fa5…` (6B).
Their representation-panel and contact P@L results are kept beside the metric
and dataset definitions in [`EVALUATION.md`](../../../docs/EVALUATION.md).

## Fairness and acceptance caveats

- Undertrained project checkpoints and released pretrained checkpoints do not
  consume matched pretraining compute; the released models are capability
  references, not compute-matched competitors.
- Architecture candidates are compared only within the fixed one-hour
  AutoResearch contract. Historical four- and sixteen-hour results are not
  eligible in that selection loop.
- EC and Human PPI remain visible but quarantined because their repository
  reconstructions fail the preregistered discrimination checks.
- One checkpoint evaluation is not a release claim. Promotion requires a clean
  rebuild, repeated seeds, uncertainty, integrity receipts, and review of every
  trusted-task regression.
