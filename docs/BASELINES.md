# Baseline contract and results

This page separates baseline evidence from the evaluation definition. Every
number below is meaningful only with the frozen datasets, probes, metrics, and
aggregation in [`EVALUATION.md`](EVALUATION.md).

## Comparison ladder

### Simple comparators

Each downstream task retains its chance, prevalence, or frequency readout.
These comparators validate metric direction and prevent a superficially nonzero
raw score from being mistaken for useful representation quality; they are not
trainable model baselines. Null normalization and representation-panel
aggregation are defined in [`EVALUATION.md`](EVALUATION.md).

### Original reference workflow

`configs/esmc-300m-original.yaml` is the checkpoint-compatible ESMC-300M
training recipe: 30 layers, width 960, 15 attention heads, and 332,997,184
parameters. It uses four GPUs, context length 512, 64 sequences per GPU, and a
14,400-second training-loop limit. This is the default public speedrun and the
simple model-training baseline for future production comparisons.

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
The experiment contract and retained changes are summarized in
[`AUTORESEARCH.md`](AUTORESEARCH.md); rejected candidates remain on the
`auto-research` branch.

## Released ESMC reference checkpoints

The strong public references are the original Biohub ESMC checkpoints, pinned
to revisions `a59b831…` (300M), `a7e8201…` (600M), and `45b0fa5…` (6B).
Their representation-panel and contact P@L results are kept beside the metric
and dataset definitions in [`EVALUATION.md`](EVALUATION.md).

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
