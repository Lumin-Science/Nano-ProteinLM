# Evaluation contract

Training checkpoints are evaluated on three levels:

1. Held-out sequence-mean MLM NLL on hash-disjoint cluster representatives.
2. P-CORE v0.2, using the existing six frozen representation tasks and the
   current content-addressed embedding/probe implementation.
3. Long-range contact P@L using all-layer/all-head symmetrized attention maps,
   a logistic probe trained on the frozen 20 structures, Cβ distance below 8 Å
   (Cα for glycine), sequence separation at least 24, and top-L precision.

The standard speedrun uses a predeclared P@L subset after fitting the exact
20-chain probe. `EVAL_PROFILE=full` adds all 108,215 P-CORE sequences. The full
20,775-chain contact report is a release evaluation rather than a routine
speedrun gate.

The tmoss runner currently imports the frozen benchmark implementations from
the adjacent `AutoResearch_ESMC` source checkout while keeping benchmark data
outside this repository. This makes the local pilot runnable without duplicating
an evaluator that is still under repair. Before a public release, the repaired
evaluator must be versioned as an immutable dependency (or vendored with its
tests and provenance) so a fresh clone does not depend on that sibling path.

## Quarantined task

Enzyme Commission remains executable but is not trusted for model-selection
claims. Released ESMC-300M and 600M score about 71 skill, while ESMC-6B collapses
to 1.607. A dedicated audit verified split disjointness, cache coverage, finite
and non-degenerate 6B embeddings, and label geometry, without explaining the
collapse. Reports therefore carry an explicit `trusted_for_model_selection:
false` marker until the probe path is repaired and independently reproduced.
