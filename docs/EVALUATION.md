# Evaluation contract

Training checkpoints are evaluated on three levels with two distinct operating
profiles:

1. Held-out sequence-mean MLM NLL on hash-disjoint cluster representatives.
2. A bounded three-task representation diagnostic for routine speedruns, or
   exact P-CORE v0.2 for release evaluation.
3. Long-range contact P@L using all-layer/all-head symmetrized attention maps,
   a logistic probe trained on the frozen 20 structures, Cβ distance below 8 Å
   (Cα for glycine), sequence separation at least 24, and top-L precision.

The default speedrun uses a predeclared P@L subset after fitting the exact
20-chain probe and runs the remote-homology, human-PPI, and FLIP2 fitness task
metrics concurrently. It embeds only the 55,977 required sequences, stores only
protein means, skips bootstrap, and limits each probe to ten minutes. A timeout
or failure is reported as partial coverage; the diagnostic has no aggregate and
must never be reported as P-CORE. Stage and final checkpoints are evaluated on
two GPUs concurrently when available. Embedding windows from different proteins
share a residue-budget GPU batch; pooling, deterministic long-sequence windows,
and content-addressed cache values otherwise retain the frozen evaluator's
contract.

`EVAL_PROFILE=full` remains the exact six-task P-CORE v0.2 release profile. It
embeds protein means for all 108,215 sequences but writes residue embeddings
only for the 11,411 secondary-structure sequences. This removes the former
122 GB all-sequence residue cache without changing any full-suite task metric.
The expensive six tasks should ultimately run as restartable task-parallel jobs;
secondary structure in particular performs four full-residue LBFGS fits and is
not suitable for a 30-minute training gate. The full 20,775-chain contact report
is likewise a release evaluation.

Component receipts (`VALIDATION_MLM.json`, `CONTACT.json`, diagnostic embedding,
and per-task JSON) are written atomically. A later failure therefore does not
erase completed work, and the runner reuses completed MLM/contact components on
restart. Output roots are checkpoint-specific; do not point a different
checkpoint at an existing evaluation directory.

## Next evaluator work

The next safe speed improvement is to calibrate one frozen regularization value
per diagnostic task on released reference models, then confirm that rankings are
unchanged across archived checkpoints. That would remove the current four-value
grid from every routine checkpoint without silently redefining the release
benchmark. Secondary structure needs a separate validation study comparing a
bounded residue sample or SGD/ridge surrogate against the full LBFGS task across
multiple model scales. Until rank preservation is demonstrated, the surrogate
may be reported only as another diagnostic. The many-small-file NumPy cache
should also move to content-addressed shards or LMDB after byte/numeric parity
tests; this is primarily a metadata and storage optimization.

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
