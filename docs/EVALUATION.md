# Evaluation contract

Training checkpoints can be evaluated on three levels:

1. Held-out sequence-mean MLM NLL on hash-disjoint cluster representatives.
2. Frozen representation probes, including P-CORE-Q4 v0.3 for full reports.
3. Long-range contact P@L using all-layer/all-head symmetrized attention maps,
   a logistic probe trained on the frozen 20 structures, Cβ distance below 8 Å
   (Cα for glycine), sequence separation at least 24, and top-L precision.

AutoResearch uses only the full frozen 20,775-chain P@L result for model
selection. MLM and P-CORE results may be reported for completed reference
checkpoints, but they cannot keep or discard an AutoResearch candidate. Human
PPI remains quarantined from model selection.

`EVAL_PROFILE=full` runs all six exact v0.2 probe contracts and reduces the four
trusted tasks to P-CORE-Q4 v0.3. It
embeds protein means for all 108,215 sequences but writes residue embeddings
only for the 11,411 secondary-structure sequences. This removes the former
122 GB all-sequence residue cache without changing any full-suite task metric.
The expensive six tasks run as atomic, restartable taskwise subprocesses with
bounded parallelism (six tasks with four probe threads each on the 256-thread
tmoss host), followed by a digest-checked exact reduction. Secondary structure
still performs four full-residue LBFGS fits and is not suitable for a short
training gate. The full 20,775-chain contact report is likewise a release
evaluation. For the final 300M checkpoint, three deterministic contact shards
run concurrently with the exact P-CORE embedding job on the four A100s. The
contact merger restores the global SHA-ranked chain order and performs the same
5,000-replicate chain bootstrap over all 20,775 rows; sharding does not change
the metric or reduce its data.

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
outside this repository. Before a public release, the repaired
evaluator must be versioned as an immutable dependency (or vendored with its
tests and provenance) so a fresh clone does not depend on that sibling path.

## Quarantined tasks

Enzyme Commission and Human PPI remain executable and their raw metrics are
always reported, but neither may influence model selection. EC has an unresolved
cross-scale anomaly: released ESMC-300M and 600M score about 71 skill while 6B
collapses to 1.607 despite a clean integrity audit. Human PPI has only 237 test
pairs in the current reconstruction, released-model scale ordering is
non-monotonic, and the four-hour undertrained checkpoint (0.7979 AP) sits close
to released ESMC-300M (0.8155 AP). This is inadequate discrimination for a
promotion gate.

Reports therefore use the versioned `pcore-v0.3-q4` selection aggregate over
remote homology, secondary structure, DeepLoc2, and FLIP2. They carry a trust
record for every task and retain the old six-task number only as
`legacy_pcore_v0_2`. Human PPI can return only after a preregistered replacement
adds substantially more family-disjoint test pairs, hard negatives, a leakage
audit, and scale/checkpoint ranking validation. EC requires independent
reproduction and repair before reinstatement.

## P-CORE Next research track

The proposed CATH, CAFA5-MF, PRING, MegaScale, and CAID3 additions have zero
selection weight until they pass the benchmark-qualification program in
[`plans/pcore-next/FORMULATION.md`](../plans/pcore-next/FORMULATION.md).
P-CORE-Q4 remains authoritative throughout feasibility work, falsification
pilots, full qualification, aggregation study, and shadow deployment.

The concrete proposed successor—usable now as the implementation target—is
[`PROPOSED_PCORE_V05.md`](PROPOSED_PCORE_V05.md), with a machine-readable
contract at [`configs/pcore_v05_proposed_q9.yaml`](../configs/pcore_v05_proposed_q9.yaml).
