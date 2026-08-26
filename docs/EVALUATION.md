# Evaluation contract

Training checkpoints can be evaluated on three levels:

1. Held-out sequence-mean MLM NLL on hash-disjoint cluster representatives.
2. Frozen representation probes, including P-CORE-Q4 v0.3 for full reports.
3. Long-range contact P@L using all-layer/all-head symmetrized attention maps,
   a logistic probe trained on the frozen 20 structures, Cβ distance below 8 Å
   (Cα for glycine), sequence separation at least 24, and top-L precision.

Exact dataset lineage, source publications, split roles, and counts are listed
in [`EVALUATION_DATASETS.md`](EVALUATION_DATASETS.md).

The full frozen 20,775-chain P@L result is the separate production promotion
axis. MLM and P-CORE remain reportable reference metrics, and Human PPI remains
quarantined from promotion decisions. Experiment-specific selection rules live
under `dev/`.

## Released ESMC baselines on this evaluation

The table below reports the original released Biohub checkpoints through this
repository's frozen embedding, split, probe, and metric implementations. These
are local reproductions, not values copied from the ESMC paper. Checkpoints are
pinned to Biohub revisions `a59b831…` (300M), `a7e8201…` (600M), and
`45b0fa5…` (6B). All six raw probes remain visible, while only the four rows
marked **trusted** enter P-CORE-Q4 v0.3.

| Task / metric | Trust | ESMC-300M | ESMC-600M | ESMC-6B |
|---|---|---:|---:|---:|
| Remote homology / balanced accuracy | **trusted** | 0.1159 | 0.1152 | 0.1186 |
| Secondary structure / residue macro-F1 | **trusted** | 0.8315 | 0.8407 | 0.8780 |
| Enzyme Commission / macro average precision | quarantined | 0.7174 | 0.7096 | 0.0237 |
| DeepLoc2 / macro average precision | **trusted** | 0.6442 | 0.6568 | 0.6942 |
| Human PPI / average precision | quarantined | 0.8155 | 0.8026 | 0.8345 |
| FLIP2 Hydro low-to-high / Spearman correlation | **trusted** | 0.4132 | 0.4276 | 0.4616 |
| **P-CORE-Q4 v0.3** / four-task null-normalized geometric mean | **selection** | **37.4847** | **38.1553** | **40.6011** |
| Legacy P-CORE v0.2 / six-task geometric mean | diagnostic | 45.5503 | 45.6899 | 25.7677 |

Contact P@L is reported separately because it probes attention maps rather than
final-layer frozen embeddings. The full column uses the exact 20,775-chain
manifest. The diagnostic column uses the same frozen SHA-ranked 1,024-chain
subset for every model and includes its 95% chain-bootstrap interval.

| Model | Local 1,024-chain P@L | Local full 20,775-chain P@L |
|---|---:|---:|
| ESMC-300M | 0.5340 [0.5229, 0.5445] | 0.5387 |
| ESMC-600M | 0.5778 [0.5668, 0.5881] | 0.5803 |
| ESMC-6B | 0.7097 [0.6997, 0.7195] | not completed |

The missing 6B full-manifest value is left explicit rather than estimated from
the 1,024-chain diagnostic. Protocol and dataset differences also mean these
local contact values should not be substituted for the ESMC paper's published
P@L-LR column.

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

## Production acceleration

The production speed improvements already retained on `main` are cross-protein
residue-budget batching, secondary-structure-only residue caches, bounded
parallel probe processes, three-way contact sharding, deterministic global
P@L merge, and atomic restartable receipts. These change execution only; the
frozen examples, probe, row ordering, metric, and bootstrap remain unchanged.

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
