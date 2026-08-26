# P-CORE v0.5-alpha validation report

Date: 2026-08-22  
Protocol: `pcore-v0.5-alpha-q9`  
Status: **source-qualified alpha; released-model campaign complete**  
Selection status: **not production P-CORE; q9 aggregate prohibited**

## Executive result

The proposed evaluator is executable, versioned, and substantially faster than
the earlier P-CORE implementation, but the nine-task proposal does not pass its
own source qualification unchanged. The frozen CAFA5 MF no-knowledge hard set
contains only 110 proteins after the 30%-identity screen, below the
preregistered minimum of 500. CAFA therefore receives no model score. PRING is
executable on its full 64,038-pair test population after an independent 30%
identity screen, but remains provisional because the official release provides
one negative realization rather than the proposed ten. No missing value is
imputed and no q9 ranking is reported.

Five new tasks are runnable now: CATH v4.4 remote retrieval, PRING Human BFS,
the bounded FLIP2 shift panel, MegaScale family-screened single-mutant
stability, and CAID2-to-CAID3 temporal disorder. The previous trusted Q4 is
dual-reported for continuity, under its original protocol, and is never mixed
into the new alpha aggregate.

All three released ESMC scales completed on the same corrected receipt. Zero of
the five runnable tasks increases strictly from 300M to 600M to 6B. Only the
MegaScale 6B-minus-300M paired interval excludes zero; the apparent PRING
6B-minus-600M increase is not qualifying because PRING's bootstrap is
non-centered and the source has only one negative realization. The panel is
therefore useful as an auditable diagnostic profile, but it fails promotion to
a model-selection core in this campaign.

## Frozen artifacts

| Artifact | SHA-256 / pin |
|---|---|
| `uv.lock` | `dd2c994698b42ddf7f2d298179d82e220d86c62563e361aa4d24453a78cc8f18` |
| source registry | `51fa8a340cb0d40315419718f71c75bf0280d93259398f404633cb38500ccd09` |
| dataset receipt | `621701370c1aa471b2a07514c39282b4a350169766c67c03cacca0094621cd07` |
| runnable sequence index | `ddd31023ac8820addee4f095e4f2a482ef0dbfc15e89fb960cf79254b6d1090b` |
| three-scale comparison | `4faaab75d3dabdaad1c24340678627a466918e4c684885c0a4fe2e9ef55be9b2` |
| Biohub Transformers | commit `ef32577f55da19a4989cd7b22e004dc43a4998cb` |
| MMseqs2 | commit `f71d0a6b57c19e3a8954d8557f065c0fe1f46142` |
| uv runners | `0.9.26` on labp; `0.11.31` on tmoss; both `uv sync/run --frozen` |

The exact dependency pins used on `labp` include Python 3.11.15, PyTorch
2.8.0/CUDA 12.8, NumPy 2.3.2, PyArrow 25.0.1, scikit-learn 1.7.1, and the
pinned Biohub Transformers fork reporting version 4.57.6. At validation time,
the then-current local verification suite recorded 176 passing checks with two
expected synthetic no-positive-class warnings.

## Probe 1 — authoritative sources and population reconstruction

**Assumption.** Every proposed task can be reconstructed from an authoritative,
versioned artifact with a sufficiently large population and a model-independent
split.

**Probe.** Acquire and checksum the primary artifacts; reconstruct splits;
screen PRING, MegaScale, and CAFA with one pinned MMseqs2 implementation; hash
every selected row and unique sequence before loading any released model.

**Observation.**

| Task | Frozen rows | Unique sequences | Frozen split counts | Source decision |
|---|---:|---:|---|---|
| CATH 4.4 remote retrieval | 11,180 | 11,177 | 2,631 query / 8,549 gallery | runnable |
| PRING Human C3-30 | 167,321 pairs | 9,775 | 82,621 / 20,662 / 64,038 | provisional |
| FLIP2 shift | 104,946 | 68,888 | 39,948 / 10,161 / 54,837 | runnable bounded alpha |
| MegaScale family30 ddG | 88,172 | 88,468 including WT | 50,000 / 10,000 / 28,172 | runnable bounded alpha |
| CAID2 → CAID3 Disorder-PDB | 666 proteins | 666 | 275 / 72 / 319 | runnable temporal alpha |
| CAFA5 MF NK30 hard | 78,747 | 77,742 | 78,637 train / **110 test** | **blocked: below 500** |

The full source index has 247,370 sequences. Excluding sequences used only by
the blocked CAFA task leaves 178,819 executable sequences. Source provenance is
anchored in the [official CATH downloads](https://www.cathdb.info/download),
[PRING repository](https://github.com/SophieSarceau/PRING),
[FLIP2 site](https://flip.protein.properties/),
[RosettaCommons MegaScale dataset](https://huggingface.co/datasets/RosettaCommons/MegaScale),
[CAID challenge results](https://caid.idpcentral.org/challenge/results), and
[official CAFA5 final-evaluation archive](https://zenodo.org/records/20186533).

For PRING, 189 non-test nodes shared a 30%-identity cluster with a test node;
removing them dropped 3,203 train and 794 validation pairs but no test pair.
Its AUPRC interval resamples both endpoint identity clusters and weights each
pair by the product of the two sampled cluster weights; it is therefore
orientation-invariant rather than grouping only the first listed protein.
For MegaScale, wild type was reconstructed by reversing the declared single
mutation; one non-test parent shared a 30%-identity cluster with a test parent
and was excluded before sampling. For CAFA, 431 of 541 MF no-knowledge targets
matched an MF-labelled training sequence at at least 30% identity and 80% query
coverage, leaving 110.

For CAID, an explicit temporal-overlap audit found `DP02732` unchanged in both
CAID2 and CAID3. Its CAID2 training row was removed before any model result was
read; the CAID3 test row remains. The task now has zero exact sequence or
identifier overlap from CAID2 into CAID3.

Post-build role checks are also zero for PRING fit-versus-test nodes, 30%-identity
clusters, and unordered pairs; for MegaScale fit-versus-test parents, 30%
clusters, and mutant sequences; and for FLIP2 fit-versus-test sequences within
every landscape. FLIP2 intervals hierarchically resample its seven datasets and
then the 16 landscapes within sampled datasets; they represent panel
uncertainty, not per-variant assay uncertainty.

**Verdict.** Partially falsified. Five new tasks are runnable, PRING is
provisional, and CAFA is blocked.

**Consequence.** The alpha report contains raw runnable-task results only. It
does not contain or imply a nine-task P-CORE score. CAFA must be replaced or
reframed prospectively; its identity threshold is not relaxed after this
observation.

## Probe 2 — evaluator, environment, and acceleration

**Assumption.** The evaluator respects the frozen readout and can finish a
three-scale panel without simplifying a scientific task.

**Probe.** Unit-test the loaders, hashing, batching, retrieval metric, temporal
split, and bootstrap; compare accelerated ESMC-300M embeddings against the
immutable serial cache; measure a real 1,000-protein throughput pass on the
shared RTX 3090.

**Observation.** The focused v0.5 suite has 24 passing tests and the repository
suite has 171 passing tests. Loading the ESMC backbone rather than the masked-LM
wrapper gives an exactly identical one-sequence embedding: max absolute
difference 0.0 and cosine 1.0. Cross-protein bf16 batching changes reduction
order slightly: across eight proteins, maximum absolute difference was 0.00246
and minimum cosine was 0.999945. To avoid mixing numerical regimes, the alpha
campaign regenerates every runnable sequence uniformly with the batched path.

The 1,000-protein 300M pass, including model load and 1,000 atomic writes, took
10.49 seconds and 2.71 GB peak host RSS. The speedup comes from length-aware
cross-protein packing, requesting only the final backbone state, and skipping
both LM-head logits and the all-layer hidden-state tuple. No task, example, or
metric is simplified.

For 6B, a tested deterministic contiguous-sequence partition let four A100
workers write disjoint ranges into the same content-addressed store. The
complete 6B workflow---178,819 protein means, 666 residue embeddings, five
probes, 10,000 resamples, and reduction---then took 967.3 seconds (16:06.5)
with 17,465,588 kB maximum host RSS. The original one-A100 rate was about 15
proteins/s; the four-worker pass sustained roughly 94 proteins/s while the
longest proteins dominated and accelerated further as lengths shortened.

**Verdict.** Passed for execution, with the uniform-cache condition.

**Consequence.** Legacy serial embeddings are used only for legacy Q4
continuity. Every v0.5-alpha task uses a fresh, uniformly batched cache for each
released model.

## Probe 3 — released ESMC scale panel

**Assumption.** The runnable new tasks yield a stable and interpretable profile
for released ESMC-300M, ESMC-600M, and ESMC-6B under identical manifests.

**Probe.** Extract all 178,819 executable protein embeddings plus CAID residue
embeddings for each exact released checkpoint; run fixed `C=1` logistic,
`alpha=1` ridge, or parameter-free cosine retrieval; use 10,000 paired
biological-unit bootstrap resamples.

**Observation.** The corrected ESMC-300M report is hash-verified as
`29e13afa9b6f016056c40db97f231b44600ec4d021dc31c60bbbb0dab757fa40`
and binds the corrected dataset receipt. Its first pass completed before the
CAID overlap and PRING/FLIP bootstrap corrections reached `labp`; those old
partials were rejected. The corrected run resumed all 178,819 protein and 666
residue caches, recomputed every probe, and wrote new atomic partials. The
existing DeepLabCut service remained at 8,066 MiB and was not modified.

| New alpha task | ESMC-300M | ESMC-600M | ESMC-6B | Interpretation |
|---|---:|---:|---:|---|
| CATH 4.4 macro-H mAP | 0.1170 [0.1048, 0.1293] | 0.1067 [0.0953, 0.1183] | 0.1153 [0.1034, 0.1276] | 1,593 H groups |
| PRING Human AUPRC | 0.7035 [0.7255, 0.7542]\* | 0.7001 [0.7215, 0.7524]\* | 0.7113 [0.7323, 0.7622]\* | provisional; intervals non-centered |
| FLIP2 hierarchical score | 0.7823 [0.6765, 0.8164] | 0.7581 [0.7377, 0.8119] | 0.7805 [0.7489, 0.8127] | seven datasets / 16 landscapes |
| MegaScale median-parent Spearman | 0.6835 [0.5999, 0.7110] | 0.7267 [0.6481, 0.7534] | 0.7207 [0.6786, 0.7952] | 28 test parents |
| CAID3 macro-protein AP | 0.7292 [0.6931, 0.7635] | 0.7341 [0.6982, 0.7675] | 0.7340 [0.6999, 0.7663] | 233 AP-eligible proteins |
| CAFA5 MF NK30 | blocked | blocked | blocked | 110 < 500 |

\*PRING's symmetric network-bootstrap means are 0.7395, 0.7365, and
0.7466---respectively 0.0361, 0.0364, and 0.0354 above the 300M, 600M, and 6B
plug-in AUPRCs---so none of its percentile intervals covers its point estimate.
This is reported as a failed uncertainty diagnostic, not silently recentered
after observing the result. The PRING point estimates remain useful as
provisional diagnostics, but the current intervals are not qualification
intervals.

The 300M diagnostics are CATH H-recall@1 0.2113 (topology recall@1 0.3029),
PRING ROC-AUC 0.6730 and partner-recall@50 0.8481, MegaScale pooled test
Spearman 0.5686, and CAID3 pooled AP 0.9014 / MCC 0.6969. The five-task
descriptive median null-normalized gain is 0.4069 and its lower quartile is
0.2569; neither number is a selection score.

The 6B diagnostics are CATH H-recall@1 0.1866 (topology recall@1 0.2695),
PRING ROC-AUC 0.6854 and partner-recall@50 0.8491, MegaScale pooled test
Spearman 0.6344, and CAID3 pooled AP 0.8994 / MCC 0.6995. Its descriptive
median null-normalized gain is 0.4225 and lower quartile is 0.2523. FLIP2's
32-permutation random-null estimate varies slightly by model (0.7070, 0.7011,
0.7064), so these descriptive gain summaries are explicitly unsuitable for
cross-model ranking; the raw primary metrics and paired biological-unit
intervals are authoritative.

The 600M report is hash-verified as
`c452dccb366cf317919acfc0e1579c8b7d47c43f5e74cd0ec387510a81af9377`.
It completed in 1,214.17 seconds wall time with 4,128,324 kB maximum host RSS.
The paired 600M-minus-300M deltas are: CATH -0.0102
[-0.0198, -0.0007], PRING -0.0034 [-0.0118, 0.0060], FLIP2 -0.0242
[-0.0481, 0.0816], MegaScale +0.0432 [-0.0216, 0.0799], and CAID3
+0.0049 [-0.0119, 0.0219]. Only the CATH decline excludes zero. Thus two of
five point estimates increase with scale, but four of five paired intervals do
not establish a difference.

The paired 6B-minus-300M deltas are: CATH -0.0017
[-0.0135, 0.0100], PRING +0.0078 [-0.0020, 0.0162], FLIP2 -0.0018
[-0.0378, 0.1362], MegaScale +0.0372 [0.0064, 0.1171], and CAID3 +0.0047
[-0.0136, 0.0229]. Only MegaScale excludes zero. The paired 6B-minus-600M
deltas are CATH +0.0085 [-0.0023, 0.0196], PRING +0.0112
[0.0011, 0.0193], FLIP2 +0.0224 [-0.0343, 0.0545], MegaScale -0.0060
[-0.0267, 0.0682], and CAID3 -0.0001 [-0.0185, 0.0187]. The PRING interval
would exclude zero but cannot qualify while its centering diagnostic fails.

**Verdict.** ESMC-300M, ESMC-600M, and ESMC-6B all pass task execution,
checkpoint identity, and manifest integrity. The 6B report is hash-verified as
`1caa0bf25c8a3e73cdf2ca50f51722b31d0b0d2878be60fba103b673dfd12d97`.
PRING remains provisional and its current intervals fail centering. No runnable
task is strictly monotone across all three released scales.

**Consequence.** No q9 aggregate or model ranking is issued. PRING cannot
qualify for selection without a preregistered replacement uncertainty
estimator and ten negative realizations, evaluated in a later campaign rather
than tuned here. MegaScale is the only new task with a clear 6B-over-300M
signal in this panel; CATH, FLIP2, and CAID remain valuable domain diagnostics
but do not qualify as released-scale promotion gates from this evidence alone.

## Legacy P-CORE continuity (different protocol)

These values are the prior exact frozen evaluation and are not recomputed or
combined with v0.5-alpha. EC and the previous Human PPI reconstruction remain
quarantined; Q4 uses remote homology, secondary structure, DeepLoc2, and FLIP2
Hydro only.

| Metric | ESMC-300M | ESMC-600M | ESMC-6B |
|---|---:|---:|---:|
| Remote homology balanced accuracy | 0.1159 | 0.1152 | 0.1186 |
| Secondary-structure macro-F1 | 0.8315 | 0.8407 | 0.8780 |
| DeepLoc2 macro-AP | 0.6442 | 0.6568 | 0.6942 |
| FLIP2 Hydro Spearman | 0.4132 | 0.4276 | 0.4616 |
| **P-CORE-Q4 v0.3** | **37.4847** | **38.1553** | **40.6011** |
| Legacy six-task P-CORE v0.2, diagnostic | 45.5503 | 45.6899 | 25.7677 |

The released checkpoint identities are pinned to Biohub ESMC revisions
`a59b831…` (300M), `a7e8201…` (600M), and `45b0fa5…` (6B), consistent with the
[official Biohub ESM code](https://github.com/Biohub/esm/blob/main/esm/pretrained.py).

## Exact commands

```bash
UV_CACHE_DIR=.uv-cache uv lock
UV_CACHE_DIR=.uv-cache uv sync --frozen
UV_CACHE_DIR=.uv-cache uv run --frozen ruff check .
UV_CACHE_DIR=.uv-cache uv run --frozen ruff format --check .

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  -m autoresearch_esm.pcore_v05_sources \
  --registry configs/pcore_v05_alpha_sources.yaml \
  --output-root tmp/pcore-v05-alpha/raw

UV_CACHE_DIR=.uv-cache uv run --frozen python \
  -m autoresearch_esm.pcore_v05_data \
  --raw-root tmp/pcore-v05-alpha/raw \
  --output-root tmp/pcore-v05-alpha/processed \
  --mmseqs tmp/pcore-v05-alpha/tools/mmseqs/bin/mmseqs

# On labp, after hash verification and uv sync:
nohup ./hpc/labp_pcore_v05_alpha.sh 300m \
  > ../logs/300m.stdout 2> ../logs/300m.stderr &

nohup /usr/bin/time -v -o ../logs/600m.time \
  ./hpc/labp_pcore_v05_alpha.sh 600m \
  > ../logs/600m.stdout 2> ../logs/600m.stderr &

# On tmoss, after uv sync --frozen, focused tests, manifest hashes,
# and exact 6B revision prefetch, four disjoint workers inside job 10092:
bash tmoss_pcore_v05_alpha_6b_overlap.sh

# After all three hash-verified reports are local:
UV_CACHE_DIR=.uv-cache uv run --frozen python \
  -m autoresearch_esm.pcore_v05_compare \
  --report ESMC-300M=results/pcore-v05-alpha/300m/pcore-v0.5-alpha.json \
  --report ESMC-600M=results/pcore-v05-alpha/600m/pcore-v0.5-alpha.json \
  --report ESMC-6B=results/pcore-v05-alpha/6b/pcore-v0.5-alpha.json \
  --output results/pcore-v05-alpha/released-scale-comparison.json
```

The runner embeds to a content-addressed store, resumes atomically, executes one
partial per task, verifies the dataset receipt and task hash, and reduces only
exact task coverage. The completed 6B run reused the four idle GPUs in the
existing allocation through deterministic disjoint shards, with the same task,
index, and receipt hashes. The redundant one-A100 fallback job was canceled
only after the report hash, receipt, five-task coverage, and `q9=null` were
verified. The allocation itself remained `RUNNING` and was not stopped.

## Probe 4 — interpretation and next decisions

**Assumption.** A new task is useful only if its signal is not dominated by a
source defect, tiny hard population, one protein family, or one released
negative draw.

**Probe.** Inspect source gates, group counts, task-native diagnostics,
scale ordering, bootstrap intervals, and simple nulls before considering any
task for selection status.

**Observation.** Before model results, CAFA already fails size qualification
and PRING fails the ten-negative-realization requirement. CATH macro-averages H
groups, FLIP aggregates landscapes within datasets, MegaScale bootstraps test
parents, and CAID macro-averages test proteins so large groups cannot silently
dominate their primary intervals.

**Verdict.** The proposed core is correctly treated as an alpha validation
suite, not a finished replacement for Q4.

**Consequence.** Retain all raw diagnostics, but promote only MegaScale as a
candidate selection task from this run. Keep PRING provisional until ten
preregistered degree/identity-matched negative draws and a centered uncertainty
estimator exist; replace or reframe CAFA; enlarge the effective number of
FLIP2 datasets; and add deliberately undertrained anchors before freezing any
candidate production core or promotion thresholds.
