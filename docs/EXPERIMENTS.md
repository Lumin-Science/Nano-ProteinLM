# Operational qualification

All trials used four A100 80GB PCIe GPUs, the same seed/data manifest, exact
ESMC-300M geometry, BF16, and 60 measured training seconds. These are systems
checks, not model-quality comparisons.

| Trial | Stage 1 / 2 microbatch | Model tokens | Sequences | Outcome |
|---|---:|---:|---:|---|
| Per-layer attention packing | 8 / 1 | 4,259,840 shaped; 1,662,828 residues | not recorded | rejected: pad rows still crossed every MLP |
| Whole-model packing baseline | 8 / 1 | 2,096,285 | 8,128 | passed |
| Larger microbatch | 32 / 8 | 2,821,267 | 11,072 | passed, +35% tokens |
| Selected | 64 / 16 | 4,521,293 | 17,408 | passed, 2.16x baseline tokens |
| `torch.compile(dynamic=True)` | 32 / 8 | 0 | 0 | rejected before step 1 by Torch AOT metadata failure |

The selected point reached up to 101,752 non-pad model tokens/s and 16.3% 6N MFU in
Stage 1; Stage 2 reached up to 46,097 tokens/s and 7.4% 6N MFU. MFU uses the
dense 312 BF16 TFLOP/s peak per A100 and therefore excludes attention's
quadratic FLOPs; it is an explicit, conservative operational proxy.

`torch.compile` failed inside AOTAutograd with `AttributeError: 'float' object
has no attribute 'meta'` on the dynamic packed graph. Eager execution is frozen
for the canonical pilot. The successful experiment configs remain under
`configs/experiments/`; the known-failing compile config is intentionally not
shipped as a runnable contract.

The exact 575,036,992-parameter ESMC-600M contract was separately qualified
with its conservative 4 / 1 microbatches and gradient accumulation of 8. It
completed both context stages in 61.3 measured seconds, processing 794,061
model tokens across 3,040 sequences without an out-of-memory error. This is a
runnability check only; the 300M tier remains the canonical speedrun target.

The first 1,800-second canonical attempt processed 135,114,643 model tokens but
exposed a DDP stopping race before the final checkpoint: workers compared
rank-local clocks, so one worker could leave the loop while another entered one
more gradient collective. The corrected trainer synchronizes a maximum elapsed
wall clock across ranks at every loop boundary and separately max-reduces step
compute duration for its throughput receipt. This makes stage and stop decisions
identical on all ranks; the incomplete attempt is retained as a failure receipt
and is not reported as a completed speedrun.

The corrected canonical run completed from clean commit `fd9fc61` on four A100
80GB PCIe GPUs. Stage 1 crossed at 1,200.317 measured seconds and optimizer step
1,756, with 111,748,787 non-pad model tokens across 449,536 sequences
(93,099 tokens/s on average). The full run stopped at 1,800.024 seconds and step
2,958, with 135,353,999 model tokens, 134,301,071 residues, and 526,464
sequences (75,196 tokens/s cumulatively). Stage 2 therefore contributed
23,605,212 tokens in 599.707 seconds (39,361 tokens/s). Peak logged throughput
was 102,843 / 52,745 tokens/s for Stages 1 / 2. The stage and final checkpoint
SHA-256 digests begin `a213518e` and `5ed58935`; full digests live in the compact
result receipt. Evaluation is recorded separately because it is outside the
30-minute training clock.

## First production-gated 300M campaign

The next run is `configs/esmc_300m_stage1_4xa100_4h.yaml`: ESMC-300M, Stage 1
only, context 512, 21,000 optimizer steps on four A100s, global batch 256, and
2,100 warmup steps (exactly 10% of the target). The prior measured Stage-1 rate
projects 3:59:15 of training, while 14,400 seconds remains a fail-safe cap.
Normal completion is step-based, and a cap-limited run is reported as
incomplete.

Do not run this recipe on `pilot-v1`. That manifest excludes exact evaluation
sequence hashes but records `homology_exclusion: false`. The trainer fails
closed unless the production manifest explicitly certifies homology exclusion
against every P-CORE split and the complete contact manifest. After training,
`runs/stage1_300m_4xa100_4h.sh` evaluates the final checkpoint with held-out MLM,
all six exact P-CORE tasks with 10,000 bootstrap replicates, and all 20,775
contact chains. Evaluation is restartable and outside the four-hour training
budget. The release evaluation uses all four GPUs without a surrogate: three
deterministic contact-chain shards run alongside one exact P-CORE embedding
job; the six frozen P-CORE probe/bootstraps then run as restartable parallel CPU
tasks, and the contact merger performs the 5,000-replicate bootstrap over the
complete 20,775-chain row set.

## Evaluation runtime correction

The first full-profile attempt proved operationally unsuitable as a speedrun
gate. Contact scoring took about seven minutes, P-CORE embedding took about 42
minutes, and the old cache wrote protein and residue arrays for every one of
108,215 sequences (216,431 files, about 122 GB). After the cache was resumed,
the serial secondary-structure probe alone remained in four full-residue LBFGS
fits for more than one hour. The evaluator was interrupted without cancelling
the allocation; its content-addressed cache was retained.

The corrected default is `pcore-diagnostic-v1`, not a partial P-CORE score. It
retains remote homology, human PPI, and FLIP2 fitness, uses no bootstrap, runs
the three CPU probes concurrently with independent ten-minute limits, and never
forms an aggregate. Stage and final checkpoints use two GPUs concurrently.
Embedding is restricted to the 55,977 sequences needed by those tasks and
packs windows from different proteins into the same residue-budget batch. On
the resumed timing run, the final checkpoint reused 4,145 interrupted entries
and wrote the remaining 51,832 means; it reached the probe phase 156 seconds
after process launch, including checkpoint load and cache scan. Exact full
P-CORE remains available as a release profile, with residue arrays restricted
to the 11,411 secondary-structure sequences.

The completed resumed run evaluated stage and final concurrently in 528.71 wall
seconds. Stage/final diagnostic times were 414.26/513.41 seconds and both had
3-of-3 coverage. Remote homology balanced accuracy was 0.0682/0.0724, human-PPI
average precision 0.8519/0.8574, and FLIP2 Spearman correlation 0.3162/0.3150.
The earlier contact components took about seven to eight minutes concurrently;
combining measured phases projects a roughly 16-minute cold routine evaluation,
still below training wall time. These are diagnostic directions, not a P-CORE
aggregate or a statistically powered model comparison.
