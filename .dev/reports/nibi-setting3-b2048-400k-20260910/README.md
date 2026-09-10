# Nibi Setting 3 continuation: 100k to 400k Stage-1 steps

The user authorized implementation and training on September 10, 2026, choosing
Setting 3 alone on all eight Nibi H100s. Data preparation is underway; production
must pass the recorded qualification gate before launch. This report will be
updated with the verified launch and finish estimate.

The [recipe](../../../configs/setting3-nibi-b2048-400k.yaml) continues the verified
100k checkpoint with 300k additional optimizer steps. It preserves global batch
2,048 as **64 sequences/GPU × 8 GPUs × 4 accumulation**, context 512, FA3/BF16,
the full Setting 3 architecture/optimizer, batch balance and sqrt loss. Base LR
is 5e-4 and base WD is 0.01; Muon attention/FFN LRs remain 4.5e-4/3.75e-4 and WD
0.0075. Warmup remains 1,000 steps already completed; Stage 1 continues at
constant LR without decay. The total endpoint is 400,000, not 400,000 additional.

Parent checkpoint:
`/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-paired-unique-b2048-100k-20260909/setting3/checkpoint-final.pt`

Parent SHA-256:
`d0f891cfd46e5517e1ce90a29f02bc88646a80023b7cf8e86fc3008311b66412`

Parent results: validation loss **2.375701**, P@L **36.567%** (95% chain-bootstrap
CI **36.328–36.815%**), with 204.8M distinct records and no repeats.
[The completed side-by-side comparison](../nibi-paired-unique-b2048-100k-20260909/README.md)
contains both full curves and checkpoint audits.

## Data and consumption history

The [data plan](DATA_PLAN.json) expands from 211 to 399 training shards: all 92
UniRef90 shards, 63 MGnify shards and all 244 OMG/IMG shards. It contains
427,516,221 unique training records. The added 188 shards require about 36.3 GB
compressed downloads. Preparation uses the same immutable release
`bd38448d50d8f426d7b9bd4410b53159ea001259` and preserves held-out validation.

| Source | Prepared unique records | Expected total draws at 400k | Expected passes | Policy |
|---|---:|---:|---:|---|
| UniRef90 | 74,175,974 | 291,992,079 | 3.936 | Reuse after completing a global pass |
| MGnify | 90,495,061 | 89,219,802 | 0.986 | Stop on exhaustion |
| OMG/IMG | 262,845,186 | 437,988,119 | 1.666 | Reuse after completing a global pass |

The migration verifies every existing record index and encoded residue against
the expanded prefix. It reconstructs the consumed 100k history from the four
saved rank samplers, excludes those records from the rest of each first pass,
and partitions the global stream across eight GPUs. Source RNG/cursors and
origin metadata are retained in subsequent checkpoints, including for future
eight-to-four-GPU continuation. The sampler reports unique and repeated counts
explicitly. See [data coverage and migration](../../../docs/data-coverage.md).

## Qualification and execution

Relevant local tests passed: 44 tests covering global sampler migration, 4→8→4
record conservation, legacy source history, optimizer restoration, distributed
loss, batch balance, training CLI and evaluation pauses. See [test output](local-tests.txt).

The [launch script](../../../runs/nibi_setting3_400k.sh) has three explicit modes:
`migration`, `qualification`, and `production`. All run in existing allocation
12162637 on g27. Qualification resumes the untouched 100k parent to 100200,
performs the full evaluation at 100100, then reloads the new checkpoint and
continues to 100210. Production starts again from the untouched 100k parent.
The [qualification verifier](verify_qualification.py) gates the production run.

Production evaluates every 10k steps from 110k through 400k using the same
4,096 MLM validation sequences and 20,775 contact chains, 16 contact shards and
5,000 chain-bootstrap replicates. Evaluation uses four GPUs while the eight
training ranks pause; its scientific protocol is unchanged. The rolling full
checkpoint and durable 200k/300k/400k checkpoints retain optimizer and global
sampler state. Monitoring runs every two hours once production is healthy.

| Artifact | Location |
|---|---|
| Remote run root | `/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-400k-20260910` |
| Frozen source | Remote root with `-run` appended |
| Node-local data | `/localscratch/muchenli.12162637.0/nano-nibi-setting3-400k-20260910` |
| Preserved prepared data | Remote root plus `/data` |
| Production output | Remote root plus `/full` |
| Durable checkpoints/recipe/source bundle | `/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-setting3-b2048-400k-20260910` |

The earlier eight-GPU run suggests 40–45 hours after production launch, including
evaluation pauses. Replace this planning estimate with the expanded-data
qualification measurement and actual launch time.
