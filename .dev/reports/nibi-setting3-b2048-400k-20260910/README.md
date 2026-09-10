# Nibi Setting 3 continuation: 100k to 400k Stage-1 steps

**Production launched successfully on September 10, 2026 at 12:36 a.m. Toronto**
in Slurm step **12162637.32**, allocation **12162637**, Nibi **g27**. Setting 3
continues from its verified 100k checkpoint on all eight H100s. The initial
local audit covers 660 additional updates, with finite losses/gradients and
preserved source history. All eight GPUs belonged to this run and showed
97–99% utilization at the GPU audit. See [launch verification](LAUNCH_VERIFIED.json)
and [GPU/preservation proof](launch/LAUNCH_GPU_AND_PRESERVATION.json).

The measured speed is **0.447 seconds/update**, projecting about **39.4 hours
from launch**, including 30 evaluations and an overhead allowance. The initial
finish estimate is **September 11 around 4 p.m. Toronto**, with roughly two hours
uncertainty. Monitoring is active every two hours. No production evaluation
endpoint has completed in this launch snapshot.

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

The [completed migration](launch/DATA_MIGRATION.json) verified every existing record index and encoded residue against
the expanded prefix. It reconstructs the consumed 100k history from the four
saved rank samplers, excludes those records from the rest of each first pass,
and partitions the global stream across eight GPUs. The prepared manifest SHA is
`52399ee87da1d4d6d019ebc9b232cbf48bfe038c9ee5f9ea65814b9c3b70a2f6`.
[Full 400k capacity verification](PRODUCTION_CAPACITY_VERIFIED.json) passed against
the actual prepared corpus. Source RNG/cursors and
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
[Qualification passed](launch/QUALIFICATION_PASSED.json): both full model/optimizer
restoration audits were exact, the new checkpoint continued correctly, and no
source repeated prematurely. Its full evaluation at step 100100 measured validation
loss **2.375535**, P@L **36.387%**, and 95% chain-bootstrap CI **36.147–36.632%**.
That is a trial result; production starts again from the original 100k checkpoint.
The first 20 common logged trial/production steps have identical source counts,
exposure histories and token counts.

The initial trial attempt exited before model training because its shortened
max-step limit retained the 400k schedule limit. Both trial limits were corrected;
the production recipe was unchanged. The failed startup is preserved under
`launch/qualification-attempts/schedule-limit-before-training/`; see the
[recovery receipt](launch/QUALIFICATION_RECOVERY.json). The passed trial and
production use clean frozen source commit
`bd0b754e817363bf5f5f9d48b2c482818a79559f`.

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

The source recipe, actual production config, project-storage recipe and source
bundle were hash-verified. The [local launch verifier](verify_launch.py) binds
the receipts to the recipe and the untouched parent checkpoint. The durable
recipe SHA-256 is
`afdeaad2f3810a2ad6fb0e86e1c57ba100a402c12e6157e8f406a7c7936bb43f`.
The parent 100k checkpoint remains preserved separately; the 200k/300k/400k
milestones are scheduled outputs, not checkpoints claimed to exist at launch.

The existing [100k comparison](../nibi-paired-unique-b2048-100k-20260909/README.md)
was published to main at `1c13852`. Later report commits record this launch;
they do not mutate the frozen running checkout.
