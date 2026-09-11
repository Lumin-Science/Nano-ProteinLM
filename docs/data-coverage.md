# Data coverage when scaling training

Training now defaults to `data_resampling: error`. Before allocating model memory,
the trainer checks every source against the actual global batch and optimizer-step
budget, including `data_capacity_headroom: 0.01`. It records this calculation in
`DATA_COVERAGE.json` and `run_contract.json`. A larger total corpus is insufficient
if one source is too small for its mixture weight.

For batch 2,048 and 100,000 updates, each model consumes 204,800,000 sequence draws.
With the normalized Stage-1 mixture 36:11:54 and 1% headroom, preparation needs at
least 73,728,000 UniRef90, 22,528,000 MGnify and 110,592,000 OMG/IMG records. On the
pinned release, the smallest whole-shard prefixes contain 208,457,369 training
records (211 training shards), plus the same three complete validation shards.

Use a fresh data directory and run:

```bash
DATA_ROOT=/path/to/new/data bash runs/setup.sh --training-samples 206848000
```

The sample budget includes 1% headroom. Setup preserves the fixed source mixture;
for a different mixture, use `python -m nanoprotein.sharded_data --weights ...`
and plan per-source capacity from that recipe. Existing small directories are
never silently reused for a different shard selection.

Expected counts and headroom cannot guarantee stochastic source selection never
exhausts a rank. Therefore the sampler also refuses to start a second shuffled
pass. Rank partitions are disjoint, and stage transitions share each source's
sampler instead of restarting it. Together with the release's global exact
deduplication, this prevents repeating a protein within a fresh run. Similar
homologous proteins can remain; exact uniqueness does not imply independence.

Logs include global source draw counts and maximum source epoch indices. A
no-repeat run must keep every epoch index at zero. Time-only or token-only
budgets without a step ceiling get a runtime guard but cannot receive a full
advance sample-capacity guarantee. Supply `max_steps` for an advance check.

An intentionally repeated-data experiment must explicitly set
`data_resampling: allow`; its coverage receipt still reports insufficient data
and expected exposure counts. Validation sampling retains its established
behavior and is unaffected by training's exhaustion guard.

Same-layout continuation restores the saved source cursors and random states.
Legacy rank-partitioned no-repeat continuation refuses a changed GPU count or
missing sampler states. The full model and optimizer states remain portable.
The explicit global-sampler migration below preserves consumed-record history
when extending the corpus and changing the GPU layout.

## Global source epochs and explicit corpus migration

`data_sampler: global` generates the same global microbatch source assignments
and source record order on every rank, then assigns disjoint slices to GPUs.
Source cursors advance globally. A source completes its entire current pass
before a new permutation starts, including when exhaustion falls inside a batch.
Checkpoint state contains source RNG state, epoch/cursor, the migration origin,
and rank-local crop/RNG state. It does not store the large permutation arrays.

For controlled reuse, configure every source explicitly:

```yaml
data_sampler: global
data_resampling: per_source
data_source_resampling:
  uniref90: allow
  mgnify: error
  omg_img: allow
```

The capacity check enforces unique-data headroom for strict sources and reports
expected exposures for reused sources. Logs include `source_exposure_global`
with cumulative draws, unique records seen, repeated draws, and source epochs.
These counts include the parent run. Historical counts redistributed across
new ranks are accounting shares; they do not describe earlier physical GPU work.

Ordinary resume still requires the same data manifest and scientific recipe.
An append-only expansion from a legacy no-repeat checkpoint requires a receipt:

```bash
python -m nanoprotein.data_migration \
  --checkpoint /path/to/checkpoint-final.pt \
  --old-data-root /path/to/parent/data --data-root /path/to/expanded/data \
  --output /path/to/migration.json --seed 20260910
```

The verifier binds the parent checkpoint and both manifests, verifies the same
decontaminated release and unchanged validation, and compares every old index
entry and encoded residue with the expanded source prefix. It reconstructs
previously consumed identities from all old rank cursors and permutations.
Training uses the remaining old and newly added records before permitting reuse.
Pass `--resume-data-migration /path/to/migration.json` alongside `--resume` only
for that first conversion. Wrapped legacy histories and incompatible data are
rejected; validation/evaluation data are never added to training.

Subsequent global checkpoints resume directly on a different GPU count while
preserving global batch and global microbatch grouping. For example, 64×8×4
becomes 128×4×4. The next global record sequence is conserved; floating-point
reduction order and rank-local crops/masks can change, so cross-layout resume is
not bitwise identical. Same-layout resume restores rank RNG states exactly.
Each global-sampler run covers one training stage. An explicit [Stage 1-to-Stage 2 transition](stage2-continuation.md) preserves the model, full optimizer, RNG and per-source history while changing context, mixture and the decay clock. Strict sources can expand before their first repeat; their new queue excludes every previously consumed identity. Continuation capacity checks compare unused records with only the remaining updates.

See [the 400k Setting 3 recipe](../configs/setting3-nibi-b2048-400k.yaml) and
[the run record](../.dev/reports/nibi-setting3-b2048-400k-20260910/README.md).

## Nibi correction, September 8, 2026

The earlier 100k-step Nibi AdamW baseline and partially completed Setting 3 used
only 7,109,469 staged records. The baseline's 204.8M draws averaged 28.8 exposures
per record; they were not 204.8M unique proteins. The user requested stopping the
old Setting 3 at approximately 43,250 updates and restarting both models with
enough data. Preserve the earlier results as repeated-data experiments.

The replacement pair uses GPUs 0–3 for AdamW and 4–7 for Setting 3. Its intended
layout is microbatch 128 × four GPUs × four accumulation steps = 2,048, subject
to the recorded qualification trial. This preserves the prior 512-sequence
distributed microstep and Setting 3's sqrt-loss denominator size. Both start
from scratch and use the same corpus, seed, learning-rate/weight-decay settings,
1,000-step warmup, constant Stage-1 learning rate and 100k-step endpoint. Full
evaluation occurs every 10k steps; final checkpoints retain optimizer and sampler
states for continuation on four GPUs.
