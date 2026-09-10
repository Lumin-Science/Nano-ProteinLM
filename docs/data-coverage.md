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
No-repeat continuation refuses a changed GPU count or missing sampler states,
because starting a fresh stream could repeat earlier proteins. The full model
and optimizer states remain portable; preserving unique-data history across a
changed GPU layout requires an additional sampler redistribution mechanism.

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
