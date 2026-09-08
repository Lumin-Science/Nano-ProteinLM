# Continuing the Nibi AdamW baseline

The Nibi baseline uses replicated DDP: every GPU has the entire model and AdamW
state. The checkpoint written by rank 0 contains all weights, AdamW first and
second moments, optimizer steps, training counters, the recipe, and a data-manifest
hash. It also contains each rank's RNG and sampler state. No optimizer-shard merge
or conversion is needed to change from eight to four GPUs.

The new run saves `checkpoint-latest.pt` every 10,000 steps, replacing it atomically,
and `checkpoint-final.pt` when training stops. The launch script copies the final
checkpoint to persistent project storage before evaluation and verifies its SHA-256.
The revised run evaluates full validation MLM and contact P@L every 10,000 steps.
Intermediate evaluations pause training without restarting its processes. When
moving to another cluster, update the operational `periodic_evaluation_command`
to its local launcher and provide the frozen contact dataset via `CONTACT_ROOT`.
The data-root CLI argument is forwarded to the evaluator automatically. Evaluation
supports both four and eight training GPUs and does not alter the optimizer state.

## Four-GPU continuation

Keep the global batch at 2,048 by using 64 sequences per GPU and eight accumulation
steps: `64 × 4 × 8 = 2048`. Copy the original YAML, set `expected_world_size: 4`,
and change the Stage-1 `gradient_accumulation` from 4 to 8. Keep LR `0.0005`, weight
decay `0.01`, and warmup `1000` unchanged. To train another 100,000 steps after the
first 100,000, set both `max_steps` and `schedule_steps` to **200000**. These fields
are total endpoints, not additional-step counts. Use a sufficiently long wall-time
guard for the slower four-GPU run and a fresh output directory. For example,
`walltime_seconds: 172800` allows up to 48 hours; the allocation must also have
enough time remaining. The short four-GPU qualification measured about 0.85
seconds/step, or roughly 24 hours for another 100,000 steps.

Run inside a four-GPU allocation with the same source/runtime and verified data:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m nano_protein.train --config continuation-4gpu.yaml \
  --resume /path/to/checkpoint-final.pt \
  --data-root /path/to/verified/data --output-root /path/to/new-output
```

The optimizer starts at the saved step and warmup does not restart. Extending this
constant Stage-1 schedule is supported; changing a decay schedule requires a new
training plan. The loader rejects changes to the model, optimizer recipe, data
manifest, or effective global batch.

With the same GPU count, the loader restores each rank's RNG and sampler position.
This is not a bitwise determinism guarantee for subsequent parallel computation:
the Nibi test loaded every model and AdamW tensor exactly and reproduced the data
stream, but later weights differed after 100 more updates.
With a different count it restores weights, optimizer state, and counters, then
starts a new deterministic, disjoint-per-rank data stream. The trajectory therefore
is not bitwise equivalent to uninterrupted eight-GPU training, and previously seen
sequences can be sampled again. Source counts and epoch maxima describe that new
stream after a GPU-count change; sequence/token counters remain cumulative.

This resume support requires the new checkpoint metadata. Historical checkpoints
without a data-manifest hash and world size need an explicit provenance migration
before this loader will accept them.
