# Continuing Setting 3 into Stage 2

<div class="ai">

The Nibi Stage 2 recipe is [setting3-nibi-stage2-b2048-300k.yaml](../.dev/configs/nibi/setting3-nibi-stage2-b2048-300k.yaml). It continues the completed 400,000-update Setting 3 model for 300,000 additional updates, ending at global step 700,000. Global batch remains 2,048: eight GPUs, 64 sequences per GPU, four accumulation steps. Context increases from 512 to 2,048, and the UniRef90/MGnify/OMG mixture becomes 63%/6%/31%. The model, Muon/AdamW optimizer groups, normalization, batch balancing, sqrt loss and weight decay remain unchanged.

</div>

The base learning rate decays linearly from 5e-4 to 5e-5 over Stage 2; existing attention and FFN multipliers remain 0.9 and 0.75. `schedule_start_step: 400000` and `schedule_steps: 700000` define this clock using global update counts. The original 1,000-update warmup is already complete. A short qualification or segmented run may stop before step 700,000 while keeping this exact schedule; ordinary resume rejects a changed decay endpoint or start boundary.

The prepared stores retain full sequences and crop at training time, so context expansion does not require a new tokenizer. Additional MGnify shards come from the same pinned, deduplicated and decontaminated release. UniRef90 and OMG continue their existing permitted source epochs. MGnify retains its strict no-repeat policy: every new queue contains only the old queue's unconsumed suffix plus appended record identities. A nested origin records the previous permutation, cursor and legacy history so reconstruction does not require a large checkpoint bitset. The capacity check uses unused records against only the remaining updates, with 5% expected-draw headroom and a runtime exhaustion guard.

An explicit transition creates a separate full checkpoint with identical model tensors, optimizer tensors, counters and random states, plus the new stage label, mixture, data manifest and reconstructed source history. Prefix verification compares the existing record identities and encoded residues; validation and decontamination contracts must remain identical. The original checkpoint and dataset are preserved.

<div class="ai">

```bash
python -m nanoprotein.stage_transition \
  --checkpoint /path/to/stage1/checkpoint-400000.pt \
  --config .dev/configs/nibi/setting3-nibi-stage2-b2048-300k.yaml \
  --old-data-root /path/to/stage1/data \
  --data-root /path/to/stage2/data \
  --output /path/to/stage2/transition/checkpoint-stage2-start.pt
```

</div>

[The Nibi launcher](../runs/nibi_setting3_stage2.sh) verifies this transition, qualifies 200 real updates with a full evaluation at update 400,100, audits a further restart to update 400,210, and launches production from the untouched Stage 2 start checkpoint. Production evaluates at global steps 410,000, 420,000, …, 700,000: 30 endpoints. Validation retains the historical fixed 512-token protocol and 4,096 sequences, while P@L retains all 20,775 chains and 5,000 chain-bootstrap replicates. A separate long-context MLM metric would be a different evaluation and is not substituted into the historical curve.

Checkpoints contain the full replicated optimizer state. Later eight-to-four-GPU continuation preserves the next global record identities when the global microbatch remains 512 sequences: use microbatch 128 on each of four GPUs and four accumulation steps, update `expected_world_size` to 4, and qualify memory and throughput on that hardware. Source history and the Stage 2 decay clock survive this repartition; rank-local crop/masking randomness is reseeded when the GPU count changes.
