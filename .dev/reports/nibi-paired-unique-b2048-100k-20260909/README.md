# Nibi: paired 171M training with sufficient unique data

Replacement for the earlier repeatedly sampled Nibi runs, requested September 8,
2026 Toronto. Preparation and qualification are in progress; this record does
not yet claim a successful production launch.

| Run | Physical GPUs | Parameters | Global batch | Steps |
|---|---|---:|---:|---:|
| ESMC-like AdamW baseline | 0–3 | 170,671,168 | 2,048 | 100,000 |
| Setting 3 | 4–7 | 170,559,856 | 2,048 | 100,000 |

Initial qualification uses microbatch128 × four GPUs × four accumulation steps.
Both train from scratch with RoPE10k, FFN2048, untied embeddings, FA3/BF16,
base LR5e-4, WD0.01 and 1,000 warmup steps followed by constant Stage-1 LR.
Setting3 retains its hybrid Muon/AdamW groups, RMSNorm, residual routing,
depth-scaled initialization, batch balance and sqrt-mask-count loss.
Its attention/FFN Muon LRs remain4.5e-4/3.75e-4 and Muon WD0.0075.

The same pinned corpus and validation population are used by both models.
The selected training stores contain74,175,974 UniRef90,22,984,503 MGnify and
111,296,892 OMG/IMG records:208,457,369 in total. This covers204.8M draws per
run with source-specific headroom. Source samplers must remain in epoch0;
they fail on exhaustion instead of repeating proteins. Both models may see the
same protein once; the no-repeat condition applies within each run.

Evaluate4096 MLM sequences and all20,775 contact chains at each10k endpoint;
report P@L with5000-replicate chain-bootstrap CI. Retain final full model,
optimizer and sampler checkpoints on project storage. See
[data coverage](../../../docs/data-coverage.md) for the corrected interpretation
of older experiments and the new training safeguards.

Remote root:`/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909`.
Source checkout:the same path with`-run`appended. Allocation12162637 on g27 stays
intact. Old Setting3 step12162637.14 was intentionally stopped at the user's
request; previous artifacts remain available.
