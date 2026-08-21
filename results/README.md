# Result receipts

This directory contains compact, reviewable summaries of completed canonical
runs. Checkpoints, per-sequence embeddings, and other large artifacts remain in
controlled storage and are addressed here by SHA-256 digest. A receipt is only
committed after training and every declared evaluation has completed.

Each receipt records the code revision, locked environment, data manifest,
hardware, measured training counters, checkpoint digests, and evaluation report
digests. Released copies should use a Git tag for source and immutable
Hugging Face revisions under `LuminScience` for approved checkpoints and data
manifests.

Routine `pcore-diagnostic-v1` task vectors are recorded with explicit coverage
and `aggregate_score: null`. They are useful for stage/final direction checks but
must not be compared with published P-CORE aggregates. Only the six-task release
profile may populate a P-CORE score.
