# Nibi baseline with evaluation every 10,000 steps

Status: implementation and qualification in progress; production relaunch pending.

At the user's request, the initial Nibi training step `12162637.7` was cancelled
after approximately 500 steps. Allocation `12162637` remains running on `g27`.
The [initial run](../nibi-baseline-b2048-100k-20260908/README.md) and its successful
eight-to-four-GPU resume qualification remain archived.

The fresh run keeps the ESMC-like AdamW baseline: 170,671,168 parameters,
batch 2,048 on eight H100s (64/GPU × four accumulation steps), 100,000 steps,
context 512, LR 5e-4, WD 0.01, warmup 1,000, original LayerNorm/RoPE10k/FFN2048,
BF16 and the pinned FA3 kernel. Only checkpoint evaluation scheduling changes.

Every 10,000 steps, including step 100,000, evaluation uses the same **4,096 MLM
sequences** and **20,775 contact chains** as prior comparisons. Contact P@L uses
16 shards and a 5,000-replicate chain-bootstrap 95% CI. Each result is bound to
the evaluated checkpoint's SHA-256 and optimizer step.

For intermediate evaluations, all training ranks pause while a separate evaluator
runs. Frequent status broadcasts keep the process group alive during evaluation.
Training then continues in the same processes, retaining model/optimizer state,
sampler position and RNG. Evaluation time is recorded separately and excluded
from the training clock. Results live under `full/evaluations/step-NNNNNN/`.
The rolling checkpoint is retained until the next interval; the final full-state
checkpoint is copied to persistent project storage before its final evaluation.

The new qualification uses the full model/batch, evaluates the complete frozen
benchmark at step 10, then requires training to continue to step 20. Its short-run
scores are technical checks, not production quality results.

Artifact root: `/scratch/muchenli/Nano-Protein-LM-nibi-b2048-100k-eval10k-20260908`.
Source checkout: the artifact root with `-run` appended.
Persistent final checkpoint directory:
`/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-baseline-b2048-100k-eval10k-20260908`.
