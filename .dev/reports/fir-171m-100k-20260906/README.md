# 100k-step ESMC-171M comparison on four H100s

Both recipes completed **100,000 Stage-1 optimizer steps** from scratch on
September 6, 2026. R02 improved both held-out MLM loss and full contact P@L.

| Recipe | Parameters | Training time | Validation loss ↓ | Perplexity ↓ | Full P@L ↑ | P@L 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Project default (AdamW) | 170,671,168 | 12h 00m 42s | 2.47436048 | 11.87411093 | 26.504938% | 26.294763–26.718848% |
| R02 (Muon + retained architecture) | 170,559,856 | 12h 56m 30s | **2.43698294** | **11.43847806** | **30.310361%** | **30.078864–30.547478%** |

R02 reduced validation loss by **0.03737754 (1.51%)**, and increased P@L by
**3.8054 percentage points (14.36% relative)**. Training took **7.74% longer**.
Equal steps and data exposure do not imply equal compute or wall time.

## Shared training and evaluation contract

- Four H100 80GB GPUs per recipe: default on Fir `fc10111`, R02 on `fc10212`.
- Seed **20260824**, context **512**, global batch **1,024** = 64 sequences
  per GPU × 4 GPUs × 4 accumulation microsteps.
- Base LR **5e-4**, base WD **0.01**, **1,000-step warmup**, then constant LR.
- BF16 mixed precision and pinned FA3; no compilation or gradient checkpointing.
- Identical data mixture and verified corpus; **102,400,000 sequences** and
  **24,200,224,761 model tokens** per run. This is sampled exposure, not that
  many unique proteins. The common 16-hour guard was not reached.
- MLM: the same **4,096 held-out sequences**, **139,963 masked residues**,
  context 512, evaluation seed 20260821, sequence-mean NLL.
- Contact: the same **20,775 chains**, fixed 16-chain probe fit + 4-chain
  regularization selection, canonical refit, and 16 deterministic inference
  shards. All shard hashes, unique chain coverage and checkpoint bindings were
  verified before publishing these results.

P@L intervals use **5,000 bootstrap resamples of chains**. Each recipe has only
one training seed, so these intervals do **not** measure training-seed variation.
The contact benchmark is paper-aligned; identity to the unpublished ESMC paper
chain list cannot be established. See [evaluation protocol](../../../docs/EVALUATION.md).

## Recipe differences

Default uses AdamW, LayerNorm, fixed residuals, standard initialization, and
RoPE base 10,000. R02 uses Muon for transformer matrices, parameter-free
transformer RMSNorm, learned residual/input routing, depth-scaled residual
projection initialization, and RoPE base 20,000. This is a full recipe
comparison; it does not isolate the optimizer.

| Parameter group | Default peak LR | R02 configured peak LR | Default WD | R02 WD |
|---|---:|---:|---:|---:|
| Attention matrices | 0.0005 | 0.00045 | 0.01 | 0.0075 |
| FFN matrices | 0.0005 | 0.000375 | 0.01 | 0.0075 |
| Other decayed parameters (AdamW) | 0.0005 | 0.0005 | 0.01 | 0.01 |
| Non-decayed parameters (AdamW) | 0.0005 | 0.0005 | 0 | 0 |

Muon LR values precede its internal `match_rms_adamw` matrix-shape adjustment.
The project default is not an exact reproduction of the paper's undisclosed
numerical calibration. [Full recipe comparison](../archive/171m-adamw-muon-recipes.md).

## Reproduction and evidence

Training source: `68f8cdc2cd8db6a0edacaea0cad687127632aa3f`.
Training corpus manifest SHA-256:
`43675d51421066ce8c5f68427886d57980e808c53c5bb1641de90cb74dda39ab`.
Contact manifest SHA-256:
`c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3`.

The actual runtime was Torch **2.13.0+cu130** / CUDA **13.0**, rather than the
lockfile's Linux cu126 index. The saved environment and run contracts record
that difference and the qualified FA3 kernel revision.

| Recipe | Training step | Contact step | Final checkpoint SHA-256 |
|---|---|---|---|
| Default | `58303658.7` | `58303658.10` | `96783380e4ecebab468e429e76a2ffcbb2bcfb4d11d7ab960a86791e6e7bc477` |
| R02 | `58303724.4` | `58303724.5` | `ad26e0c6e363ddaf1b37f7f079272ada83951bd3c6dcb8acfbdfdb9030c57cc9` |

- [Machine-readable results](results.json) and [paired per-chain contact scores](contact-per-chain.tsv).
- Exact executed configs: [default](default/config.yaml), [R02](r02/config.yaml).
- Completion, environment, validation, probe, and contact receipts are in
  [default](default) and [r02](r02); [file digests](artifact-sha256.json)
  cover the preserved numerical evidence.
- Checkpoints and original logs remain on Fir under
  `/scratch/muchenli/Nano-Protein-LM-paired-100k-20260906/{default-fc10111,r02-fc10212}`.
  The source checkout is `/scratch/muchenli/Nano-Protein-LM-paired-100k-20260906-run`.
- Full local audit artifacts, including all contact shard reports and launch
  scripts, are in `.exps/fir-171m-paired-100k-20260906` (untracked). Absolute
  paths inside copied receipts identify their original execution locations.

The earlier 10k pilots were cancelled at the user's request and replaced by
these fresh 100k runs. Their results are not included here.
