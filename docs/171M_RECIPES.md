# ESMC-171M recipe comparison

“175M” in this project refers to the 24-layer, width-768, 12-head family.
The new default is prepared in
[`esmc-171m-default-h100-fa3-b1024-stage1-100k.yaml`](../configs/esmc-171m-default-h100-fa3-b1024-stage1-100k.yaml).
Its peak LR of **5e-4**, base weight decay of **0.01**, and **1,000-step warmup**
are explicit user settings.
The matched R02 variant is prepared in
[`esmc-171m-r02-h100-fa3-b1024-stage1-100k.yaml`](../configs/esmc-171m-r02-h100-fa3-b1024-stage1-100k.yaml).
The table now compares these two aligned presets with the paper reference.
The original one-hour R02 record remains unchanged; the aligned variant has
not yet been trained or quality-qualified.

| Setting | Aligned 171M R02 recipe | New project default | ESMC paper reference |
|---|---|---|---|
| Parameters | 170,559,856 | 170,671,168 | 170.7M in the scaling experiment |
| Optimizer | Muon on transformer matrices; AdamW elsewhere | AdamW | AdamW |
| Peak/base LR | **0.0005**; Muon attention multiplier 0.9, FFN multiplier 0.75 | **0.0005** | Numerical calibrated value not disclosed; µP transfer described |
| Weight decay | **0.01**; Muon multiplier 0.75 gives 0.0075 | **0.01** | Selective decay with µP scaling; numerical base not disclosed |
| Adam betas / epsilon | (0.9, 0.95) / 1e-8 | (0.9, 0.95) / 1e-8 | (0.9, 0.95) / 1e-8 |
| Gradient clip norm | 1.0 | 1.0 | 1.0 |
| Warmup | **1,000 optimizer steps** | **1,000 optimizer steps** | 1,000 steps in family training specification |
| Transformer normalization | Parameter-free RMSNorm; Q/K normalization retained | LayerNorm | LayerNorm, including Q/K normalization |
| Residual/input routing | Learned | Fixed | Fixed |
| Depth-scaled projection initialization | Enabled | Disabled | Our added initialization is not part of the documented reference |
| RoPE | Base 20,000 | Base 10,000 | RoPE; numerical base not stated in cited architecture text |
| Context | 512 | 512 | 512 for the scaling experiment / family Stage 1 |
| Batch | **1,024 sequences**: 64/GPU × 4 GPUs × 4 accumulation | **1,024 sequences**: 64/GPU × 4 GPUs × 4 accumulation | Scaling experiment: 2^22 nominal tokens; family Stage 1: 8,192 sequences |
| Budget | **100,000 Stage-1 optimizer steps**, with a 16-hour emergency guard | **100,000 Stage-1 optimizer steps**, with a 16-hour emergency guard | 170M scaling experiment: 80,000 steps |
| Schedule | Warmup then constant LR | Warmup then constant LR for Stage 1 | WSD for family training; decay occurs in Stage 2 |
| Attention execution | FA3 configured on H100 | FA3 on H100 | FA3 is our execution choice, not a claimed paper requirement |

The paper's released 300M/600M/6B models use a different budget: 1M Stage-1
steps followed by 500k Stage-2 steps at context 2,048. Do not substitute that
budget for the paper's 170M scaling experiment. Paper sources: Appendix
A.1.1–A.1.4.1 and Tables S3–S4 of
[Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf?__clerk_synced=true)
([bioRxiv record](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1)).

## Matched comparison contract

### 10k-step pilot

The initial paired pilot uses
[`esmc-171m-default-h100-fa3-b1024-stage1-10k.yaml`](../configs/esmc-171m-default-h100-fa3-b1024-stage1-10k.yaml)
on Fir `fc10111` and
[`esmc-171m-r02-h100-fa3-b1024-stage1-10k.yaml`](../configs/esmc-171m-r02-h100-fa3-b1024-stage1-10k.yaml)
on Fir `fc10212`. Both stop at 10,000 optimizer steps, with `schedule_steps`
also set to 10,000 and a four-hour emergency guard. All other settings match
the respective 100k-step presets, including 1,000-step warmup, base LR 5e-4,
base WD 0.01, global batch 1,024, and seed 20260824. Each completed pilot sees
10.24 million sequences. Use the shared held-out MLM evaluation below.

### Full comparison

Both prepared recipes use seed 20260824, global batch 1,024, context 512,
identical source mixture weights, 1,000-step warmup then constant LR, base
LR 5e-4, base WD 0.01, Adam betas (0.9, 0.95), clip norm 1.0, BF16/FA3,
no compilation or gradient checkpointing, and 100,000 Stage-1 optimizer steps.
Use the same verified data root and four-H100 layout for both runs.

The intended differences are the R02 optimizer and architecture. At peak:

| Parameter group | Default LR | R02 configured LR | Default WD | R02 WD |
|---|---:|---:|---:|---:|
| Attention matrices | 0.0005 | 0.00045 | 0.01 | 0.0075 |
| FFN matrices | 0.0005 | 0.000375 | 0.01 | 0.0075 |
| Decayed AdamW parameters | 0.0005 | 0.0005 | 0.01 | 0.01 |
| Non-decayed AdamW parameters | 0.0005 | 0.0005 | 0 | 0 |

Muon retains `match_rms_adamw` adjustment, momentum 0.95, five Newton–Schulz
steps, and Nesterov updates. The table reports configured group LRs before
Muon's internal matrix-shape adjustment. R02's RMSNorm, learned routing,
depth-scaled initialization, and RoPE base 20,000 remain enabled. This compares
the complete improved recipe with the default, not an optimizer-only ablation.

Count only completed 100k-step runs as the matched comparison. Equal update
and data budgets do not imply equal wall time or FLOPs. The common 16-hour
wall-time guard is an emergency limit, not the scientific comparison budget.
If either run reaches it first, that partial run needs separate labeling.

Evaluate both with the same checkpoint endpoint, data root, and options:
`--validation-batches 256 --validation-batch-size 16 --validation-context 512`.
The evaluator fixes its seed to 20260821 and uses the same held-out MLM
protocol. These 4,096-sequence evaluations do not estimate training-seed
uncertainty; repeated paired training seeds are needed for that.

## Project choices and evidence limits

The shared base weight decay is the user-selected 0.01. This choice is informed
by related protein pretraining recipes, particularly
[AMPLIFY 120M/350M](https://huggingface.co/chandar-lab/AMPLIFY_350M#training-descritpion),
which uses AdamW with weight decay 0.01, betas (0.9, 0.95), BF16, context 512,
and a 1,000-step warmup in Stage 1. It is a project choice, not a recovered
ESMC calibration or a guarantee of optimality for this 100k-step budget.
No additional batch-size or µP multiplier is applied to the shared base WD;
R02 retains its explicit Muon multiplier of 0.75. The shared 1,000-step warmup
is also a user setting rather than a sample-count-matched transfer.

The repository's older `esmc-171m-original.yaml` is an ESMC-style adaptation,
with assumed proxy LR/WD values transferred by `mup_hyperparameters`; those
numbers are not disclosed paper values. Its residual scaling follows the
released implementation, `sqrt(n_layers / 36)`, while the paper text writes
`sqrt(n_layers)`. Our corpus reconstruction also substitutes OMG IMG/M for
the unavailable exact JGI source; see [DATA.md](DATA.md).

The historical `autoresearch_171m_4xl40s_1h.yaml` keeps base LR 0.000326599,
base WD 0.0183712, warmup 554, global batch 256, and the one-hour L40S budget.
The saved R02 recipe's three-seed contact P@L is **0.1077 ± 0.0017**. Its
reported +8.06% improvement is relative to a starting recipe that already
contained Muon and the retained architecture, not the new AdamW default.
Validation-loss mean/SD is not reported in the saved campaign record.
See [BASELINES.md](BASELINES.md#validated-esmc-171m-preset).

The historical one-hour score cannot serve as the measured outcome of the
aligned 100k-step comparison. The exact paper run remains a literature
reference while numerical hyperparameters and source data are missing.

The four-H100 default timing estimate is approximately **12.8 hours for 100k
steps**, extrapolated from the measured batch-256 FA3 throughput. Batch-1,024
and R02 timings have not been measured. The already-running batch-256 job
loads its own immutable config at startup and is not changed by this preset.
