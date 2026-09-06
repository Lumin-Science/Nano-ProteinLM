# ESMC-171M recipe comparison

“175M” in this project refers to the 24-layer, width-768, 12-head family.
The new default is prepared in
[`esmc-171m-default-h100-fa3-b1024-stage1-100k.yaml`](../configs/esmc-171m-default-h100-fa3-b1024-stage1-100k.yaml).
Its peak LR of **5e-4** and **1,000-step warmup** are explicit user settings.
The R02 column below preserves the retained campaign record; it is not a
claim that R02 has been validated at the new default's batch or budget.

| Setting | Retained 171M R02 recipe, as recorded | New project default | ESMC paper reference |
|---|---|---|---|
| Parameters | 170,559,856 | 170,671,168 | 170.7M in the scaling experiment |
| Optimizer | Muon on transformer matrices; AdamW elsewhere | AdamW | AdamW |
| Peak/base LR | 0.000326599; Muon attention multiplier 0.9, FFN multiplier 0.75 | **0.0005** | Numerical calibrated value not disclosed; µP transfer described |
| Weight decay | 0.0183712; Muon multiplier 0.75 | 0.0367424 | Selective decay with µP scaling; numerical base not disclosed |
| Adam betas / epsilon | (0.9, 0.95) / 1e-8 | (0.9, 0.95) / 1e-8 | (0.9, 0.95) / 1e-8 |
| Gradient clip norm | 1.0 | 1.0 | 1.0 |
| Warmup | 554 optimizer steps | **1,000 optimizer steps** | 1,000 steps in family training specification |
| Transformer normalization | Parameter-free RMSNorm; Q/K normalization retained | LayerNorm | LayerNorm, including Q/K normalization |
| Residual/input routing | Learned | Fixed | Fixed |
| Depth-scaled projection initialization | Enabled | Disabled | Our added initialization is not part of the documented reference |
| RoPE | Base 20,000 | Base 10,000 | RoPE; numerical base not stated in cited architecture text |
| Context | 512 | 512 | 512 for the scaling experiment / family Stage 1 |
| Batch | 256 sequences on four GPUs | **1,024 sequences**: 64/GPU × 4 GPUs × 4 accumulation | Scaling experiment: 2^22 nominal tokens; family Stage 1: 8,192 sequences |
| Budget | One synchronized training hour on four L40S | **100,000 Stage-1 optimizer steps**, with a 16-hour emergency guard | 170M scaling experiment: 80,000 steps |
| Schedule | Warmup then constant LR | Warmup then constant LR for Stage 1 | WSD for family training; decay occurs in Stage 2 |
| Attention execution | FA2 in the recorded run | Verified FA3 on H100 | FA3 is our execution choice, not a claimed paper requirement |

The paper's released 300M/600M/6B models use a different budget: 1M Stage-1
steps followed by 500k Stage-2 steps at context 2,048. Do not substitute that
budget for the paper's 170M scaling experiment. Paper sources: Appendix
A.1.1–A.1.4.1 and Tables S3–S4 of
[Language Modeling Materializes a World Model of Protein Biology](https://biohub.ai/papers/esm_protein.pdf?__clerk_synced=true)
([bioRxiv record](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1)).

## Project choices and evidence limits

The default weight decay remains at the preceding batch-1,024 proposal's
0.0367424. After the user set LR to 5e-4, it no longer preserves exactly the
old batch-256 decay per processed sequence. That optional matching convention
would instead give WD ≈ 0.048; this has not been applied. The user-specified
1,000-step warmup also deliberately replaces sample-count-matched warmup.

The repository's older `esmc-171m-original.yaml` is an ESMC-style adaptation,
with assumed proxy LR/WD values transferred by `mup_hyperparameters`; those
numbers are not disclosed paper values. Its residual scaling follows the
released implementation, `sqrt(n_layers / 36)`, while the paper text writes
`sqrt(n_layers)`. Our corpus reconstruction also substitutes OMG IMG/M for
the unavailable exact JGI source; see [DATA.md](DATA.md).

The saved R02 recipe's three-seed contact P@L is **0.1077 ± 0.0017**. Its
reported +8.06% improvement is relative to a starting recipe that already
contained Muon and the retained architecture, not the new AdamW default.
Validation-loss mean/SD is not reported in the saved campaign record.
See [BASELINES.md](BASELINES.md#validated-esmc-171m-preset).

For a new training comparison, match dataset, batch, context, optimizer-step
budget, seed set, precision/backend, and evaluation. Report R02's actual Muon
LRs separately from its AdamW/base LR. Its historical one-hour score cannot
serve as the measured outcome of the proposed 100k-step comparison. The exact
paper run remains a literature reference while numerical hyperparameters and
source data are missing.

The four-H100 default timing estimate is approximately **12.8 hours for 100k
steps**, extrapolated from the measured batch-256 FA3 throughput. Batch-1,024
and R02 timings have not been measured. The already-running batch-256 job
loads its own immutable config at startup and is not changed by this preset.
