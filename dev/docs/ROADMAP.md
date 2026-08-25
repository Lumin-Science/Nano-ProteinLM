# Roadmap beyond fixing evaluation

The objective is not “train once”; it is a trustworthy model-development
system. The ordered work is:

1. **Prove selected-setting reproducibility.** Reproduce the promoted
   four-A100 setting from a fresh clone; verify Flash SDPA, memory headroom,
   checkpoint integrity, Q9-decontaminated data receipts, and frozen full-chain
   P@L while preserving the original ESMC-compatible default.
2. **Calibrate compute.** Sweep microbatch/accumulation, activation
   checkpointing, and `torch.compile` with short, predeclared trials. Select on
   tokens/s, MFU proxy, memory margin, and identical loss—not downstream test
   scores.
3. **Calibrate optimization.** Re-estimate the undisclosed 55M proxy LR/decay,
   test the paper-vs-release residual scaling discrepancy, and validate transfer
   to 300M then 600M with multiple seeds.
4. **Complete data governance.** Preserve the completed all-splits homology
   decontamination; finish source provenance, license review, corpus statistics,
   immutable manifests, and a Hugging Face dataset card. Keep raw licensed
   corpora out of public hosting.
5. **Keep evaluation decision-grade.** Preserve exact deterministic
   20,775-chain P@L for selected-setting provenance, and report the broader Q9
   P-CORE suite separately without allowing quarantined tasks into model
   selection. Add confidence intervals and seed replication for claims beyond
   the fixed-budget search, and version every evaluator/data receipt before
   comparing data or architecture variants.
6. **Run controlled science.** Source-mixture ablations, stage-duration/context
   ablations, and 300M/600M scaling comparisons. Change one scientific variable
   at a time and preserve compute/token parity.
7. **Productionize long runs.** Sharded checkpoints, robust resume, health
   monitoring, NaN/throughput alarms, artifact retention, and Slurm batch scripts
   distinct from the AutoResearch loop.
8. **Release responsibly.** Model card, dataset card, training/evaluation
   receipts, limitations and dual-use review, license inventory, safetensors
   export, and reproducible Git tags. Publish code under `Lumin-Science` and
   approved data/checkpoints under `LuminScience` only after review.

Promotion gates are evidence based: environment qualified → AutoResearch baseline
reproduced → evaluation trusted → multi-seed improvement → long-run readiness →
release review.
