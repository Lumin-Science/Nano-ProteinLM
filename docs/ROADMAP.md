# Roadmap beyond fixing evaluation

The objective is not “train once”; it is a trustworthy model-development
system. The ordered work is:

1. **Prove the speedrun.** Reproduce a four-A100, 30-minute run from a fresh
   clone; verify Flash SDPA, memory headroom, checkpoint resume, and deterministic
   data receipts.
2. **Calibrate compute.** Sweep microbatch/accumulation, activation
   checkpointing, and `torch.compile` with short, predeclared trials. Select on
   tokens/s, MFU proxy, memory margin, and identical loss—not downstream test
   scores.
3. **Calibrate optimization.** Re-estimate the undisclosed 55M proxy LR/decay,
   test the paper-vs-release residual scaling discrepancy, and validate transfer
   to 300M then 600M with multiple seeds.
4. **Complete data governance.** Homology decontamination, source provenance,
   license review, corpus statistics, immutable manifests, and a Hugging Face
   dataset card. Keep raw licensed corpora out of public hosting.
5. **Make evaluation decision-grade.** The routine gate is now bounded and
   writes restartable component receipts, but it is deliberately not a P-CORE
   score. Repair EC, validate a faster secondary-structure estimator before any
   protocol change, add confidence intervals and seed replication, freeze the
   evaluator as an immutable repository dependency, finish task-parallel full
   P-CORE and 20,775-chain P@L, and define one frozen aggregate before comparing
   data or architecture variants.
6. **Run controlled science.** Source-mixture ablations, stage-duration/context
   ablations, and 300M/600M scaling comparisons. Change one scientific variable
   at a time and preserve compute/token parity.
7. **Productionize long runs.** Sharded checkpoints, robust resume, health
   monitoring, NaN/throughput alarms, artifact retention, and Slurm batch scripts
   distinct from the interactive speedrun.
8. **Release responsibly.** Model card, dataset card, training/evaluation
   receipts, limitations and dual-use review, license inventory, safetensors
   export, and reproducible Git tags. Publish code under `Lumin-Science` and
   approved data/checkpoints under `LuminScience` only after review.

Promotion gates are evidence based: environment qualified → speedrun
reproduced → evaluation trusted → multi-seed improvement → long-run readiness →
release review.
