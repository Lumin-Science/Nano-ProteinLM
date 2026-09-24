# Training configurations

Recipes come in two settings. [autoresearch/](autoresearch/) holds the search setting used by the task commands: global batch 256 and 500 warmup steps. [test-100k/](test-100k/) holds the final-evaluation setting: global batch 1,024 and 1,000 warmup steps. Both use 64 sequences per GPU on four GPUs, constant learning rate after warmup with no decay, and no stopping budget in the YAML; the command sets the time or token target.

| Recipe | Search setting | Final evaluation | Parameters | Description |
|---|---|---|---:|---|
| Plain ESMC reference | [esmc-171m.yaml](autoresearch/esmc-171m.yaml) | [esmc-171m.yaml](test-100k/esmc-171m.yaml) | 170,671,168 | AdamW, LR 5e-4, WD 0.01; the AutoResearch starting recipe |
| [nanop-best-171m-round1](../docs/leaderboard/nanop-best-171m-round1.md) | [yaml](autoresearch/nanop-best-171m-round1.yaml) | [yaml](test-100k/nanop-best-171m-round1.yaml) | 170,559,856 | Muon with RMSNorm, residual routing and depth-scaled init, batch balance, sqrt loss |
| [nanop-best-171m-round2](../docs/leaderboard/nanop-best-171m-round2.md) | [yaml](autoresearch/nanop-best-171m-round2.yaml) | [yaml](test-100k/nanop-best-171m-round2.yaml) | 170,559,856 | Round 1 plus separate Q/K/V Muon updates; our current best recipe |

All three use the paper's 170M scaling backbone (24 layers, width 768, 12 heads), context 512 and base LR 5e-4 / WD 0.01. [scripts/speedrun.sh](../scripts/speedrun.sh) trains `test-100k/nanop-best-171m-round2.yaml` for 100k steps by default; the task commands in [tasks/](../tasks/) take an `autoresearch/` recipe. [AUTORESEARCH.md](../docs/AUTORESEARCH.md) defines both settings, and [USAGE.md](../docs/USAGE.md#training) documents the training API; every run saves its resolved configuration.
