# Training configurations

| Recipe | File | Parameters | Role |
|---|---|---:|---|
| Improved 171M default | [default.yaml](default.yaml) | 170,559,856 | Muon, RMSNorm, residual routing and initialization, batch balance, sqrt loss |
| ESMC-style 171M | [esmc-171m.yaml](esmc/esmc-171m.yaml) | 170,671,168 | AdamW reference for small-budget experiments |
| ESMC-style 300M | [esmc-300m.yaml](esmc/esmc-300m.yaml) | 332,997,184 | Original-size architecture reference |
| ESMC-style 600M | [esmc-600m.yaml](esmc/esmc-600m.yaml) | 575,036,992 | Original-size architecture reference |

The default uses RoPE 10k, base LR 5e-4, base WD 0.01, 1,000 warmup steps and a
four-H100 batch layout of 1,024 with FA3. ESMC presets retain their own LR/WD,
warmup and batch settings; see [esmc/README.md](esmc/README.md) for architecture
sources and training assumptions. Selecting a preset does not align its budget
or hyperparameters with another recipe.

[runs/speedrun.sh](../runs/speedrun.sh) calls setup and trains the default for
100k steps with a 16-hour guard. Supply a preset, fresh run name and ordinary
training options to customize it. The direct API is documented in
[USAGE.md](../docs/USAGE.md#training); every run saves its resolved configuration.

The [171m-validation-loss.md](../tasks/171m-validation-loss.md) task defines the
fixed-time comparison and ±5% parameter bound. Retired presets and their original
hashes are preserved in [.dev/configs/archive/README.md](../.dev/configs/archive/README.md).
