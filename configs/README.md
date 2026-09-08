# Training configurations

Two 171M recipes are recommended for small-budget experiments:

| Recipe | File | Parameters | Role |
|---|---|---:|---|
| Current best — Setting 3 | [default.yaml](default.yaml) | 170,559,856 | Muon, RMSNorm, residual routing and initialization, batch balance, sqrt loss |
| Original ESMC-like AdamW | [esmc-171m-original.yaml](esmc-171m-original.yaml) | 170,671,168 | Original 171M reference recipe |

Both use the paper's 170M scaling backbone from
[Appendix A.1.4.1, Table S4](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1.full.pdf#page=29):
24 layers, width 768, FFN width 2,048 and untied embeddings. The default has RoPE 10k, base LR 5e-4, base WD 0.01, warmup 1,000,
and a four-H100 batch layout of 1,024 with FA3. The original retains its
family-scaled LR/WD, 554-step warmup and four-L40S batch layout of 256 with FA2.
Selecting the original recipe does not make these hyperparameters match.

[`runs/speedrun.sh`](../runs/speedrun.sh) calls setup, then trains the current
best for 100k steps with a 16-hour guard. It accepts a recipe, fresh run name
and ordinary training options; see [training](../docs/USAGE.md#training).
The Python API remains available directly. Every run saves its effective
configuration, so execution settings do not need another permanent YAML.

The [171M research task](../task/171m-validation-loss.md) supplies its own
fixed-time protocol and requires **162,137,610–179,204,726** trainable parameters
(±5% of the original reference).

The [archive](archive/README.md) preserves all 27 retired presets and the scale-up
manifest, including the 300M, paired-seed, H100 and Nibi experiments. Historical
configs and hashes are unchanged. For the original-size ESMC architectures, use
the [300M and 600M reference presets](reference/README.md), with explicit local
training assumptions and links to the paper's architecture tables.
