# Original-size ESMC references

These presets use the released ESMC architecture shapes in
[Appendix A.1.1, Table S1 (p. 29)](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1.full.pdf#page=29).
The [171M baseline](../esmc-171m-original.yaml) targets small-budget experiments
using the paper's 170M scaling backbone from
[Appendix A.1.4.1, Table S4](https://www.biorxiv.org/content/10.64898/2026.06.03.729735v1.full.pdf#page=29).

| Preset | Layers | Width | Heads | FFN width | Parameters |
|---|---:|---:|---:|---:|---:|
| [300M](esmc-300m-original.yaml) | 30 | 960 | 15 | 2,560 | 332,997,184 |
| [600M](esmc-600m-original.yaml) | 36 | 1,152 | 18 | 3,072 | 575,036,992 |

The model code supplies the architecture dimensions from the selected model
name. Both references use AdamW, LayerNorm, RoPE 10k and untied embeddings.

These are local Stage-1 presets for the public corpus. Their batch layout gives
256 sequences on four GPUs at context 512, with BF16/FA2 and 1,000 warmup steps.
Set step/token/time budgets through the standard training CLI and adjust the
batch layout for available GPU memory. The new presets have been checked for
configuration and model-shape consistency, without a GPU training run.

Learning rate and weight decay use the repository's
[width/depth transfer rule](../../nano_protein/schedule.py), with **assumed**
proxy values LR=6e-4 and WD=0.01 at width 512/depth 16. The paper does not publish
those calibrated proxy values, so these are baseline hypotheses. The resulting
LR/WD are approximately 2.337e-4 / 0.02567 for 300M and 1.778e-4 / 0.03375 for 600M.

The paper's released models use a much larger two-stage training protocol;
its schedule and batch settings are in Appendix A.1.3, Table S3. These presets
do not reproduce that full protocol or claim its pretrained performance.
The earlier [300M pilot config](../archive/esmc-300m-original.yaml) retains its
original 21k-step, four-hour settings in the archive.
