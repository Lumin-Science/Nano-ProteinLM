# AutoResearch leaderboard

Results are grouped into two leaderboards: owner-run final evaluation and the AutoResearch search budget. The [protocol](AUTORESEARCH.md) defines both settings, and [AUTORESEARCH_BASELINE.md](AUTORESEARCH_BASELINE.md) describes the two search rounds behind nanop-best-171m-round1 and round2. Round 2's final-evaluation row reports its previous-protocol run while its H100 rerun is pending; other historical measurements remain on the recipe pages.

## Final evaluation

Under the current protocol, each recipe trains from scratch to 24,200,224,761 non-padding model tokens with its [test-100k config](../configs/test-100k/): global batch 1,024, 1,000 warmup steps followed by constant learning rate, and training seed 42. Scores use the final checkpoint, all 12,288 MLM validation proteins and all 20,775 contact chains, with a 5,000-replicate chain-bootstrap 95% interval for P@L.

| Recipe | Run / MLM validation proteins | MLM validation loss ↓ | P@L ↑ | P@L 95% CI |
|---|---|---:|---:|---:|
| [ESMC 171M reference](../configs/test-100k/esmc-171m.yaml) | Current H100 rerun / 12,288 | 2.459574 | 26.088% | 25.875–26.298% |
| [nanop-best-171m-round1](leaderboard/nanop-best-171m-round1.md) | Current H100 rerun / 12,288 | 2.411284 | 32.418% | 32.185–32.651% |
| [nanop-best-171m-round2](leaderboard/nanop-best-171m-round2.md) | Previous L40S run / 4,096 | 2.410035 | 33.450% | 33.215–33.679% |

The ESMC and round-1 reruns completed on September 24 and 25, 2026, respectively, using four H100 GPUs with FA3. Both stopped after 100,015 updates at 24,200,455,289 model tokens, completing the optimizer step that crossed the token target. Training took 12h 03m for ESMC and 12h 33m for round 1, excluding evaluation.

The previous round-2 run used four L40S GPUs with FA2, seed 42, global batch 1,024 and 100,000 updates. It processed 24,196,983,520 model tokens from a 111-shard corpus in 35h 28m of training. Its 4,096-protein MLM validation sample and masking differ from the current 12,288-protein protocol, so its loss is not directly comparable with the reruns. All three P@L scores use the same 20,775 contact chains and 5,000 bootstrap replicates. Each row uses one training seed; the intervals measure variation across chains. See the [round-2 component study](leaderboard/nanop-best-171m-round2.md#100k-step-component-study-under-the-previous-protocol) for the original result.

## Search budget

Each recipe trains its [autoresearch config](../configs/autoresearch/) for 1,200 seconds on four H100 GPUs with FA3: global batch 256 and 500 warmup steps followed by constant learning rate. Values are the mean ± sample SD over training seeds 42, 43 and 44; P@L SD is in percentage points. Scores use the same evaluation as final evaluation. These nine runs completed on 2026-09-24.

| Recipe | MLM validation loss ↓, mean ± SD | P@L ↑, mean ± SD |
|---|---:|---:|
| [ESMC 171M reference](../configs/autoresearch/esmc-171m.yaml) | 2.70552 ± 0.00245 | 9.841% ± 0.250 pp |
| [nanop-best-171m-round1](leaderboard/nanop-best-171m-round1.md) | 2.66008 ± 0.00072 | 11.299% ± 0.327 pp |
| [nanop-best-171m-round2](leaderboard/nanop-best-171m-round2.md) | 2.65983 ± 0.00044 | 11.480% ± 0.445 pp |
