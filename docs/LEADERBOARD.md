# AutoResearch leaderboard

Results are grouped into two leaderboards: owner-run final evaluation and the AutoResearch search budget. The [protocol](AUTORESEARCH.md) defines both settings, and [AUTORESEARCH_BASELINE.md](AUTORESEARCH_BASELINE.md) describes the two search rounds behind nanop-best-171m-round1 and round2. Rows marked — have not yet been measured under the current protocol; earlier measurements used different settings and remain on the recipe pages.

## Final evaluation

Each recipe trains from scratch to 24,200,224,761 non-padding model tokens with its [test-100k config](../configs/test-100k/): global batch 1,024, 1,000 warmup steps followed by constant learning rate, and training seed 42. Scores use the final checkpoint, all 12,288 MLM validation proteins and all 20,775 contact chains, with a 5,000-replicate chain-bootstrap 95% interval for P@L.

| Recipe | MLM validation loss ↓ | P@L ↑ | P@L 95% CI |
|---|---:|---:|---:|
| [ESMC 171M reference](../configs/test-100k/esmc-171m.yaml) | — | — | — |
| [nanop-best-171m-round1](leaderboard/nanop-best-171m-round1.md) | — | — | — |
| [nanop-best-171m-round2](leaderboard/nanop-best-171m-round2.md) | — | — | — |

## Search budget

Each recipe trains its [autoresearch config](../configs/autoresearch/) for 1,200 seconds on four H100 GPUs with FA3: global batch 256 and 500 warmup steps followed by constant learning rate. Values are the mean ± sample SD over training seeds 42, 43 and 44; P@L SD is in percentage points. Scores use the same evaluation as final evaluation. These nine runs completed on 2026-09-24.

| Recipe | MLM validation loss ↓, mean ± SD | P@L ↑, mean ± SD |
|---|---:|---:|
| [ESMC 171M reference](../configs/autoresearch/esmc-171m.yaml) | 2.70552 ± 0.00245 | 9.841% ± 0.250 pp |
| [nanop-best-171m-round1](leaderboard/nanop-best-171m-round1.md) | 2.66008 ± 0.00072 | 11.299% ± 0.327 pp |
| [nanop-best-171m-round2](leaderboard/nanop-best-171m-round2.md) | 2.65983 ± 0.00044 | 11.480% ± 0.445 pp |
