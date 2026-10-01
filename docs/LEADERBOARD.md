# AutoResearch leaderboard

The [AutoResearch protocol](AUTORESEARCH.md) fixes training budgets; the [evaluation contract](EVALUATION.md) fixes scoring. All rows below use the same evaluator, populations, masks and probe settings within their profile.

## Final evaluation

Each recipe trains from scratch to 24,200,224,761 non-padding model tokens at global batch 1,024, with training seed 42, 1,000 warmup steps and constant learning rate afterwards. All three checkpoints completed 100,015 updates at 24,200,455,289 tokens on four H100 GPUs with FA3. Round 2 includes its completed resume.

| Recipe | P@L ↑, 26,062 chains | MLM NLL ↓, 26,062 chains | MLM NLL ↓, 12,288 proteins |
|---|---:|---:|---:|
| ESMC 171M reference | 26.140% ± 0.049 pp | 2.458089 ± 0.000651 | 2.459364 ± 0.001016 |
| nanop-best-171m-round1 | 32.450% ± 0.113 pp | 2.371350 ± 0.001292 | 2.411270 ± 0.001380 |
| nanop-best-171m-round2 | 33.679% ± 0.049 pp | 2.363115 ± 0.001122 | 2.403480 ± 0.001271 |

P@L SD measures variation across five probe fits; MLM SD measures variation across five fixed masks. All three models use one training seed. These SDs do not measure training-seed uncertainty. The training-overlap audit has not established homology exclusion for 5,287 contact chains; [evaluation details](EVALUATION.md#validation-set) document known exact matches in the available training prefix.

## Search budget

Each recipe trains from scratch for 1,200 seconds on four H100 GPUs with FA3, global batch 256 and 500 warmup steps followed by constant learning rate.

| Recipe | MLM NLL ↓, 8,192 chains | P@L ↑, 8,192 chains |
|---|---:|---:|
| ESMC 171M reference | 2.765341 ± 0.001651 | 9.804% ± 0.249 pp |
| nanop-best-171m-round1 | 2.713510 ± 0.002109 | 11.298% ± 0.288 pp |
| nanop-best-171m-round2 | 2.713035 ± 0.000218 | 11.460% ± 0.443 pp |

Search values are mean ± sample SD across three independent training seeds (42, 43, 44), with one fixed mask and one fixed probe per checkpoint. P@L SD is in percentage points.

[Full-precision per-seed measurements, repeat values and checkpoint digests](benchmarks/baselines-evaluation-v3.json).
