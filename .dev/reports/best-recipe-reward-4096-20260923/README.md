# Best-recipe reward re-evaluation on 4,096 proteins

Completed September 23, 2026. Re-scored all six saved final checkpoints of the current default recipe, with separate Q/K/V Muon updates and no query centering/RMS restoration. These are the three 20-minute H100 runs on Fir and three one-hour L40S runs on CCK, using training seeds 42, 43 and 44. No training was launched. The original 32-protein evaluations and summaries remain unchanged.

## Measurements

| Training profile | Seed | MLM loss ↓, 4,096 proteins | P@L ↑, reused | MLM seconds | Evaluator seconds |
|---|---:|---:|---:|---:|---:|
| Fir, 4 H100, 20 min training | 42 | 2.667486 | 11.497828% | 21.053 | 25.061 |
| Fir, 4 H100, 20 min training | 43 | 2.668200 | 11.087363% | 18.644 | 22.405 |
| Fir, 4 H100, 20 min training | 44 | 2.667282 | 11.036266% | 19.406 | 23.109 |
| CCK, 4 L40S, 1 h training | 42 | 2.661036 | 11.774792% | 18.483 | 23.588 |
| CCK, 4 L40S, 1 h training | 43 | 2.660779 | 11.513413% | 18.543 | 25.473 |
| CCK, 4 L40S, 1 h training | 44 | 2.658639 | 12.025905% | 18.416 | 24.912 |

| Training profile | MLM loss ↓, mean ± sample SD | P@L ↑, mean ± sample SD, reused |
|---|---:|---:|
| Fir H100 | **2.667656 ± 0.000482** | 11.207152% ± 0.253026 pp |
| CCK L40S | **2.660151 ± 0.001316** | 11.771370% ± 0.256263 pp |

The MLM reward was recomputed with the ordinary `nanoprotein.evaluate` API. P@L was reused from each checkpoint's original full 20,775-chain evaluation after verifying its checkpoint hash, unique-chain count and mean over chain rows. No contact inference was repeated. The chain-bootstrap intervals remain in [RESULTS.json](RESULTS.json), separately from the training-seed sample SD above. Timing covers fresh MLM evaluation only; the evaluator total includes checkpoint loading and hashing but excludes environment qualification, data preparation and contact evaluation.

All six runs used 1,024 batches of four at context 512, sampling/masking seed 20260821, and the same hash-verified validation token stores. They evaluated **140,009 masked residues**, with **1,325 UniRef90**, **1,393 MGnify** and **1,378 OMG/IMG** proteins. The recipe, trained weights and training history were unchanged. The old 32-protein means were 2.601592 on Fir and 2.589492 on CCK; the change in loss reflects evaluation on the larger sample and must not be interpreted as a training regression. The observed training-seed SD is smaller on this sample, but three seeds from one recipe do not establish reliable ranking across all recipes.

## Reproduction and verification

Use the same pinned evaluator source and dependencies recorded in `RESULTS.json`. For each saved checkpoint, run the following command in its qualified GPU environment, using a fresh output directory and the verified validation data:

```bash
python -m nanoprotein.evaluate \
  --checkpoint "$CHECKPOINT" --data-root "$VALIDATION_DATA" \
  --output-root "$FRESH_EVALUATION_ROOT" \
  --validation-batches 1024 --validation-batch-size 4 --validation-context 512
```

The source snapshot is based on commit `62faec9b7faeaf3cbcae3762838195a72c67e007` plus the local evaluator/summary consistency changes; every source file's SHA-256 is in `RESULTS.json`. The MLM calculation is unchanged from the training studies' evaluator; the new code changes defaults, records evaluation settings and checks cache/aggregation compatibility. Fresh `mlm/EVALUATION.json` receipts are retained separately from the derived `evaluation/EVALUATION.json` files that attach the verified historical contact component. The derived files identify the original component paths and hashes explicitly.

Independent local verification checked the downloaded results against the earlier locally archived completion/evaluation receipts for every seed: checkpoint SHA-256, unchanged training config and completion receipt, sample size/settings, contact-chain population and mean, source-file/data hashes, and both aggregate means and sample SDs. The frozen source tar, launch/verification scripts and full receipts remain in the remote study directories below and in the local ignored `outputs/reward-4096-20260923/` directory.

| Cluster | Existing allocation / completed step | Node | New study directory |
|---|---|---|---|
| Fir | 60039326 / 60039326.5 | fc10219 | `/scratch/muchenli/Nano-ProteinLM-benchmarks/best-recipe-reward-4096-20260923` |
| CCK | 5218651 / 5218651.5 | kn086 | `/scratch/muchenli/Nano-ProteinLM/outputs/best-recipe-reward-4096-20260923` |

The enclosing allocations remain unchanged. The [Fir training study](../fir-human-ai-baseline-09-26-20260921/README.md) and [CCK training study](../cck-human-ai-baseline-09-26-20260921/README.md) retain their original 32-protein scores, checkpoint locations and training budgets. The six original checkpoints were present and verified when this re-evaluation ran.
