# Released 150M contact comparison — September 13, 2026

This evaluation compares the official [ESM-2 150M](https://huggingface.co/facebook/esm2_t30_150M_UR50D) and [Profluent-E1 150M](https://huggingface.co/Profluent-Bio/E1-150m) checkpoints with our final 171M model and the earlier released ESMC references. All P@L scores use the same frozen 20,775-chain split. No pretrained contact head is used; each encoder receives the same fitted-probe protocol.

## Results

| Model | Our split P@L ↑ | Our split 95% CI | Estimated FLOPs | Training tokens (S1 + S2 / total) |
|---|---:|---:|---:|---:|
| Profluent-E1 150M | 61.743% | 61.480–61.999% | — | 4.000T total |
| ESMC-600M | 58.031% | — | 2.491e+22 | 4.194T + 2.097T |
| ESMC-300M | 53.867% | — | 1.480e+22 | 4.194T + 2.097T |
| **Auto Research Best · 171M** | **46.264%** | **46.016–46.523%** | 2.334e+21 | 0.419T + 1.258T |
| ESM-2 150M | 44.927% | 44.682–45.177% | — | 1.000T total |

Both new results passed exact chain-ID equality against the frozen manifest and the saved final 171M per-chain record. The mean and 95% interval were independently recomputed from each collected TSV, using a different bootstrap chunk size; [the local audit](LOCAL_AUDIT.json) matches both remote results exactly. [ESM-2 verified result](esm2/RESULT_VERIFIED.json) and [per-chain values](esm2/contact-per-chain.tsv.gz); [Profluent-E1 verified result](e1/RESULT_VERIFIED.json) and [per-chain values](e1/contact-per-chain.tsv.gz).

ESM-2 has 148,140,154 loaded parameters and Profluent-E1 has 154,423,330; both retain their official 150M release names. Our final 171M score exceeds ESM-2 by 1.337 percentage points; Profluent-E1 exceeds ours by 15.478 points. These are differences between pretrained models with different training corpora and token budgets, not a matched recipe ablation.

## Protocol and adapters

The unchanged `fit_contact_probe_receipt` API fits L1 logistic probes on 16 fixed training chains, chooses C from 0.01, 0.1, 1 and 10 on four fixed validation chains, then refits on all 20 chains. The frozen SAGA solver allows 500 iterations per candidate and 1,000 for the final refit. Pair sampling uses seed 20260819 and at most 4,096 positive and 4,096 negative pairs per chain. The unchanged `run_contact_lite` API scores all 20,775 eligible chains with the existing symmetrization without average-product correction, long-range pair selection and top-L precision calculation. Each protein contributes equally to the final mean.

The [released-model adapter](../../../src/nanoprotein/released_contact.py) loads official weights and preserves native inference. ESM-2 returns its eager attention maps through Transformers. Profluent-E1 runs in **single-sequence mode without retrieved homologs**: hooks observe its already clipped and rotary-embedded queries and keys, then reconstruct full bidirectional attention probabilities with float32 matrix multiplication and softmax. The original attention output and all downstream hidden states remain unchanged. E1’s global and within-sequence layers both admit full attention for this single unpadded input.

E1 tokenizes a protein as `<bos> 1 residues 2 <eos>`, so its residue attention block begins at offset 2. The adapter removes these four boundary tokens and pads one zero token on each side for the frozen evaluator’s offset-1 interface, without renormalizing the residue probabilities. ESM-2 already uses one boundary token on each side. Evaluation crops sequences to the same first 510 residues as our existing protocol.

The GPU smoke check requires bit-identical native logits with attention capture enabled versus disabled, finite normalized attention maps at every layer, and recovered E1 attention-times-value outputs within 2% relative L2 error of native BF16 attention outputs. Synthetic CPU tests compare the reconstruction to PyTorch scaled-dot-product attention for both ordinary and grouped-query heads, and reject cached or multiple-protein inputs. These checks validate extraction, rather than changing the scientific evaluation.

## Provenance and execution

The official E1 source is pinned at `bfd2620a602248499f3d2583d85a7ecddf0b6e02`; its code, LICENSE, NOTICE and ATTRIBUTION are retained in the remote dependency directory. Model revisions and file SHA-256 digests are recorded in `MODEL_SOURCES.json`. The official Triton RMSNorm kernel resolved to `kernels-community/triton-layer-norm` revision `e1ac02e79e08c20f42ff12db8360c53f566326b5`. The base evaluator source is repository commit `8d36a45`, supplemented only by the adapter preserved on main. Optional dependencies live in an isolated Python 3.12 / PyTorch 2.8 / Transformers 4.56.2 environment; the training runtime is unchanged.

The [preparation script](prepare.sh) and [evaluation launcher](evaluate.sh) run in retained Nibi allocation 12162637 on g27. ESM-2 uses GPUs 0–3 and E1 uses GPUs 4–7, one process per GPU after fitting one shared probe per model. The remote root is `/scratch/muchenli/Nano-Protein-LM-released-150m-contact-20260913`. Existing training allocations and completed run artifacts are preserved.

The [merge auditor](../../scripts/merge_released_contact.py) checks checkpoint digests, shared-probe digests, frozen manifest identity, exact train/validation IDs, deterministic shard assignment and exact full-split chain coverage. Confidence intervals use 5,000 chain-bootstrap resamples with seed 20260820. These intervals describe variation across evaluation chains; they do not include pretraining-seed or probe-training-set uncertainty.

ESM-2 emitted a convergence warning during the frozen probe fit. Its candidate and final iteration caps remain the canonical 500/1,000; the fit log is retained as `esm2/fit.log`. The warning cannot be assigned to a particular concurrent C candidate from that log, and the frozen receipt does not expose each estimator’s iteration count. The reported interval measures chain uncertainty, not uncertainty in probe optimization.

## Training-budget sources

[Zeming Lin’s dissertation, section A.3.2.4](https://cs.nyu.edu/media/publications/ZemingLin-phd.pdf), reports 500,000 updates at two million tokens per batch for ESM-2 models other than 15B, implying approximately **1T training-token exposures** for 150M. We retain the reported rounded token budget rather than converting it to an assumed power-of-two batch. The [Profluent-E1 paper, page 3](https://storage.googleapis.com/e1-paper-a26c3c79/profluent-e1.pdf), reports **4T tokens** for each released size, with context increasing from 8,192 to 32,768 and a mixture of within-sequence and global attention. These are total exposures, not distinct proteins, and they are not divided into ESMC’s two training stages.

ESMC and our model retain the prior nominal maximum-context token budgets and stage-specific FLOP estimates in the [overview methods](../readme-overview-20260913/README.md). We do not infer E1 FLOPs from the ESMC dense-attention formula: its variable context and blocked attention curriculum require additional information. Our model’s actual non-padding token count is separately reported to distinguish it from the nominal budget.
