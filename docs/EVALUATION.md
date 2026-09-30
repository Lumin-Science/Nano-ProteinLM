# Evaluation contract

There are two kinds of measurement: masked-language-model (MLM) loss and contact precision at L (P@L). The final report has three entries because MLM is evaluated on two protein populations. The default profiles below are versioned as `nanoprotein-evaluation-v3` and replace the previous search and final settings.

| Profile | Contact P@L | MLM population and masks | Selection/reporting |
| --- | --- | --- | --- |
| `search` | Fixed 8,192 chains; one fitted probe | The exact same 8,192 chains; one fixed mask | Select by MLM NLL; report P@L alongside it |
| `scaleup` (CLI default) | All 26,062 non-probe eligible chains; five probe fits | Same 26,062 chains × five fixed masks; original 12,288 validation proteins × five fixed masks | Three means and sample standard deviations |

Search keeps the existing MLM reward gate, including its training-seed confirmation rule. P@L does not veto or select a candidate. Search `EVALUATION.json` retains `validation_mlm.sequence_mean_nll` as the selection field and `contact.precision_at_l` as the diagnostic. Both metrics must finish successfully. Baselines must be remeasured under this profile; historical 12,288-protein baseline values are not expected values for the new 8,192-chain score.

## Validation set

The search population is a frozen list of 8,192 chain IDs selected from the original 20,775-chain benchmark by SHA-256 ordering with seed 20260820. Its ID-file SHA-256 is `9948c40ab11a18bcdf096b078c13e841a02eb1d2286af638565945ebf71dbb44`. Expanding the contact pool does not resample this subset. Distinct chains remain distinct units even when their sequence strings match.

The full recovered pool contains 26,082 eligible representatives: 20 reserved probe chains and 26,062 evaluation chains. The original 12,288 MLM validation proteins remain unchanged: 4,096 cluster representatives per source from UniRef90, MGnify and OMG/IMG. The training release includes the original validation set and evaluated fragments of the 20 probes and original 20,775 contact chains in its protected union, under its declared identity/coverage exclusion rule. That rule does not exclude every longer parent sequence. Homology exclusion of the additional 5,287 contact chains has not been verified. Expanded-pool results must carry that qualification until the training exclusion audit is complete; they must not be labeled a newly established blind holdout.

The installed expanded training prefix contains exact full-sequence matches for 30 additional chains (29 evaluated-fragment matches). Four original evaluation chains have full-length parent matches but no evaluated-fragment matches. These are training availability checks, not proof of which sequences a given run sampled; homology exclusion remains unverified for the expanded pool.

## MLM masks and crops

All inputs use at most 510 residues plus BOS/EOS, for context 512. Contact chains use the same first-510-residue fragments as structural scoring. The original validation proteins use the existing deterministic crop offset derived from sequence digest and seed 20260821. That crop is fixed across all five masks. Mask seeds are `20260821, 20260822, 20260823, 20260824, 20260825`; the first reproduces the previous masking stream. Each seed is combined with the sequence digest, independently of GPU assignment and batching. The tokenizer's 15% masking and replacement rules are unchanged.

Each mask attempt averages masked-token NLL within each protein, then averages equally over proteins. A chain with no canonical amino-acid targets retains the historical zero-loss contribution; every cache and attempt receipt explicitly lists these units. This preserves the earlier paired score and its denominator. The five-attempt mean and sample SD are calculated from those five population means, not from pooled residues or pooled per-protein losses. This SD measures mask sensitivity at fixed proteins, crops, checkpoint and model weights. It is not a training-seed SD, confidence interval or estimate of population sampling uncertainty. Clean contact attention and masked MLM use separate forward passes.

## Contact P@L

A contact is a finite Cβ distance below 8 Å (Cα for glycine); long range means sequence separation at least 24. For each chain of evaluated length L, score the top L eligible pairs and compute the fraction that are true contacts. Report the arithmetic mean of these precisions across chains. P@L is stored as a fraction; multiply both its mean and SD by 100 for percentages. Random expected precision is each chain's contact density among eligible pairs, not 50%.

Features are symmetrized attention channels from all layers and heads. The encoder is frozen. The original ordered split of 16 probe-fit and four regularization-validation chains is retained. Each chain supplies at most 4,096 pairs per class. L1 logistic regression with SAGA selects C from 0.01, 0.1, 1 and 10 by validation average precision, then refits on all 20 probes. Evaluation-chain labels never fit or select this probe.

Five final probe attempts use seeds `20260819, 20260820, 20260821, 20260822, 20260823` for both pair sampling and logistic fitting. The 16/4 split stays fixed. Each attempt produces a macro P@L over the same 26,062 chains; the reported mean and sample SD are over those five values. This is probe-fit variability, not variation over different probe-chain splits, training seeds or bootstrap samples. No separate chain-bootstrap or single-mask result is promoted to the final three-metric report.

### Population provenance

The preserved build uses the 2024-01-01 RCSB annual snapshot plus first-release structures through 2024-02-28. Its receipts record 216,478 source structures, 781,077 extracted chains, 41,339 MMseqs representatives and 26,082 eligible representatives. Clustering used 40% sequence identity and 80% shorter-sequence coverage (`--min-seq-id 0.4 -c 0.8 --cov-mode 1 --cluster-mode 2 --kmer-per-seq 100`). Eligibility requires at least L eligible long-range pairs and L true long-range contacts after cropping and missing-coordinate filtering.

The old 20,775-chain evaluation was a deterministic subset after reserving 20 probes. In that historical subset, 2,612 original chains exceed 510 residues and 20,758 evaluated sequence strings are unique. The expanded profile uses all 26,062 non-probe eligible representatives. Recovering the extra chains does not establish their absence from the training corpus. Earlier ESMC comparisons and leaderboard values keep their original population labels; they are archived in [the prior contract](history/EVALUATION_PRE_V3.md).

## Data layout and preparation

```text
$DATA_ROOT/training/{uniref90,mgnify,omg_img}/validation/
$DATA_ROOT/evaluation/source/          frozen scoring mathematics
$DATA_ROOT/evaluation/contact/         immutable historical 20,775-chain release
$DATA_ROOT/evaluation/contact-v3/      26,062 evaluation + 20 probe chains and fixed IDs
$DATA_ROOT/evaluation/prepared-v3/     11 MLM caches and bound receipts
```

Setup downloads the immutable historical v2 source bundle and the [expanded v3 contact archive](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC/resolve/65b2308ce2d13db5a7844044ef9a657ba0da9980/evaluation/contact-evaluation-v3.tar.gz). Setup verifies the checksums and prepares the eleven mask caches locally. With the training/validation stores already prepared, the evaluation installer can also be run directly:

```bash
uv run --frozen python -m nanoprotein.setup_evaluation \
  --data-root "$DATA_ROOT"
```

An organizer may alternatively supply a verified local recovery using `--recovered-contact-pool PATH`. For offline setup, the evaluation installer accepts `--archive` for the v2 archive and `--contact-v3-archive` for the v3 archive. Subsequent setup verifies the installed caches and populations. New artifacts live beside the immutable historical data. Cache preparation is outside evaluation timing and does not use model predictions. Masks are shared across candidates, and every evaluation binds its checkpoint, code, source, population and mask-cache hashes.

## Evaluation execution

```bash
# One fixed paired search evaluation:
uv run --frozen python -m nanoprotein.evaluate --profile search \
  --checkpoint "$OUTPUT_ROOT/trial/checkpoint-final.pt" \
  --data-root "$DATA_ROOT/training" --output-root "$OUTPUT_ROOT/trial/evaluation"

# Three final mean/SD metrics (also the default of nanoprotein.evaluate):
bash scripts/speedrun.sh --evaluate default-100k
```

The task measurement wrapper selects `--profile search` explicitly; the speedrun evaluation wrapper selects `--profile scaleup`. Both use the same standard API. GPU selection follows `--contact-gpus`, then `EVAL_GPUS`, `CUDA_VISIBLE_DEVICES`, then visible devices. Contact defaults to eight workers per GPU; MLM uses one worker per GPU, batch size 32. Each final probe is scored independently. Optional static contact caches must match the expanded manifest.

Use `--resume-components` only with an identical request. A different checkpoint, source fingerprint, population, masks or execution setting requires a fresh output directory. Component files retain individual attempts for audit and recovery. The final `EVALUATION.json` contains only these aggregate metric entries:

```text
metrics.p_at_l.{mean,std,attempts}             # 5 probes; fraction
metrics.mlm_contact26062.{mean,std,attempts}  # 5 masks; nats
metrics.mlm_original12288.{mean,std,attempts} # 5 masks; nats
```

All SDs use denominator n−1. There is no standalone single-mask final score. The search training-seed summarizer accepts only the new search profile and checks both 8,192-chain populations. Historical one-off evaluations and optional P-CORE probes require explicit `--profile component`; they are outside the default search/final reports. Historical scripts and results retain their original meaning.
