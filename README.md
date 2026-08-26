# LuminBench Nano ESMC

A compact, auditable, uv-managed stack for training protein embedding models
and evaluating them with frozen protein benchmarks.

`main` contains the supported training, data, and evaluation path. Active
research, reports, completed experiment receipts, and AutoResearch instructions
are isolated under [`dev/`](dev/).

## Quick start

The repository now has a nanochat-style speedrun: from a fresh clone, the only
tool it expects is `uv >=0.11.31,<0.12`. It creates the locked Python/CUDA
environment, downloads and independently hashes the necessary public data
shards, materializes the mmap corpus, qualifies all four GPUs, and trains
ESMC-300M end to end:

```bash
git clone https://github.com/Lumin-Science/LuminBench-Nano-ESMC.git
cd LuminBench-Nano-ESMC
bash speedrun.sh
```

By default, every generated artifact stays in the current clone under `.exps/`:
the uv environment and cache, Hugging Face cache, verified prepared data,
receipts, checkpoints, and metrics. The default is the original four-GPU,
four-hour ESMC-300M setting. Existing run directories are never overwritten.

For a short end-to-end smoke run, use a fresh run name and a smaller time/data
budget. Whole-shard verification means even the smallest run downloads about
577 MB compressed:

```bash
TRAINING_SAMPLES=1 \
WALLTIME_SECONDS=60 \
RUN_NAME=smoke \
  bash speedrun.sh
```

This exact smoke path was verified from an empty environment on Killarney node
`kn082` (four NVIDIA L40S GPUs): the pinned six-file download selected 3,339,505
train-plus-validation records, the CUDA/FlashAttention forward-backward checks
passed, and the 332,997,184-parameter model completed 97 optimizer steps in
60.38 seconds. The final checkpoint independently matched its recorded SHA-256
`c7aae89f2c059d17ecf057a58fb4b304f1119b5c9c3e62bab86255805d502e7f`.

Copy [`.env.example`](.env.example) to `.env` to configure the artifact root,
dataset revision, GPU count, budget, config, or individual cache/output paths.
`speedrun.sh` automatically loads it; exported environment variables take the
same names. The default dataset revision is immutable, so a moving Hub branch
cannot change a run midway.

The training-only speedrun deliberately stops after its checkpoint and complete
receipts. The canonical training-plus-full-evaluation entrypoint additionally
expects the separately sourced evaluation datasets described in
[`docs/EVALUATION_DATASETS.md`](docs/EVALUATION_DATASETS.md):

```bash
bash runs/stage1_300m_4xa100_4h.sh
```

The current P@L-selected architecture is an explicit opt-in; model defaults and
the original config are unchanged:

```bash
CONFIG=$PWD/configs/esmc-300m-current-best.yaml \
TRAINING_SAMPLES=5766144 \
OUTPUT_ROOT=$PWD/outputs/stage1-300m-4xa100-4h-best \
  bash runs/stage1_300m_4xa100_4h.sh
```

That preset combines learned residual/input routing, parameter-free transformer
RMSNorm, depth-scaled residual-output initialization, and a final-20% linear
learning-rate cooldown. Every addition was selected by frozen full-chain P@L;
its provenance is documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Every training command fails closed unless the corpus has matching content
hashes, exact evaluation exclusion, all-splits MMseqs2 homology exclusion, zero
excluded-sequence intersections, a bidirectional sampled audit of the accelerated
search orientation, and zero train-validation overlap.

## Complete remote reservoir, budget-sized local corpus

The [Hugging Face dataset](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC)
is live and public. The speedrun pins verified commit
[`bd38448d`](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC/tree/bd38448d50d8f426d7b9bd4410b53159ea001259),
whose published manifest records 665,970,495 globally unique training proteins,
zero train/validation intersections, and zero intersections with the exact and
homology-based evaluation exclusions. It contains all three source arms,
validation shards, file/sequence checksums, and provenance receipts.

This is the complete verified release, not the former 2.55-GiB Stage-1 subset.
Before final evaluation decontamination, the controlled reservoir has 92,230,941
UniRef90, 348,135,082 MGnify, and 324,923,979 OMG/IMG 70%-identity
representatives. Final released records, residues, compressed bytes, and
rejection counts are bound by the verified Hub manifest. See
[`docs/DATA.md`](docs/DATA.md).

The roughly 459-GiB processing tree is not itself the public payload: it also
contains cluster-membership tables and build intermediates. Hugging Face gets
only decontaminated train/validation Parquet shards plus compact manifests,
checksums, attribution, and evaluation-exclusion receipts. A workstation or
training node normally downloads only the shard prefix its run requires.

There is no second GitHub repository for the binary corpus. Code, builders, and
portable manifests live here; the separately versioned data artifact belongs in
[`LuminScience/LuminBench-Nano-ESMC`](https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC).

## Full-reservoir and budget-sized downloads

The full release is organized as immutable, SHA-ordered Parquet shards rather
than one large archive. A run asks for its estimated total sequence exposures;
the downloader selects the minimum per-source shard prefixes that cover that
budget under the configured source mixture, pins one Hugging Face commit,
downloads every validation shard, checks all file and sequence hashes, and
materializes the existing fast mmap layout:

```bash
uv run --frozen python scripts/download_data.py \
  --repo-id LuminScience/LuminBench-Nano-ESMC \
  --revision bd38448d50d8f426d7b9bd4410b53159ea001259 \
  --training-samples 5376000 \
  --download-workers 8 \
  --cache-root .exps/cache/huggingface-dataset \
  --output-root .exps/data/training-samples-5376000
```

This is shard-level on-demand download, like nanochat—not row-level network
streaming during training. `--training-samples` is a count of planned sequence
draws (for example, steps × global sequence batch), not a residue count. The
saved download plan reports selected proteins, residues, and compressed bytes
separately. The full raw-to-release builder, all 959 pinned OMG object hashes,
and the measured processing report live under
[`dev/data/`](dev/data/).

## Distribution license

The released database compilation—our selection, arrangement, decontamination
ledger, packing, and release metadata—is distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Source protein
records retain their upstream terms and are marked path-by-path: UniRef90 under
CC BY 4.0, OMG/IMG under CC BY-SA 4.0, and MGnify under the EMBL-EBI Terms of
Use plus applicable original-owner rights. The umbrella license grants only
rights Lumin Science holds; it does not relicense upstream records.

The code remains under the repository [`LICENSE`](LICENSE). Dataset releases
carry their own source-specific notices and attribution alongside the published
manifest.

## Production evaluation performance

The accelerated evaluator is part of `main`, not `dev/`. It preserves the
frozen metrics while reducing wasted compute:

- packs windows from different proteins into residue-budgeted GPU batches;
- stores residue embeddings only for secondary-structure sequences instead of
  all P-CORE proteins;
- runs exact P-CORE probes concurrently as bounded, restartable subprocesses;
- provides a P@L-only path with 32 deterministic process shards distributed
  over four GPUs for rapid AutoResearch selection;
- retains the full evaluator, which shards contact P@L across three GPUs while
  P-CORE uses the fourth; and
- merges P@L rows back into the global deterministic order before the unchanged
  5,000-replicate chain bootstrap.

The implementation is in [`nano_protein/evaluate.py`](nano_protein/evaluate.py),
[`runs/evaluate_full_parallel.sh`](runs/evaluate_full_parallel.sh), and
[`runs/evaluate_p_at_l_parallel.sh`](runs/evaluate_p_at_l_parallel.sh).

## Repository map

```text
configs/          supported production training configuration
nano_protein/     tokenizer, ESMC model, mmap data, trainer, evaluation
runs/             production preparation, training, and evaluation entrypoints
scripts/          uv-invoked command-line interfaces and receipt verification
docs/             stable architecture, data, evaluation, and release contracts
release/          portable release cards and immutable-manifest templates
dev/              AutoResearch, proposed work, reports, plans, and run receipts
```

Source and issues: [`Lumin-Science/LuminBench-Nano-ESMC`](https://github.com/Lumin-Science/LuminBench-Nano-ESMC).
