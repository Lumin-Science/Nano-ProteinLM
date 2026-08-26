# Publication and artifact layout

Nothing is published automatically. The local repository is the development
source of truth until evaluation, licensing, and release gates pass.

| Artifact | Local location | Proposed public destination | Gate |
|---|---|---|---|
| Source, configs, docs, compact receipts | this Git repository | `github.com/Lumin-Science/LuminBench-Nano-ESMC` | clean CI and reproducibility review |
| Prepared-data manifest and dataset card | `data/processed/<version>/manifest.json` plus provenance | `huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC` | mixed-terms card, hash verification, and immutable revision receipt |
| Model weights, config, tokenizer, model card | content-addressed training output | `huggingface.co/LuminScience/<model-release>` | trusted evaluation, multi-seed confirmation, dual-use review |
| Full P-CORE/contact embedding caches | controlled scratch storage | not public by default | benchmark terms and storage policy |
| Raw UniRef/MGnify/OMG-IMG payloads | controlled source storage | never mirrored by this project | upstream terms govern access |
| Paper source and compact result tables | `dev/report/` and `dev/results/` | GitHub release/tag; archival venue later | result hashes match the tagged code |

## Release transaction

1. Freeze a clean Git commit and exact `uv.lock`.
2. Reproduce the canonical production baseline from a fresh clone and verify every receipt
   hash against the candidate artifacts.
3. Pass homology decontamination, repaired evaluation, confidence-interval, and
   multi-seed promotion gates.
4. Export inference-only weights as `safetensors`; verify a load-and-embed smoke
   test against the training checkpoint.
5. Complete model, dataset, license, limitations, and dual-use cards.
6. Upload immutable Hugging Face revisions, then commit those revision IDs to
   the Git receipt.
7. Tag the matching Git commit and publish the release notes. Never point a
   scientific result only at a mutable `main` branch or Hugging Face revision.

Git and Hugging Face credentials are intentionally absent from the canonical
training path. Publishing is a separate, reviewed operation.

## Full-reservoir dataset license decision

The project-authored database compilation for `full-open-v2` is
published under CC BY-SA 4.0. This covers only Lumin Science's selection,
arrangement, decontamination ledger, packing, and release metadata. It does not
relicense third-party sequence records. UniRef90 remains CC BY 4.0, the direct
`tattabio/OMG` distribution makes the derived OMG/IMG arm CC BY-SA 4.0, and
MGnify remains under the EMBL-EBI Terms of Use plus any original-data-owner
rights. The binary token stores are reversible sequence representations, so
tokenization does not erase these obligations. Each published dataset artifact
must carry its release card and portable ledgers beside its manifest.

The canonical upload is run from the storage host with uv-managed tooling:

```bash
HF_XET_HIGH_PERFORMANCE=1 uv run --frozen python scripts/upload_data.py \
  --release-root /path/to/full-open-v2 \
  --repo-id LuminScience/LuminBench-Nano-ESMC \
  --confirm-public-repo LuminScience/LuminBench-Nano-ESMC \
  --num-workers 16
```

The command refuses an unverified release, is safe to rerun after interruption,
and prints the resulting immutable Hugging Face commit SHA. Record that SHA in
Git before tagging the release.
