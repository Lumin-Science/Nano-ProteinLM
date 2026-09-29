# Data and setup

Training uses the processed LuminScience/LuminBench-Nano-ESMC corpus at immutable revision `bd38448d50d8f426d7b9bd4410b53159ea001259`. It contains 665,970,495 proteins across 565 training shards from UniRef90, MGnify and OMG/IMG. The starting mixture is 0.36:0.11:0.54, normalized by the sampler. Each source has a separate 4,096-protein validation shard; all three validation shards are always prepared.

```bash
bash scripts/setup.sh --training-shards 30
# Or size a download for an explicitly chosen sample budget:
# bash scripts/setup.sh --training-samples N
```

Setup downloads checksum-bound source prefixes, materializes local sequence stores and verifies the homology-exclusion receipts before training. The default 30 training shards contain 29,979,351 proteins; allow roughly 20 GB for compressed and prepared data plus separate space for dependencies, checkpoints and evaluation. Use a fresh DATA_ROOT when changing the shard selection. Existing prepared roots retain their saved selection.

Size the corpus for the actual source mixture and training budget, including sampling headroom. Read DATA_COVERAGE.json before interpreting a run. With the default policy the trainer stops on source exhaustion. Record any explicitly enabled reuse. Stored residues and requested samples are distinct from model tokens: proteins are cropped to at most 510 residues, BOS/EOS are added and padding is excluded from the training-token meter.

## Frozen evaluation assets

Setup installs a checksum-verified contact bundle under `DATA_ROOT/evaluation`. Its numerical evaluator sources, contact splits and payload hashes are fixed. The installer retains only the required payloads, source modules and license; archive metadata is discarded. The organizer should finish setup and inspect the prepared workspace before granting an agent access.

## Attribution

UniRef90 is provided by the UniProt Consortium, MGnify by EMBL-EBI and OMG/IMG by their source data providers. The processed corpus carries source provenance and decontamination receipts. Contact chains derive from PDB structures; preserve the supplied dataset records and their applicable terms. Evaluator code is distributed under the accompanying MIT license. These source-code terms do not replace the original datasets' terms.

On Hopper, the pinned FlashAttention-3 kernel comes from kernels-community/flash-attn3 at revision `e29f138fc363b396e5d2706c8a5f6fa7d36f41e0`. Its downloaded distribution retains its own license and metadata. Other supported GPUs use the FlashAttention-2 operators in the pinned PyTorch distribution. PyTorch and the remaining dependencies retain their upstream licenses.
