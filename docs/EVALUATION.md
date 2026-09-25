# Evaluation

## MLM validation loss

Compute masked-token negative log-likelihood within each protein, then average over proteins. Lower is better. The evaluator scores every protein in the three held-out validation shards once: 12,288 proteins, 4,096 per source, at maximum context 512. Each protein's crop offset and mask positions come from mask seed 20260821 and the protein's SHA-256, so batch size and order do not change the score.

Each receipt records the protocol, settings, a SHA-256 of the evaluated protein digests and per-source mean losses. Compare only receipts with identical protocol and settings. The score does not estimate variability across training seeds.

## Contact P@L

Contact P@L is the mean, over 20,775 frozen chains, of precision among the top L predicted long-range residue contacts. L is the evaluated chain length. Higher is better. Fit one contact probe per checkpoint using the frozen probe split and procedure, then evaluate all chains. Use 5,000 chain-bootstrap replicates for a 95% interval. The interval describes variation over chains.

## Evaluate a checkpoint

```bash
uv run --frozen python -m nanoprotein.evaluate \
  --checkpoint outputs/trial-001/checkpoint-final.pt \
  --data-root data/training --output-root outputs/trial-001/evaluation \
  --validation-context 512 \
  --run-contact --contact-mode parallel --contact-chains 20775 --contact-bootstrap 5000 \
  --contact-gpus 0,1,2,3 --contact-workers 32 \
  --contact-root data/evaluation/contact --external-src data/evaluation/source
```

Adjust only the file paths and visible GPU identifiers to match the provisioned workspace. Use the final checkpoint from the declared training budget. The organizer evaluates with a trusted copy of the scoring code and records checkpoint and evaluator hashes.
