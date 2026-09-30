"""Contract tests for population pairing, crop stability, sharding and repeat statistics."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nanoprotein.evaluate import parse_args, validation_example
from nanoprotein.evaluation_profiles import MASK_SEEDS, PROBE_SEEDS, mean_sd, merge_mlm
from nanoprotein.prepared_mlm import prepare_cache, score_cache, verify_cache
from nanoprotein.tokenizer import ProteinTokenizer, mask_tokens


class ToyModel:
    def __call__(self, tokens, attention):
        logits = torch.arange(64, dtype=torch.float32)[None, None, :].expand(*tokens.shape, 64)
        return {"logits": logits * (tokens.float().sum(1)[:, None, None] % 7 + 1) / 30}


class ProfileContracts(unittest.TestCase):
    def test_legacy_mask_zero_and_fixed_crop(self):
        tokenizer = ProteinTokenizer.esmc()
        residues = tokenizer.encode_residues("ACDEFGHIKLMNPQRSTVWY" * 40)
        digest = hashlib.sha256(b"long-test-protein").digest()
        generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256((20260821).to_bytes(8, "big") + digest).digest()[:8], "big"))
        offset = int(torch.randint(len(residues) - 510 + 1, (1,), generator=generator))
        expected = torch.tensor([tokenizer.bos_id, *residues[offset:offset + 510], tokenizer.eos_id]).long()[None, :]
        old_tokens, old_labels = mask_tokens(expected, torch.ones_like(expected, dtype=torch.bool), tokenizer, generator=generator)
        masks = []
        for seed in MASK_SEEDS:
            tokens, labels = validation_example(residues, digest, residue_limit=510, tokenizer=tokenizer, mask_seed=seed)
            recovered = torch.where(labels != -100, labels, tokens)
            self.assertTrue(torch.equal(recovered, expected[0]))
            masks.append(labels != -100)
            if seed == MASK_SEEDS[0]:
                self.assertTrue(torch.equal(tokens, old_tokens[0]))
                self.assertTrue(torch.equal(labels, old_labels[0]))
        self.assertEqual(len({tuple(mask.tolist()) for mask in masks}), 5)

    def test_shards_macro_average_and_receipt_binding(self):
        tokenizer = ProteinTokenizer.esmc()
        examples = [(f"unit-{i}", tokenizer.encode_residues("ACDEFGHIKLMNPQRSTVWY" * i), hashlib.sha256(str(i).encode()).digest()) for i in range(1, 8)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.npz"
            receipt = prepare_cache(path, examples, population="toy", seed=20260821, binding={})
            whole = score_cache(ToyModel(), path, device=torch.device("cpu"), batch_size=2, worker=0, workers=1)
            reports = [{"worker": i, "workers": 3, "checkpoint_sha256": "checkpoint",
                        "results": {path.name: score_cache(ToyModel(), path, device=torch.device("cpu"), batch_size=2, worker=i, workers=3)}} for i in range(3)]
            result = merge_mlm(reports, path.name, receipt, "checkpoint")
            self.assertAlmostEqual(result["sequence_mean_nll"], sum(r["nll"] for r in whole["rows"]) / 7)
            with self.assertRaises(ValueError):
                merge_mlm(reports, path.name, receipt, "different-checkpoint")
            with self.assertRaises(ValueError):
                merge_mlm(reports[:-1], path.name, receipt, "checkpoint")
            with path.open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaises(ValueError):
                verify_cache(path)

    def test_unknown_only_protein_preserves_zero_target_convention(self):
        tokenizer = ProteinTokenizer.esmc()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.npz"
            receipt = prepare_cache(path, [("unknown", tokenizer.encode_residues("X" * 64), hashlib.sha256(b"X" * 64).digest())], population="toy", seed=20260821, binding={})
            self.assertEqual(receipt["zero_target_unit_ids"], ["unknown"])
            scored = score_cache(ToyModel(), path, device=torch.device("cpu"), batch_size=2, worker=0, workers=1)
            self.assertEqual(scored["rows"][0]["nll"], 0)

    def test_mean_sample_sd(self):
        report = mean_sd([1, 2, 3, 4, 5], unit="nats", variation="masks", sequences=12288)
        self.assertEqual(report["mean"], 3)
        self.assertAlmostEqual(report["std"], math.sqrt(2.5))
        self.assertEqual(report["attempts"], 5)
        self.assertEqual(len(set(PROBE_SEEDS)), 5)
        with self.assertRaises(ValueError):
            mean_sd([1, float("nan")], unit="fraction", variation="probe", sequences=26062)

    def test_profiles_and_component_isolation(self):
        common = ["--checkpoint", "model.pt", "--data-root", "/data/training", "--output-root", "/out"]
        args = parse_args(common)
        self.assertEqual(args.profile, "scaleup")
        self.assertEqual(args.contact_root, Path("/data/evaluation/contact-v3"))
        self.assertEqual(parse_args(common + ["--profile", "search"]).profile, "search")
        with self.assertRaises(SystemExit):
            parse_args(common + ["--skip-validation-mlm"])
        with self.assertRaises(SystemExit):
            parse_args(common + ["--profile", "component", "--contact-chains", "0"])


if __name__ == "__main__":
    unittest.main()
