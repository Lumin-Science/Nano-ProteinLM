"""Full-population MLM validation is fixed per protein and independent of batch layout."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nanoprotein.data import SOURCES, TokenStore, _StoreWriter
from nanoprotein.evaluate import validation_example, validation_mlm, validation_settings
from nanoprotein.model import build_model
from nanoprotein.tokenizer import ProteinTokenizer


class ValidationMLMTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.tokenizer = ProteinTokenizer.esmc()
        canonical = np.asarray(self.tokenizer.canonical_ids)
        rng = np.random.default_rng(7)
        self.sizes = {"uniref90": 5, "mgnify": 3, "omg_img": 4}
        for source in SOURCES:
            writer = _StoreWriter(self.root / source / "validation")
            for row in range(self.sizes[source]):
                residues = rng.choice(canonical, size=int(rng.integers(6, 40)))
                digest = hashlib.sha256(f"{source}-{row}".encode()).hexdigest()
                writer.append(residues, digest)
            writer.finish()
        torch.manual_seed(3)
        self.model = build_model("tiny", attention_backend="math").eval()

    def score(self, batch_size):
        return validation_mlm(
            self.model,
            data_root=self.root,
            device=torch.device("cpu"),
            context_length=16,
            batch_size=batch_size,
        )

    def test_scores_every_validation_protein_once(self):
        report = self.score(4)
        self.assertEqual(report["sequences"], sum(self.sizes.values()))
        self.assertEqual(report["source_counts"], self.sizes)
        self.assertEqual(report["settings"], validation_settings(16))
        manifest = hashlib.sha256()
        for source in SOURCES:
            store = TokenStore.open(self.root / source / "validation")
            for row in range(store.index.size):
                manifest.update(bytes(store.index[row]["digest"]))
        self.assertEqual(report["manifest_sha256"], manifest.hexdigest())

    def test_batch_size_does_not_change_the_score(self):
        one, five = self.score(1), self.score(5)
        self.assertEqual(one["masked_residues"], five["masked_residues"])
        self.assertAlmostEqual(one["sequence_mean_nll"], five["sequence_mean_nll"], places=5)
        for source in SOURCES:
            self.assertAlmostEqual(
                one["source_sequence_mean_nll"][source],
                five["source_sequence_mean_nll"][source],
                places=5,
            )
        self.assertEqual((one["batch_size"], five["batch_size"]), (1, 5))

    def test_crop_and_mask_depend_only_on_the_protein(self):
        tokenizer = self.tokenizer
        residues = np.asarray(tokenizer.canonical_ids * 3, dtype=np.uint8)
        digest = hashlib.sha256(b"protein").digest()
        first = validation_example(residues, digest, residue_limit=14, tokenizer=tokenizer)
        torch.manual_seed(99)  # The global RNG must not affect the fixed crop or mask.
        again = validation_example(residues, digest, residue_limit=14, tokenizer=tokenizer)
        other = hashlib.sha256(b"other").digest()
        other = validation_example(residues, other, residue_limit=14, tokenizer=tokenizer)
        for left, right in zip(first, again, strict=True):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(first[0].numel(), 16)
        self.assertGreater(int((first[1] != -100).sum()), 0)
        self.assertFalse(torch.equal(first[1], other[1]))


if __name__ == "__main__":
    unittest.main()
