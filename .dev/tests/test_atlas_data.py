import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nanoprotein.atlas_data import StoreWriter, finalize, normalize, save
from nanoprotein.data import SOURCES, MixtureBatcher, TokenStore
from nanoprotein.global_sampling import GlobalMixtureBatcher
from nanoprotein.tokenizer import ProteinTokenizer
from nanoprotein.train import training_stop_reason, validate_data_manifest


class AtlasDataTests(unittest.TestCase):
    def test_normalization_and_fixed_width_digest(self):
        self.assertEqual(normalize(" a " * 60 + "*"), b"A" * 60)
        self.assertIsNone(normalize("A" * 59))
        self.assertIsNone(normalize("X" * 60))
        with tempfile.TemporaryDirectory() as directory:
            writer = StoreWriter(directory)
            digest = b"d" * 31 + b"\0"
            writer.append(b"A" * 60, digest)
            writer.finish()
            store = TokenStore.open(Path(directory))
            self.assertEqual(store.index["digest"].tobytes(), digest)
            self.assertEqual(store.sequence(0).tolist(), [5] * 60)

    def test_merge_preserves_validation_deduplicates_and_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "validation-parent"
            old = {"sources": {}}
            for source in SOURCES:
                writer = StoreWriter(validation / source / "validation")
                writer.append(b"W" * 60, hashlib.sha256(b"W" * 60).digest())
                old["sources"][source] = {"validation": writer.finish()}
            save(validation / "manifest.json", old)
            sequences = [b"A" * 60, b"C" * 70, b"D" * 80]
            parts = []
            for i, rows in enumerate((sequences[:2], sequences[1:])):
                writer = StoreWriter(root / "partitions" / f"{i:05d}" / "store")
                for sequence in rows:
                    writer.append(sequence, hashlib.sha256(sequence).digest())
                parts.append(dict(part=i, store=writer.finish()))
            finalize(root, parts, validation, {"version": 3})
            with self.assertRaisesRegex(RuntimeError, "training blocked"):
                validate_data_manifest(root / "data")
            manifest = validate_data_manifest(root / "data", allow_unscreened=True)
            self.assertFalse(manifest["decontamination"]["homology_exclusion"])
            store = TokenStore.open(root / "data" / "esm_atlas" / "train")
            self.assertEqual(store.index.size, 3)
            self.assertEqual(store.index["offset"].tolist(), [0, 60, 130])
            tokenizer = ProteinTokenizer.esmc()
            for i, sequence in enumerate(sequences):
                np.testing.assert_array_equal(
                    store.sequence(i), tokenizer.encode_residues(sequence.decode())
                )
            for source in SOURCES:
                self.assertEqual(
                    manifest["sources"][source]["validation"],
                    old["sources"][source]["validation"],
                )
            batcher = MixtureBatcher(
                root / "data",
                "train",
                {"esm_atlas": 1},
                seed=1,
                allow_resampling=False,
            )
            batcher.batch(3, context_length=64, tokenizer=tokenizer)
            with self.assertRaises(RuntimeError):
                batcher.batch(1, context_length=64, tokenizer=tokenizer)
            GlobalMixtureBatcher(
                root / "data",
                "train",
                {"esm_atlas": 1},
                seed=1,
                policies={"esm_atlas": "error"},
                rank=0,
                world_size=1,
            )
            preparation = root / "data" / "ATLAS_PREPARATION.json"
            value = json.loads(preparation.read_text())
            value["training_store"]["records"] = 100
            save(preparation, value)
            with self.assertRaises(ValueError):
                validate_data_manifest(root / "data", allow_unscreened=True)

    def test_allocation_deadline_stops_before_walltime_budget(self):
        reason = training_stop_reason(
            model_tokens=10,
            max_model_tokens=None,
            optimizer_step=10,
            max_steps=100000,
            training_seconds=1,
            walltime_seconds=20000,
            deadline_reached=True,
        )
        self.assertEqual(reason, "allocation_deadline")


if __name__ == "__main__":
    unittest.main()
