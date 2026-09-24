"""Metadata cleanup preserves verification of existing scientific evaluation assets."""

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from nanoprotein import setup_evaluation as setup


def digest(value):
    return hashlib.sha256(value).hexdigest()


class EvaluationManifestCompatibilityTests(unittest.TestCase):
    def test_current_and_historical_metadata_verify_identical_payloads(self):
        code = b"frozen scientific implementation\n"
        manifests = [
            json.dumps({"files": {"evaluator.py": digest(code)}}).encode(),
            json.dumps(
                {"files": {"evaluator.py": digest(code)}, "origin": "historical metadata"}
            ).encode(),
        ]
        fake = types.ModuleType("autoresearch_esm.paper_contact_runtime")
        fake.ContactDataset = lambda root: types.SimpleNamespace(
            train_ids=list(range(20)), eval_ids=list(range(20775))
        )
        for manifest in manifests:
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                (root / "contact").mkdir()
                (root / "source").mkdir()
                (root / "contact/CONTACT_MANIFEST.jsonl").write_bytes(b"fixed chains\n")
                (root / "contact/PAYLOAD_INVENTORY.json").write_bytes(b"[]")
                (root / "contact/PDB_CONTACT_DATASET_READY.json").write_text(
                    '{"manifest_path":"CONTACT_MANIFEST.jsonl"}'
                )
                (root / "source/SOURCE_MANIFEST.json").write_bytes(manifest)
                (root / "source/evaluator.py").write_bytes(code)
                with (
                    patch.multiple(
                        setup,
                        MANIFEST_SHA256=digest(b"fixed chains\n"),
                        INVENTORY_SHA256=digest(b"[]"),
                        SOURCE_MANIFEST_SHA256=digest(manifests[0]),
                        LEGACY_SOURCE_MANIFEST_SHA256=digest(manifests[1]),
                    ),
                    patch.dict(
                        "sys.modules", {"autoresearch_esm.paper_contact_runtime": fake}
                    ),
                ):
                    receipt = setup.verify_evaluation(root)
                    self.assertEqual(receipt["source_manifest_sha256"], digest(manifest))
                    self.assertEqual(receipt["evaluation_chains"], 20775)
                    (root / "source/evaluator.py").write_bytes(b"changed scorer")
                    with self.assertRaisesRegex(ValueError, "source changed"):
                        setup.verify_evaluation(root)


if __name__ == "__main__":
    unittest.main()
