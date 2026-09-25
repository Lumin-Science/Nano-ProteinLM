"""Reject corrupt archives and unsafe paths before installing evaluation assets."""

import tempfile
import unittest
from pathlib import Path

from nanoprotein.setup_evaluation import extract_bundle


class EvaluationSetupTests(unittest.TestCase):
    def test_wrong_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "broken.tar.gz"
            archive.write_bytes(b"broken download")
            with self.assertRaisesRegex(ValueError, "checksum"):
                extract_bundle(archive, root / "destination")
            self.assertFalse((root / "destination").exists())
