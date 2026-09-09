"""Reject corrupt archives and unsafe paths before installing evaluation assets."""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanoprotein.data import file_sha256
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

    def test_safe_files_extract_but_traversal_and_links_fail(self):
        for name, link in [
            ("contact/manifest.json", False),
            ("../escape", False),
            ("contact/link", True),
        ]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / "data.tar.gz"
                with tarfile.open(archive, "w:gz") as handle:
                    info = tarfile.TarInfo(name)
                    if link:
                        info.type = tarfile.SYMTYPE
                        info.linkname = "../../escape"
                        handle.addfile(info)
                    else:
                        info.size = 2
                        handle.addfile(info, io.BytesIO(b"{}"))
                destination = root / "unpacked"
                with patch("nanoprotein.setup_evaluation.BUNDLE_SHA256", file_sha256(archive)):
                    if name == "contact/manifest.json":
                        extract_bundle(archive, destination)
                        self.assertEqual((destination / name).read_bytes(), b"{}")
                    else:
                        with self.assertRaises(ValueError):
                            extract_bundle(archive, destination)
                        self.assertFalse(destination.exists())
