import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from nano_protein.train import validate_data_manifest


class TrainingDataContractTests(unittest.TestCase):
    def _data_root(self, homology_exclusion: bool) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "decontamination": {
                        "homology_exclusion": homology_exclusion,
                        "homology_exclusion_receipt_sha256": "a" * 64,
                        "homology_contract": {
                            "status": "verified",
                            "protocol": "mmseqs2-evaluation-homology-exclusion-v1",
                            "scope_used_for_training": "all evaluation splits",
                            "thresholds": {
                                "coverage_mode": 0,
                                "minimum_sequence_identity": 0.3,
                                "minimum_query_coverage": 0.8,
                                "minimum_target_coverage": 0.8,
                            },
                        },
                    }
                }
            )
        )
        if homology_exclusion:
            sources = {
                source: {
                    "train_excluded_intersection": 0,
                    "validation_excluded_intersection": 0,
                    "train_validation_intersection": 0,
                }
                for source in ("uniref90", "mgnify", "omg_img")
            }
            (root / "CORPUS_VERIFICATION.json").write_text(
                json.dumps(
                    {
                        "status": "verified",
                        "protocol": "prepared-corpus-decontamination-verification-v1",
                        "manifest_sha256": hashlib.sha256(
                            manifest_path.read_bytes()
                        ).hexdigest(),
                        "homology_exclusion_receipt_sha256": "a" * 64,
                        "sources": sources,
                    }
                )
            )
        return temporary, root

    def test_training_always_rejects_exact_only_manifest(self) -> None:
        temporary, root = self._data_root(False)
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(RuntimeError, "verified MMseqs2 homology"):
            validate_data_manifest(root)

    def test_homology_gate_accepts_certified_manifest(self) -> None:
        temporary, root = self._data_root(True)
        self.addCleanup(temporary.cleanup)
        manifest = validate_data_manifest(root)
        self.assertTrue(manifest["decontamination"]["homology_exclusion"])

    def test_sixteen_hour_campaign_is_step_gated(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "esmc_300m_stage1_4xa100_16h.yaml"
        )
        config = yaml.safe_load(config_path.read_text())

        self.assertEqual(config["max_steps"], 84_000)
        self.assertEqual(config["warmup_steps"], 8_400)
        self.assertEqual(config["walltime_seconds"], 57_600)
        self.assertEqual(config["evaluation"]["pcore_tasks"], "all_six")
        self.assertEqual(config["evaluation"]["pcore_bootstrap"], 10_000)
        self.assertEqual(config["evaluation"]["contact_chains"], 20_775)
        self.assertEqual(config["evaluation"]["contact_bootstrap"], 5_000)


if __name__ == "__main__":
    unittest.main()
