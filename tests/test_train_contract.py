import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from nano_protein.train import validate_data_manifest


def orientation_audit() -> dict[str, object]:
    sources = ("uniref90", "mgnify", "omg_img")
    return {
        "protocol": "mmseqs2-search-orientation-audit-v1",
        "all_sources_reverse_recover_every_forward_pair": True,
        "minimum_sampled_training_sequences_per_source": 8192,
        "sources": {
            source: {
                "sample_training_sequences": 8192,
                "forward_pairs": 10,
                "reverse_pairs": 11,
                "forward_only_pairs": 0,
                "reverse_recovers_every_forward_pair": True,
            }
            for source in sources
        },
    }


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
                            "protocol": "mmseqs2-evaluation-homology-exclusion-v2",
                            "scope_used_for_training": "all evaluation splits",
                            "evaluation_protocols": [
                                "contact-p-at-l",
                                "pcore-v0.2",
                                "pcore-v0.5-alpha-q9",
                            ],
                            "blocked_benchmark_candidates_are_protected": True,
                            "search_contracts": {
                                "all_evaluation_splits": {
                                    "query_scope": "all-evaluation-splits",
                                    "search_orientation": (
                                        "training-representative-query-vs-evaluation-target"
                                    ),
                                    "normalized_hit_table_schema": (
                                        "evaluation_sha256,training_sha256,pident,alnlen,"
                                        "evaluation_coverage,training_coverage,evalue,bits"
                                    ),
                                    "sensitivity": 7.5,
                                    "maximum_evalue": 0.001,
                                    "configured_candidate_cap": 1_000_000,
                                    "evaluation_target_sequences": 317_000,
                                    "candidate_cap_unreachable": True,
                                    "orientation_audit": orientation_audit(),
                                }
                            },
                            "thresholds": {
                                "coverage_mode": 0,
                                "minimum_sequence_identity": 0.3,
                                "minimum_query_coverage": 0.8,
                                "minimum_target_coverage": 0.8,
                                "maximum_evalue": 0.001,
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
                        "protocol": "prepared-corpus-decontamination-verification-v2",
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

    def test_training_rejects_legacy_pre_q9_homology_receipt(self) -> None:
        temporary, root = self._data_root(True)
        self.addCleanup(temporary.cleanup)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["decontamination"]["homology_contract"]["protocol"] = (
            "mmseqs2-evaluation-homology-exclusion-v1"
        )
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "verified MMseqs2 homology"):
            validate_data_manifest(root)

    def test_q9_homology_gate_requires_all_protocols_and_blocked_candidates(self) -> None:
        temporary, root = self._data_root(True)
        self.addCleanup(temporary.cleanup)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        contract = manifest["decontamination"]["homology_contract"]
        validate_data_manifest(root)

        contract["evaluation_protocols"].remove("pcore-v0.5-alpha-q9")
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "verified MMseqs2 homology"):
            validate_data_manifest(root)

    def test_training_rejects_missing_search_provenance(self) -> None:
        temporary, root = self._data_root(True)
        self.addCleanup(temporary.cleanup)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["decontamination"]["homology_contract"]["search_contracts"]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RuntimeError, "verified MMseqs2 homology"):
            validate_data_manifest(root)

    def test_only_supported_production_budget_is_shipped(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        for legacy_path in (
            project_root / "dev" / "configs" / "autoresearch_300m_4xa100_2h.yaml",
            project_root / "dev" / "configs" / "esmc_300m_stage1_4xa100_16h.yaml",
        ):
            with self.subTest(path=legacy_path):
                self.assertFalse(legacy_path.exists())

        config = yaml.safe_load(
            (project_root / "configs" / "esmc_300m_stage1_4xa100_4h.yaml").read_text()
        )
        self.assertEqual(config["max_steps"], 21_000)
        self.assertEqual(config["warmup_steps"], 2_100)
        self.assertEqual(config["walltime_seconds"], 14_400)
        self.assertEqual(config["evaluation"]["contact_chains"], 20_775)


if __name__ == "__main__":
    unittest.main()
