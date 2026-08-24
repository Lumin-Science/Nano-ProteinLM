import hashlib
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nano_protein.data import SOURCES, TokenStore
from nano_protein.sharded_data import materialize_plan, plan_shards
from nano_protein.train import validate_data_manifest


def digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


class ShardedDataTests(unittest.TestCase):
    def _release(self, root: Path) -> dict[str, object]:
        sources: dict[str, object] = {}
        for source_index, source in enumerate(SOURCES):
            rows = sorted(
                (
                    {"sequence": "A" * (40 + source_index * 10 + index), "length": 0}
                    for index in range(6)
                ),
                key=lambda row: digest(str(row["sequence"])),
            )
            for row in rows:
                row["length"] = len(str(row["sequence"]))
                row["sha256"] = digest(str(row["sequence"]))
            split_rows = {"train": [rows[:2], rows[2:4]], "validation": [rows[4:]]}
            source_manifest: dict[str, object] = {}
            for split, chunks in split_rows.items():
                shards = []
                for index, chunk in enumerate(chunks):
                    relative = f"{split}/{source}/shard-{index:05d}.parquet"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(pa.Table.from_pylist(chunk), path, compression="zstd")
                    shards.append(
                        {
                            "path": relative,
                            "records": len(chunk),
                            "residues": sum(int(row["length"]) for row in chunk),
                            "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "minimum_sequence_sha256": chunk[0]["sha256"],
                            "maximum_sequence_sha256": chunk[-1]["sha256"],
                        }
                    )
                source_manifest[split] = shards
            sources[source] = source_manifest
        return {
            "status": "verified",
            "protocol": "protein-corpus-parquet-shards-v1",
            "release_id": "tiny",
            "sampling_unit": "test sequence",
            "global_exact_ownership": {"protocol": "global-exact-representative-ownership-v1"},
            "verification": {"global_train_exact_duplicate_intersection": 0},
            "decontamination": {
                "scope": "all_evaluation_splits",
                "evaluation_protocols": [
                    "contact-p-at-l",
                    "pcore-v0.2",
                    "pcore-v0.5-alpha-q9",
                ],
                "blocked_benchmark_candidates_are_protected": True,
                "homology_exclusion_receipt_sha256": "a" * 64,
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
                        "configured_candidate_cap": 1_000_000,
                        "evaluation_target_sequences": 317_000,
                        "candidate_cap_unreachable": True,
                    }
                },
                "thresholds": {
                    "coverage_mode": 0,
                    "minimum_sequence_identity": 0.3,
                    "minimum_query_coverage": 0.8,
                    "minimum_target_coverage": 0.8,
                },
            },
            "sources": sources,
        }

    def test_budget_plan_selects_minimum_prefix_and_all_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release = self._release(root)
            plan = plan_shards(
                release,
                total_training_samples=9,
                weights={source: 1.0 for source in SOURCES},
            )
            for source in SOURCES:
                self.assertEqual(plan["sources"][source]["required_unique_records"], 3)
                self.assertEqual(plan["sources"][source]["selected_unique_records"], 4)
                self.assertEqual(len(plan["sources"][source]["train"]), 2)
                self.assertEqual(len(plan["sources"][source]["validation"]), 1)
            self.assertGreater(plan["selected_residues_including_validation"], 0)
            self.assertGreater(plan["selected_compressed_bytes_including_validation"], 0)
            self.assertEqual(
                plan["selected_records_including_validation"],
                sum(
                    source["selected_unique_records"] + source["selected_validation_records"]
                    for source in plan["sources"].values()
                ),
            )

    def test_budget_plan_accepts_audited_legacy_plus_q9_searches(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release = self._release(root)
            normalized_schema = (
                "evaluation_sha256,training_sha256,pident,alnlen,"
                "evaluation_coverage,training_coverage,evalue,bits"
            )
            release["decontamination"]["search_contracts"] = {
                "legacy_parent": {
                    "query_scope": "p-at-l-and-pcore-v0.2-all-splits",
                    "search_orientation": (
                        "evaluation-query-vs-training-representative-target"
                    ),
                    "normalized_hit_table_schema": normalized_schema,
                    "sensitivity": 7.5,
                    "configured_candidate_cap": 1_000_000,
                    "maximum_emitted_hits_for_one_evaluation_query": 441_788,
                    "all_emitted_hit_counts_below_cap": True,
                    "command_receipt_sha256": "c" * 64,
                },
                "q9_delta": {
                    "query_scope": "q9-delta",
                    "search_orientation": (
                        "training-representative-query-vs-evaluation-target"
                    ),
                    "normalized_hit_table_schema": normalized_schema,
                    "sensitivity": 7.5,
                    "configured_candidate_cap": 1_000_000,
                    "evaluation_target_sequences": 200_159,
                    "candidate_cap_unreachable": True,
                },
            }
            plan = plan_shards(
                release,
                total_training_samples=3,
                weights={source: 1.0 for source in SOURCES},
            )
            self.assertEqual(plan["total_training_samples"], 3)

    def test_budget_plan_rejects_missing_search_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            release = self._release(Path(raw))
            del release["decontamination"]["search_contracts"]
            with self.assertRaisesRegex(ValueError, "search provenance"):
                plan_shards(
                    release,
                    total_training_samples=3,
                    weights={source: 1.0 for source in SOURCES},
                )

    def test_materialized_prefix_passes_training_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release = self._release(root)
            plan = plan_shards(
                release,
                total_training_samples=3,
                weights={source: 1.0 for source in SOURCES},
            )
            plan["release_manifest_sha256"] = "b" * 64
            output = root / "mmap"
            materialize_plan(plan, release, root, output)
            validate_data_manifest(output)
            for source in SOURCES:
                self.assertEqual(TokenStore.open(output / source / "train").index.size, 2)
                self.assertEqual(TokenStore.open(output / source / "validation").index.size, 2)


if __name__ == "__main__":
    unittest.main()
