import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


class FullCorpusPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / "dev/data/process_full_corpus.py"
        spec = importlib.util.spec_from_file_location("full_corpus_pipeline", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.pipeline = module

    def test_delta_resume_cli_routes_recovery_flags(self) -> None:
        with mock.patch.object(
            self.pipeline, "run_delta_screen", return_value={"status": "test"}
        ) as run:
            self.pipeline.main(
                [
                    "delta-screen",
                    "--query-fasta",
                    "/tmp/query.fasta",
                    "--target-db-root",
                    "/tmp/db",
                    "--output",
                    "/tmp/output",
                    "--mmseqs",
                    "/tmp/mmseqs",
                    "--threads",
                    "64",
                    "--resume",
                ]
            )
        run.assert_called_once()
        self.assertTrue(run.call_args.kwargs["resume"])
        self.assertFalse(run.call_args.kwargs["concurrent_sources"])
        self.assertEqual(run.call_args.kwargs["threads"], 64)

    def test_screen_target_resolves_fresh_cluster_representative_db(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source_root = root / "uniref90"
            database = source_root / "db/representatives"
            database.parent.mkdir(parents=True)
            database.with_suffix(".dbtype").write_bytes(b"\x00\x00\x00\x00")
            verification = source_root / "verification.json"
            verification.write_text("{}\n")
            observed_db, observed_verification = self.pipeline._resolve_screen_target(
                root, "uniref90"
            )
            self.assertEqual(observed_db, database)
            self.assertEqual(observed_verification, verification)

    def test_cross_source_ownership_keeps_one_exact_representative(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            (work / "SHA_BUCKETS_VERIFIED.json").write_text("{}\n")
            shared_sequence = "ACDEFGHIKLMNPQRSTVWY" * 3
            shared_digest = sequence_digest(shared_sequence)
            shared_bucket = int(shared_digest[:2], 16)
            for source_index, source in enumerate(self.pipeline.SOURCES):
                source_root = work / source
                source_root.mkdir()
                for bucket in range(256):
                    path = source_root / f"bucket-{bucket:02x}.tsv"
                    if bucket == shared_bucket:
                        unique_sequence = chr(65 + source_index) * (60 + source_index)
                        unique_digest = sequence_digest(unique_sequence)
                        rows = [(shared_digest, shared_sequence)]
                        if int(unique_digest[:2], 16) == bucket:
                            rows.append((unique_digest, unique_sequence))
                        path.write_text(
                            "".join(f"{digest}\t{sequence}\n" for digest, sequence in rows)
                        )
                    else:
                        path.write_text("")
            receipt = self.pipeline.build_cross_source_ownership(work)
            self.assertEqual(receipt["cross_source_shared_unique_sequences"], 1)
            self.assertEqual(receipt["removed_duplicate_source_memberships"], 2)
            self.assertEqual(receipt["nonowner_artifacts"]["uniref90"]["records"], 0)
            self.assertEqual(receipt["nonowner_artifacts"]["mgnify"]["records"], 1)
            self.assertEqual(receipt["nonowner_artifacts"]["omg_img"]["records"], 1)

    def test_evaluation_union_protects_blocked_pairs_and_wildtypes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            legacy = root / "legacy"
            q9 = root / "q9"
            (q9 / "tasks").mkdir(parents=True)
            legacy.mkdir()
            legacy_sequence = "A" * 60
            legacy_digest = sequence_digest(legacy_sequence)
            (legacy / "evaluation_all_splits.fasta").write_text(
                f">sha256_{legacy_digest}\n{legacy_sequence}\n"
            )
            (legacy / "sequence_memberships.jsonl").write_text(
                json.dumps(
                    {
                        "sha256": legacy_digest,
                        "memberships": [
                            {
                                "task": "contact",
                                "split": "evaluation",
                                "role": "test",
                            }
                        ],
                    }
                )
                + "\n"
            )
            (legacy / "EVALUATION_SPLIT_LEDGER.json").write_text("{}\n")

            task_receipts = {}
            expected_sequences = {legacy_sequence}
            for task_index, (task, fields) in enumerate(self.pipeline.Q9_TASKS.items()):
                row = {"split": "train"}
                for field_index, field in enumerate(fields):
                    sequence = chr(67 + task_index + field_index) * 60
                    row[field] = sequence
                    row[f"{field}_sha256"] = sequence_digest(sequence)
                    if field == "sequence":
                        row["sequence_sha256"] = sequence_digest(sequence)
                    expected_sequences.add(sequence)
                path = q9 / "tasks" / f"{task}.jsonl"
                path.write_text(json.dumps(row) + "\n")
                task_receipts[task] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            (q9 / "DATASET_RECEIPT.json").write_text(
                json.dumps({"task_receipts": task_receipts})
            )
            output = root / "union"
            report = self.pipeline.build_evaluation_union(
                legacy_root=legacy, q9_root=q9, output=output
            )
            self.assertEqual(report["union_unique_sequences"], len(expected_sequences))
            self.assertTrue(
                report["tasks"]["cafa5_mf_nk30_hard"]["blocked_for_scoring_but_protected"]
            )
            fasta_sequences = {
                sequence
                for _header, sequence in self.pipeline.iter_fasta(
                    output / "evaluation_all_splits.fasta"
                )
            }
            self.assertEqual(fasta_sequences, expected_sequences)

    def test_shard_and_verify_exclusions_and_global_validation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            clusters = root / "clusters"
            exact_sequence = "W" * 40
            homology_sequence = "Y" * 41
            exact_digest = sequence_digest(exact_sequence)
            homology_digest = sequence_digest(homology_sequence)
            for source_index, source in enumerate(self.pipeline.SOURCES):
                source_root = clusters / source
                source_root.mkdir(parents=True)
                rows = [(exact_digest, exact_sequence), (homology_digest, homology_sequence)]
                rows.extend(
                    (
                        sequence_digest(sequence),
                        sequence,
                    )
                    for sequence in (
                        chr(65 + source_index) * (50 + index) for index in range(5)
                    )
                )
                rows = sorted(set(rows))
                fasta = source_root / "representatives.fasta"
                with fasta.open("w") as handle:
                    for digest, sequence in rows:
                        handle.write(f">sha256_{digest}\n{sequence}\n")
                (source_root / "verification.json").write_text(
                    json.dumps(
                        {
                            "clusters": len(rows),
                            "representative_fasta_sha256": hashlib.sha256(
                                fasta.read_bytes()
                            ).hexdigest(),
                        }
                    )
                )
            evaluation = root / "evaluation"
            screen = root / "screen"
            evaluation.mkdir()
            screen.mkdir()
            (evaluation / "evaluation_exact_sha256.txt").write_text(exact_digest + "\n")
            (screen / "homology_excluded_all_splits.txt").write_text(homology_digest + "\n")
            (screen / "HOMOLOGY_EXCLUSION_VERIFIED.json").write_text(
                json.dumps(
                    {
                        "evaluation_protocols": [
                            "contact-p-at-l",
                            "pcore-v0.2",
                            "pcore-v0.5-alpha-q9",
                        ],
                        "thresholds": self.pipeline.MMSEQS_THRESHOLDS,
                    }
                )
            )
            release = root / "release"
            self.pipeline.shard_release(
                cluster_root=clusters,
                screen_root=screen,
                evaluation_root=evaluation,
                output=release,
                validation_per_source=1,
                shard_residues=120,
            )
            receipt = self.pipeline.verify_release(
                release, screen_root=screen, evaluation_root=evaluation
            )
            self.assertEqual(receipt["status"], "verified")
            manifest = json.loads((release / "manifest.json").read_text())
            self.assertEqual(manifest["rejected"]["evaluation_exact"], 3)
            self.assertEqual(manifest["rejected"]["evaluation_homology"], 3)
            self.assertEqual(
                manifest["verification"]["exact_and_homology_exclusion_intersection"],
                0,
            )
            self.assertEqual(
                manifest["verification"]["global_train_exact_duplicate_intersection"],
                0,
            )

    def test_stage_metadata_removes_template_warning_and_binds_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            release = root / "release"
            template = root / "template"
            screen = root / "screen"
            evaluation = root / "evaluation"
            for path in (release, template, screen, evaluation):
                path.mkdir()
            manifest = {
                "status": "verified",
                "protocol": self.pipeline.RELEASE_PROTOCOL,
                "sources": {
                    source: {
                        "representative_records_scanned": 10,
                        "train_records": 8,
                        "train_residues": 800,
                        "train": [{"records": 8, "residues": 800, "bytes": 500}],
                        "validation": [{"records": 2, "residues": 200, "bytes": 100}],
                        "rejected": {},
                    }
                    for source in self.pipeline.SOURCES
                },
            }
            manifest_path = release / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            (release / "RELEASE_VERIFIED.json").write_text(
                json.dumps(
                    {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
                )
            )
            warning = (
                "Do not use this template as a release receipt. Measured post-Q9 counts, "
                "bytes,\n"
                "checksums, and the immutable Hub revision are inserted only after the full "
                "shard\n"
                "verifier succeeds.\n\n"
            )
            (template / "README.md").write_text("# Card\n\n" + warning)
            (template / "LICENSE_AND_ATTRIBUTION.md").write_text("license\n")
            (template / "SOURCE_PROVENANCE.template.json").write_text(
                json.dumps(
                    {
                        "warning": "template",
                        "source_arms": {source: {} for source in self.pipeline.SOURCES},
                    }
                )
            )
            (screen / "HOMOLOGY_EXCLUSION_VERIFIED.json").write_text(
                json.dumps({"excluded_training_representatives": 7})
            )
            (evaluation / "EVALUATION_SPLIT_LEDGER.json").write_text(
                json.dumps({"union_unique_sequences": 11})
            )
            omg_manifest = root / "omg.tsv"
            omg_manifest.write_text("path\tbytes\tsha256\n")

            receipt = self.pipeline.stage_release_metadata(
                release_root=release,
                template_root=template,
                omg_manifest=omg_manifest,
                screen_root=screen,
                evaluation_root=evaluation,
            )

            self.assertEqual(receipt["status"], "verified")
            card = (release / "README.md").read_text()
            self.assertNotIn("Do not use this template", card)
            self.assertIn("Training representatives: **24**", card)
            self.assertIn("Training residues: **2,400**", card)
            self.assertIn("Training Parquet shards: **3**", card)
            self.assertIn("Validation representatives: **6**", card)
            self.assertIn("Validation residues: **600**", card)
            self.assertIn("Train plus validation compressed bytes: **1,800**", card)
            provenance = json.loads((release / "SOURCE_PROVENANCE.json").read_text())
            self.assertNotIn("warning", provenance)
            source_provenance = provenance["source_arms"]["uniref90"]
            self.assertEqual(source_provenance["release_train_shards"], 1)
            self.assertEqual(source_provenance["release_train_compressed_bytes"], 500)
            self.assertEqual(source_provenance["release_validation_residues"], 200)
            for relative, artifact in receipt["artifacts"].items():
                path = release / relative
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"]
                )

    def test_finalize_screen_unions_parent_and_q9_targets(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            evaluation = root / "evaluation"
            delta = root / "delta"
            clusters = root / "clusters"
            evaluation.mkdir()
            (delta / "results").mkdir(parents=True)

            query_sequence = "ACDEFGHIKLMNPQRSTVWY" * 3
            query_digest = sequence_digest(query_sequence)
            query_fasta = evaluation / "evaluation_q9_delta.fasta"
            query_fasta.write_text(f">sha256_{query_digest}\n{query_sequence}\n")
            all_query_fasta = evaluation / "evaluation_all_splits.fasta"
            all_query_fasta.write_bytes(query_fasta.read_bytes())
            exact = evaluation / "evaluation_exact_sha256.txt"
            exact.write_text(query_digest + "\n")
            (evaluation / "EVALUATION_SPLIT_LEDGER.json").write_text(
                json.dumps(
                    {
                        "evaluation_protocols": [
                            "contact-p-at-l",
                            "pcore-v0.2",
                            "pcore-v0.5-alpha-q9",
                        ],
                        "union_unique_sequences": 1,
                        "artifacts": {
                            "evaluation_q9_delta.fasta": {
                                "sha256": hashlib.sha256(query_fasta.read_bytes()).hexdigest()
                            },
                            "evaluation_all_splits.fasta": {
                                "sha256": hashlib.sha256(
                                    all_query_fasta.read_bytes()
                                ).hexdigest()
                            },
                        },
                    }
                )
            )

            parent_digest = sequence_digest("PARENT")
            parent_exclusions = root / "parent-exclusions.txt"
            parent_exclusions.write_text(parent_digest + "\n")
            representative_hash = "a" * 64
            parent_receipt = root / "parent.json"
            parent_receipt.write_text(
                json.dumps(
                    {
                        "status": "verified",
                        "protocol": self.pipeline.LEGACY_SCREEN_PROTOCOL,
                        "scope_used_for_training": "all evaluation splits",
                        "thresholds": self.pipeline.MMSEQS_THRESHOLDS,
                        "excluded_digest_file_sha256": hashlib.sha256(
                            parent_exclusions.read_bytes()
                        ).hexdigest(),
                        "sources": {
                            source: {"representative_fasta_sha256": representative_hash}
                            for source in self.pipeline.SOURCES
                        },
                    }
                )
            )

            artifacts = {}
            expected = {parent_digest}
            for index, source in enumerate(self.pipeline.SOURCES):
                target = sequence_digest(f"target-{index}")
                expected.add(target)
                hit = delta / "results" / f"{source}.tsv"
                hit.write_text(f"{query_digest}\t{target}\t30.0\t60\t0.8\t0.8\t1e-9\t100\n")
                artifacts[source] = {
                    "sha256": hashlib.sha256(hit.read_bytes()).hexdigest(),
                    "target_database": str((root / "db" / source).resolve()),
                }
                source_root = clusters / source
                source_root.mkdir(parents=True)
                (source_root / "verification.json").write_text(
                    json.dumps(
                        {
                            "clusters": 10,
                            "representative_fasta_sha256": representative_hash,
                        }
                    )
                )
            (delta / "MMSEQS_DELTA_SEARCH_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "protocol": "mmseqs2-evaluation-delta-search-v1",
                        "query_scope": "q9-delta",
                        "thresholds": self.pipeline.MMSEQS_THRESHOLDS,
                        "query_fasta_sha256": hashlib.sha256(
                            query_fasta.read_bytes()
                        ).hexdigest(),
                        "artifacts": artifacts,
                    }
                )
            )

            output = root / "screen"
            receipt = self.pipeline.finalize_screen(
                evaluation_root=evaluation,
                parent_receipt=parent_receipt,
                parent_exclusions=parent_exclusions,
                delta_root=delta,
                cluster_root=clusters,
                output=output,
            )

            observed = set(
                (output / "homology_excluded_all_splits.txt").read_text().splitlines()
            )
            self.assertEqual(observed, expected)
            self.assertEqual(receipt["parent_excluded_representatives"], 1)
            self.assertEqual(receipt["delta_unique_targets"], 3)
            self.assertEqual(receipt["excluded_training_representatives"], 4)

            for source in self.pipeline.SOURCES:
                verification = clusters / source / "verification.json"
                artifacts[source].update(
                    {
                        "target_database": str(
                            (clusters / source / "db/representatives").resolve()
                        ),
                        "target_verification_sha256": hashlib.sha256(
                            verification.read_bytes()
                        ).hexdigest(),
                        "target_representative_fasta_sha256": representative_hash,
                    }
                )
            (delta / "MMSEQS_FULL_SEARCH_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "protocol": "mmseqs2-evaluation-full-search-v1",
                        "query_scope": "all-evaluation-splits",
                        "thresholds": self.pipeline.MMSEQS_THRESHOLDS,
                        "query_fasta_sha256": hashlib.sha256(
                            query_fasta.read_bytes()
                        ).hexdigest(),
                        "artifacts": artifacts,
                    }
                )
            )
            full_output = root / "full-screen"
            full_receipt = self.pipeline.finalize_full_screen(
                evaluation_root=evaluation,
                search_root=delta,
                cluster_root=clusters,
                output=full_output,
            )
            self.assertEqual(full_receipt["parent_excluded_representatives"], 0)
            self.assertEqual(full_receipt["full_screen_unique_targets"], 3)
            self.assertEqual(
                set(
                    (full_output / "homology_excluded_all_splits.txt").read_text().splitlines()
                ),
                expected - {parent_digest},
            )


if __name__ == "__main__":
    unittest.main()
