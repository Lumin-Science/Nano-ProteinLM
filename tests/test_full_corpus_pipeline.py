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

    def test_download_cli_routes_bounded_workers(self) -> None:
        with mock.patch.object(
            self.pipeline, "download_raw", return_value={"status": "test"}
        ) as download:
            self.pipeline.main(
                [
                    "download",
                    "--data-root",
                    "/tmp/data",
                    "--omg-manifest",
                    "/tmp/omg.tsv",
                    "--download-workers",
                    "16",
                ]
            )
        download.assert_called_once_with(
            Path("/tmp/data"), Path("/tmp/omg.tsv"), download_workers=16
        )

    def test_reproduce_cli_routes_64_cpu_parent_independent_build(self) -> None:
        with mock.patch.object(
            self.pipeline, "reproduce_full_corpus", return_value={"status": "test"}
        ) as reproduce:
            self.pipeline.main(
                [
                    "reproduce",
                    "--data-root",
                    "/tmp/data",
                    "--omg-manifest",
                    "/tmp/omg.tsv",
                    "--legacy-evaluation-root",
                    "/tmp/legacy-eval",
                    "--q9-evaluation-root",
                    "/tmp/q9-eval",
                    "--template-root",
                    "/tmp/template",
                    "--mmseqs",
                    "/tmp/mmseqs",
                ]
            )

        reproduce.assert_called_once_with(
            data_root=Path("/tmp/data"),
            omg_manifest=Path("/tmp/omg.tsv"),
            legacy_evaluation_root=Path("/tmp/legacy-eval"),
            q9_evaluation_root=Path("/tmp/q9-eval"),
            template_root=Path("/tmp/template"),
            mmseqs=Path("/tmp/mmseqs"),
            threads=64,
            download_workers=8,
            partitions=256,
            validation_per_source=4096,
            shard_residues=268435456,
        )

    def test_reproduce_executes_fresh_all_split_pipeline_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data_root = root / "build"
            omg_file = data_root / "raw/omg/chunk.parquet"
            omg_file.parent.mkdir(parents=True)
            omg_file.write_bytes(b"fixture")
            omg_manifest = root / "omg.tsv"
            omg_manifest.write_text("path\tbytes\tsha256\nchunk.parquet\t7\t" + "a" * 64 + "\n")
            verified = {"status": "verified"}
            with (
                mock.patch.object(
                    self.pipeline, "download_raw", return_value=verified
                ) as download,
                mock.patch.object(
                    self.pipeline, "normalize_source", return_value=verified
                ) as normalize,
                mock.patch.object(
                    self.pipeline, "deduplicate", return_value=verified
                ) as deduplicate,
                mock.patch.object(
                    self.pipeline, "run_cluster", return_value=verified
                ) as cluster,
                mock.patch.object(
                    self.pipeline,
                    "build_evaluation_union",
                    return_value={"union_unique_sequences": 317_000},
                ) as evaluation,
                mock.patch.object(
                    self.pipeline, "run_delta_screen", return_value={"status": "complete"}
                ) as search,
                mock.patch.object(
                    self.pipeline,
                    "finalize_full_screen",
                    return_value={"excluded_training_representatives": 123},
                ) as finalize,
                mock.patch.object(self.pipeline, "shard_release") as shard,
                mock.patch.object(
                    self.pipeline,
                    "verify_release",
                    return_value={"manifest_sha256": "b" * 64},
                ) as verify,
                mock.patch.object(
                    self.pipeline, "stage_release_metadata", return_value=verified
                ) as metadata,
                mock.patch.object(self.pipeline, "file_hash", return_value="c" * 64),
            ):
                receipt = self.pipeline.reproduce_full_corpus(
                    data_root=data_root,
                    omg_manifest=omg_manifest,
                    legacy_evaluation_root=root / "legacy",
                    q9_evaluation_root=root / "q9",
                    template_root=root / "template",
                    mmseqs=root / "mmseqs",
                )

            download.assert_called_once()
            self.assertEqual(normalize.call_count, 3)
            deduplicate.assert_called_once()
            self.assertEqual(cluster.call_count, 3)
            evaluation.assert_called_once()
            self.assertEqual(search.call_args.kwargs["query_scope"], "all-evaluation-splits")
            self.assertFalse(search.call_args.kwargs["concurrent_sources"])
            finalize.assert_called_once()
            shard.assert_called_once()
            verify.assert_called_once()
            metadata.assert_called_once()
            self.assertEqual(receipt["status"], "verified")
            self.assertEqual(receipt["evaluation_union_unique_sequences"], 317_000)
            self.assertTrue((data_root / "FULL_CORPUS_REPRODUCTION_VERIFIED.json").is_file())

    def test_reproduce_rejects_existing_derived_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data_root = Path(raw)
            (data_root / "normalized").mkdir()
            with self.assertRaisesRegex(FileExistsError, "fresh derived-stage root"):
                self.pipeline.reproduce_full_corpus(
                    data_root=data_root,
                    omg_manifest=data_root / "omg.tsv",
                    legacy_evaluation_root=data_root / "legacy",
                    q9_evaluation_root=data_root / "q9",
                    template_root=data_root / "template",
                    mmseqs=data_root / "mmseqs",
                )

    def test_download_rejects_nonpositive_workers_before_io(self) -> None:
        with self.assertRaisesRegex(ValueError, "download_workers must be positive"):
            self.pipeline.download_raw(
                Path("/does/not/matter"),
                Path("/does/not/matter.tsv"),
                download_workers=0,
            )

    def test_download_rejects_noncanonical_manifest_before_network_io(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            manifest = Path(raw) / "omg.tsv"
            manifest.write_text("path\tbytes\tsha256\nwrong.parquet\t1\t" + "0" * 64 + "\n")
            with mock.patch.object(self.pipeline, "_download_with_curl") as download:
                with self.assertRaisesRegex(ValueError, "authoritative 959-object pin"):
                    self.pipeline.download_raw(Path(raw) / "data", manifest, download_workers=1)
            download.assert_not_called()

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

    def test_screen_reverses_search_and_normalizes_hit_orientation(self) -> None:
        search, convert = self.pipeline._screen_commands(
            mmseqs=Path("/opt/mmseqs"),
            representative_db=Path("/data/train-representatives"),
            evaluation_db=Path("/data/evaluation-union"),
            result_db=Path("/work/result"),
            temporary=Path("/work/tmp"),
            hit_table=Path("/work/hits.tsv"),
            threads=64,
        )
        self.assertEqual(search[2:4], ["/data/train-representatives", "/data/evaluation-union"])
        self.assertEqual(
            convert[2:4], ["/data/train-representatives", "/data/evaluation-union"]
        )
        self.assertEqual(
            convert[convert.index("--format-output") + 1],
            "target,query,pident,alnlen,tcov,qcov,evalue,bits",
        )
        self.assertEqual(convert[convert.index("--threads") + 1], "64")
        self.assertEqual(search[search.index("-e") + 1], "0.001")
        self.assertEqual(search[search.index("-s") + 1], "7.5")
        self.assertEqual(search[search.index("--max-seqs") + 1], "1000000")

    def test_forward_screen_preserves_safe_evaluation_query_orientation(self) -> None:
        search, convert = self.pipeline._forward_screen_commands(
            mmseqs=Path("/opt/mmseqs"),
            representative_db=Path("/data/train-representatives"),
            evaluation_db=Path("/data/evaluation-union"),
            result_db=Path("/work/result"),
            temporary=Path("/work/tmp"),
            hit_table=Path("/work/hits.tsv"),
            threads=64,
            force_reuse=True,
        )
        self.assertEqual(search[2:4], ["/data/evaluation-union", "/data/train-representatives"])
        self.assertEqual(
            convert[2:4], ["/data/evaluation-union", "/data/train-representatives"]
        )
        self.assertEqual(
            convert[convert.index("--format-output") + 1],
            "query,target,pident,alnlen,qcov,tcov,evalue,bits",
        )
        self.assertEqual(search[search.index("--force-reuse") + 1], "1")

    def test_checkpoint_query_database_requires_byte_identical_reference(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            query_fasta = root / "evaluation_q9_delta.fasta"
            query_fasta.write_text(">sha256_" + "a" * 64 + "\nACDE\n")
            for name in ("query", "reference"):
                prefix = root / name
                prefix.write_bytes(b"sequence-db")
                Path(str(prefix) + ".dbtype").write_bytes(b"\x00\x00\x00\x00")
                Path(str(prefix) + ".lookup").write_text("0\trow\n")
                Path(str(prefix) + ".source").write_text("0\tevaluation_q9_delta.fasta\n")
            receipt = self.pipeline._verify_checkpoint_evaluation_database(
                database=root / "query",
                reference=root / "reference",
                expected_sequences=1,
                query_fasta=query_fasta,
            )
            self.assertEqual(receipt["status"], "verified")
            Path(str(root / "reference") + ".lookup").write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "differs from fresh reference"):
                self.pipeline._verify_checkpoint_evaluation_database(
                    database=root / "query",
                    reference=root / "reference",
                    expected_sequences=1,
                    query_fasta=query_fasta,
                )

    def test_screen_recovery_never_trusts_a_bare_interrupted_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            (output / "results").mkdir()
            (output / "tmp").mkdir()
            (output / "results/uniref90.tsv").write_text("unverified\n")
            (output / "results/uniref90-recovery-01.index").write_text("partial\n")

            result, temporary, hit_table = self.pipeline._next_screen_attempt(
                output, "uniref90", resume=True
            )

            self.assertEqual(result.name, "uniref90-recovery-02")
            self.assertEqual(temporary.name, "uniref90-recovery-02")
            self.assertEqual(hit_table.name, "uniref90-recovery-02.tsv")

    def test_completed_source_search_is_checksum_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            results = output / "results"
            results.mkdir()
            hit_table = results / "mgnify-recovery-01.tsv"
            hit_table.write_text(f"{'a' * 64}\t{'b' * 64}\t30.0\t60\t0.8\t0.8\t1e-9\t100\n")
            target_binding = {"target_database": "/verified/mgnify"}
            artifact = {
                "relative_path": "results/mgnify-recovery-01.tsv",
                "sha256": hashlib.sha256(hit_table.read_bytes()).hexdigest(),
                **target_binding,
            }
            source_receipt = results / "mgnify.complete.json"
            source_receipt.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "protocol": self.pipeline.SOURCE_SEARCH_PROTOCOL,
                        "source": "mgnify",
                        "query_fasta_sha256": "a" * 64,
                        "search_orientation": self.pipeline.SCREEN_SEARCH_ORIENTATION,
                        "normalized_hit_table_schema": (
                            self.pipeline.NORMALIZED_HIT_TABLE_SCHEMA
                        ),
                        "thresholds": self.pipeline.MMSEQS_THRESHOLDS,
                        "sensitivity": self.pipeline.MMSEQS_SENSITIVITY,
                        "artifact": artifact,
                    }
                )
            )

            observed = self.pipeline._validated_source_search(
                output=output,
                source="mgnify",
                source_receipt=source_receipt,
                query_fasta_sha256="a" * 64,
                target_binding=target_binding,
            )
            self.assertTrue(observed["reused_from_interrupted_parent"])
            hit_table.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "hit table changed"):
                self.pipeline._validated_source_search(
                    output=output,
                    source="mgnify",
                    source_receipt=source_receipt,
                    query_fasta_sha256="a" * 64,
                    target_binding=target_binding,
                )

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
                        "search_contracts": {
                            "all_evaluation_splits": {
                                "query_scope": "all-evaluation-splits",
                                "search_orientation": (self.pipeline.SCREEN_SEARCH_ORIENTATION),
                                "normalized_hit_table_schema": (
                                    self.pipeline.NORMALIZED_HIT_TABLE_SCHEMA
                                ),
                                "sensitivity": self.pipeline.MMSEQS_SENSITIVITY,
                                "configured_candidate_cap": 1_000_000,
                                "evaluation_target_sequences": 1,
                                "candidate_cap_unreachable": True,
                            }
                        },
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
            self.assertEqual(manifest["verification"]["global_train_records"], 12)

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
            (screen / "HOMOLOGY_EXCLUSION_VERIFIED.json").write_text(
                json.dumps({"excluded_training_representatives": 7})
            )
            (evaluation / "EVALUATION_SPLIT_LEDGER.json").write_text(
                json.dumps({"union_unique_sequences": 11})
            )
            omg_manifest = root / "omg.tsv"
            omg_manifest.write_text("path\tbytes\tsha256\n")

            observed_manifest_sha256 = hashlib.sha256(omg_manifest.read_bytes()).hexdigest()
            (template / "SOURCE_PROVENANCE.template.json").write_text(
                json.dumps(
                    {
                        "warning": "template",
                        "source_arms": {
                            source: (
                                {"raw_manifest_sha256": observed_manifest_sha256}
                                if source == "omg_img"
                                else {}
                            )
                            for source in self.pipeline.SOURCES
                        },
                    }
                )
            )
            with mock.patch.object(
                self.pipeline, "OMG_MANIFEST_SHA256", observed_manifest_sha256
            ):
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
            (delta / "db").mkdir()
            evaluation_dbtype = delta / "db/evaluation.dbtype"
            evaluation_dbtype.write_bytes(b"\x00\x00\x00\x00")
            evaluation_database_artifacts = {
                "db/evaluation.dbtype": {
                    "bytes": evaluation_dbtype.stat().st_size,
                    "sha256": hashlib.sha256(evaluation_dbtype.read_bytes()).hexdigest(),
                }
            }

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
            parent_commands = root / "MMSEQS_COMMANDS.txt"
            parent_commands.write_text(
                "".join(
                    (
                        f"/opt/mmseqs search {root / 'db/query'} {root / 'db' / source} "
                        f"{root / 'results' / source} {root / 'tmp' / source} "
                        "--min-seq-id 0.30 -c 0.80 --cov-mode 0 "
                        "--max-seqs 1000000 -s 7.5 --threads 64\n"
                    )
                    for source in self.pipeline.SOURCES
                )
            )
            parent_commands_sha256 = hashlib.sha256(parent_commands.read_bytes()).hexdigest()
            parent_search_receipt = root / "MMSEQS_SEARCH_COMPLETE.json"
            parent_search_receipt.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "protocol": "mmseqs2-evaluation-homology-search-v1",
                        "mmseqs_version": "test-mmseqs",
                        "evaluation_split_ledger_sha256": "e" * 64,
                        "threads": 64,
                        "maximum_sequences_per_query": 1_000_000,
                        "minimum_sequence_identity": 0.3,
                        "minimum_query_coverage": 0.8,
                        "minimum_target_coverage": 0.8,
                        "coverage_mode": 0,
                        "commands_sha256": parent_commands_sha256,
                    }
                )
            )
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
                        "command_receipt": str(parent_search_receipt),
                        "command_receipt_sha256": hashlib.sha256(
                            parent_search_receipt.read_bytes()
                        ).hexdigest(),
                        "evaluation_split_ledger_sha256": "e" * 64,
                        "mmseqs_version": "test-mmseqs",
                        "sources": {
                            source: {
                                "representative_fasta_sha256": representative_hash,
                                "maximum_hits_for_one_query": 1,
                            }
                            for source in self.pipeline.SOURCES
                        },
                    }
                )
            )

            artifacts = {}
            orientation_audits = {}
            expected = {parent_digest}
            for index, source in enumerate(self.pipeline.SOURCES):
                target = sequence_digest(f"target-{index}")
                expected.add(target)
                hit = delta / "results" / f"{source}.tsv"
                hit.write_text(f"{query_digest}\t{target}\t30.0\t60\t0.8\t0.8\t1e-9\t100\n")
                source_receipt = delta / "results" / f"{source}.complete.json"
                source_receipt.write_text(json.dumps({"status": "complete"}))
                artifacts[source] = {
                    "sha256": hashlib.sha256(hit.read_bytes()).hexdigest(),
                    "target_database": str((root / "db" / source).resolve()),
                    "search_orientation": self.pipeline.SCREEN_SEARCH_ORIENTATION,
                    "normalized_hit_table_schema": self.pipeline.NORMALIZED_HIT_TABLE_SCHEMA,
                    "source_receipt_relative_path": f"results/{source}.complete.json",
                    "source_receipt_sha256": hashlib.sha256(
                        source_receipt.read_bytes()
                    ).hexdigest(),
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
                audit_root = delta / "audit/evaluation" / source
                audit_root.mkdir(parents=True)
                audit_artifacts = {}
                for name in (
                    "training-sample.keys",
                    "forward-normalized.tsv",
                    "reverse-normalized.tsv",
                ):
                    audit_artifact = audit_root / name
                    audit_artifact.write_text("fixture\n")
                    audit_artifacts[name] = {
                        "bytes": audit_artifact.stat().st_size,
                        "sha256": hashlib.sha256(audit_artifact.read_bytes()).hexdigest(),
                    }
                audit_receipt = audit_root / "ORIENTATION_AUDIT_VERIFIED.json"
                audit_receipt.write_text(
                    json.dumps(
                        {
                            "status": "verified",
                            "protocol": "mmseqs2-search-orientation-audit-v1",
                            "source": source,
                            "sample_training_sequences": 8192,
                            "forward_pairs": 1,
                            "reverse_pairs": 1,
                            "forward_only_pairs": 0,
                            "reverse_recovers_every_forward_pair": True,
                            "artifacts": audit_artifacts,
                        }
                    )
                )
                orientation_audits[source] = {
                    "receipt_relative_path": str(audit_receipt.relative_to(delta)),
                    "receipt_sha256": hashlib.sha256(audit_receipt.read_bytes()).hexdigest(),
                    "sample_training_sequences": 8192,
                    "forward_pairs": 1,
                    "reverse_pairs": 1,
                    "forward_only_pairs": 0,
                    "reverse_recovers_every_forward_pair": True,
                }
            (delta / "MMSEQS_DELTA_SEARCH_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "protocol": "mmseqs2-evaluation-delta-search-v1",
                        "query_scope": "q9-delta",
                        "thresholds": self.pipeline.MMSEQS_THRESHOLDS,
                        "sensitivity": self.pipeline.MMSEQS_SENSITIVITY,
                        "search_orientation": self.pipeline.SCREEN_SEARCH_ORIENTATION,
                        "normalized_hit_table_schema": (
                            self.pipeline.NORMALIZED_HIT_TABLE_SCHEMA
                        ),
                        "candidate_cap_unreachable": True,
                        "evaluation_database_artifacts": evaluation_database_artifacts,
                        "orientation_audits": orientation_audits,
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
                        "sensitivity": self.pipeline.MMSEQS_SENSITIVITY,
                        "search_orientation": self.pipeline.SCREEN_SEARCH_ORIENTATION,
                        "normalized_hit_table_schema": (
                            self.pipeline.NORMALIZED_HIT_TABLE_SCHEMA
                        ),
                        "candidate_cap_unreachable": True,
                        "evaluation_database_artifacts": evaluation_database_artifacts,
                        "orientation_audits": orientation_audits,
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
