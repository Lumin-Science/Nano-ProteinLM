import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
