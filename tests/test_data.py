import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nano_protein.data import SOURCES, MixtureBatcher, _ShuffledRows, prepare_dataset
from nano_protein.tokenizer import ProteinTokenizer


class DataContractTests(unittest.TestCase):
    def test_shuffled_rows_are_rank_disjoint_within_an_epoch(self) -> None:
        rank_zero = _ShuffledRows(12, seed=7, rank=0, world_size=2)
        rank_one = _ShuffledRows(12, seed=7, rank=1, world_size=2)
        rows_zero = {rank_zero.next() for _ in range(6)}
        rows_one = {rank_one.next() for _ in range(6)}
        self.assertFalse(rows_zero & rows_one)
        self.assertEqual(rows_zero | rows_one, set(range(12)))

    def test_prepare_split_and_exact_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            clusters = root / "clusters"
            records = []
            for index in range(40):
                sequence = "ACDEFGHIKLMNPQRSTVWY" + "A" * (12 + index)
                digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
                records.append((digest, sequence))
            records.sort()
            excluded_digest, excluded_sequence = records[0]
            homology_digest, _homology_sequence = records[1]
            for source in SOURCES:
                source_root = clusters / source
                source_root.mkdir(parents=True)
                with (source_root / "representatives.fasta").open("w") as handle:
                    for digest, sequence in records:
                        handle.write(f">sha256_{digest}\n{sequence}\n")
                (source_root / "verification.json").write_text(
                    json.dumps({"source": source, "clusters": len(records)})
                )
            pcore = root / "pcore.jsonl"
            pcore.write_text(
                json.dumps({"sha256": excluded_digest, "sequence": excluded_sequence}) + "\n"
            )
            contact = root / "contact.jsonl"
            contact.write_text(json.dumps({"sequence_sha256": "f" * 64}) + "\n")
            homology_digests = root / "homology.txt"
            homology_digests.write_text(homology_digest + "\n")
            homology_receipt = root / "homology.json"
            homology_receipt.write_text(
                json.dumps(
                    {
                        "status": "verified",
                        "protocol": "mmseqs2-evaluation-homology-exclusion-v1",
                        "excluded_digest_file_sha256": hashlib.sha256(
                            homology_digests.read_bytes()
                        ).hexdigest(),
                        "sources": {
                            source: {
                                "screening_coverage": {
                                    "original_source_records_scanned": len(records)
                                }
                            }
                            for source in SOURCES
                        },
                    }
                )
            )
            output = root / "prepared"
            receipt = prepare_dataset(
                cluster_root=clusters,
                output_root=output,
                pcore_index=pcore,
                contact_manifest=contact,
                train_per_source=4,
                validation_per_source=2,
                validation_modulus=2,
                minimum_length=1,
                homology_exclusion_digests=homology_digests,
                homology_exclusion_receipt=homology_receipt,
            )
            self.assertEqual(receipt["decontamination"]["excluded_digest_count"], 3)
            self.assertTrue(receipt["decontamination"]["homology_exclusion"])
            self.assertTrue(
                all(
                    receipt["sources"][source]["scanned_records"] <= len(records)
                    for source in SOURCES
                )
            )
            batcher = MixtureBatcher(
                output,
                "train",
                {"uniref90": 1.0, "mgnify": 1.0, "omg_img": 1.0},
                seed=11,
            )
            tokens, mask = batcher.batch(
                6, context_length=32, tokenizer=ProteinTokenizer.esmc()
            )
            self.assertEqual(tokens.shape, (6, 32))
            self.assertEqual(mask.shape, (6, 32))
            self.assertTrue(mask[:, 0].all())


if __name__ == "__main__":
    unittest.main()
