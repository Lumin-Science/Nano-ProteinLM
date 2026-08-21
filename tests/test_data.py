import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nano_protein.data import SOURCES, MixtureBatcher, prepare_dataset
from nano_protein.tokenizer import ProteinTokenizer


class DataContractTests(unittest.TestCase):
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
            )
            self.assertEqual(receipt["decontamination"]["excluded_digest_count"], 2)
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
