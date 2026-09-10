"""Prevent accidental repeated-data scaling and check distributed exhaustion."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nanoprotein.data import TokenStore, _ShuffledRows, file_sha256
from nanoprotein.data_budget import data_coverage
from nanoprotein.sharded_data import _materialize_source
from nanoprotein.tokenizer import ProteinTokenizer


def recipe(steps=100_000):
    return {
        "max_steps": steps,
        "stages": [
            {
                "name": "stage1",
                "micro_batch_size": 64,
                "gradient_accumulation": 8,
                "mixture": {"uniref90": 0.36, "mgnify": 0.11, "omg_img": 0.54},
            }
        ],
    }


def manifest(counts):
    return {"sources": {name: {"train": {"records": n}} for name, n in counts.items()}}


class DataBudgetTests(unittest.TestCase):
    def test_real_small_subset_blocks_2048_by_100k(self):
        data = manifest({"uniref90": 2430914, "mgnify": 1437829, "omg_img": 3240726})
        with self.assertRaisesRegex(ValueError, "capacity is insufficient"):
            data_coverage(recipe(), data, world_size=4)
        config = {**recipe(), "data_resampling": "allow"}
        receipt = data_coverage(config, data, world_size=4)
        self.assertEqual(receipt["status"], "insufficient")
        self.assertAlmostEqual(
            receipt["sources"]["omg_img"]["expected_exposures"], 33.7878, places=3
        )

    def test_source_specific_capacity_and_accumulation(self):
        data = manifest({"uniref90": 74_175_974, "mgnify": 328_949_335, "omg_img": 262_845_186})
        receipt = data_coverage(recipe(), data, world_size=4)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["stages"][0]["draws"], 204_800_000)
        self.assertEqual(
            receipt["sources"]["uniref90"]["required_records_with_headroom"], 73_728_000
        )
        # A huge total corpus cannot compensate for an exhausted mixture arm.
        data["sources"]["uniref90"]["train"]["records"] = 1_000_000
        with self.assertRaisesRegex(ValueError, "uniref90"):
            data_coverage(recipe(), data, world_size=4)

    def test_two_stage_counts_match_step_boundary(self):
        config = recipe(10)
        config["stage1_fraction"] = 0.65
        config["stages"].append({**config["stages"][0], "name": "stage2"})
        receipt = data_coverage(
            config,
            manifest(dict.fromkeys(["uniref90", "mgnify", "omg_img"], 1_000_000)),
            world_size=4,
        )
        self.assertEqual([s["steps"] for s in receipt["stages"]], [7, 3])
        self.assertEqual(sum(s["draws"] for s in receipt["stages"]), 20480)

    def test_rank_partitions_never_repeat_even_with_uneven_source_draws(self):
        seen = []
        for rank in range(4):
            sampler = _ShuffledRows(
                103, seed=9, rank=rank, world_size=4, allow_resampling=False
            )
            rows = [sampler.next() for _ in range(len(sampler.rows))]
            seen.extend(rows)
            with self.assertRaisesRegex(RuntimeError, "resampling is disabled"):
                sampler.next()
            self.assertEqual(sampler.epoch, 0)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), set(range(103)))

    def test_intentional_legacy_resampling_remains_available(self):
        sampler = _ShuffledRows(3, seed=9, rank=0, world_size=1, allow_resampling=True)
        self.assertEqual(len([sampler.next() for _ in range(7)]), 7)
        self.assertEqual(sampler.epoch, 2)

    def test_large_corpus_tokenization_matches_existing_tokenizer_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = {}
            sequences = ["ACDEFGHIKLMNPQRSTVWY", "XXBZO.-|U", "acdy", "?Q"]
            sequences.sort(key=lambda s: hashlib.sha256(s.encode()).hexdigest())
            for split in ("train", "validation"):
                p = root / f"{split}.parquet"
                pq.write_table(
                    pa.table(
                        {
                            "sequence": sequences,
                            "sha256": [
                                hashlib.sha256(s.encode()).hexdigest() for s in sequences
                            ],
                            "length": [len(s) for s in sequences],
                        }
                    ),
                    p,
                )
                selected[split] = [
                    {
                        "path": p.name,
                        "sha256": file_sha256(p),
                        "records": len(sequences),
                        "residues": sum(map(len, sequences)),
                    }
                ]
            _materialize_source(("uniref90", selected, root, root / "out"))
            store = TokenStore.open(root / "out/uniref90/train")
            for i, sequence in enumerate(sequences):
                np.testing.assert_array_equal(
                    store.sequence(i), ProteinTokenizer.esmc().encode_residues(sequence)
                )
            selected["train"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                _materialize_source(("uniref90", selected, root, root / "bad"))


if __name__ == "__main__":
    unittest.main()
