"""Whole-shard budgets preserve every validation shard and the benchmark prefix."""

import unittest
from unittest.mock import patch

from nanoprotein.data import SOURCES
from nanoprotein.sharded_data import plan_shards


def release():
    counts = {
        "uniref90": [810761, 811150, 809003, 810000],
        "mgnify": [1437829, 1400000, 1400000, 1400000],
        "omg_img": [1078627, 1080000, 1082099, 1080000],
    }
    return {
        "sources": {
            source: {
                split: [
                    {
                        "path": f"{split}/{source}/{i}.parquet",
                        "records": n,
                        "residues": n * 100,
                        "bytes": n * 50,
                    }
                    for i, n in enumerate(rows)
                ]
                for split, rows in [("train", counts[source]), ("validation", [4096])]
            }
            for source in SOURCES
        }
    }


class ShardSelectionTests(unittest.TestCase):
    def plan(self, **budget):
        # Selection is independent of release provenance, validated at the API boundary.
        with patch("nanoprotein.sharded_data.validate_release_manifest"):
            return plan_shards(
                release(), weights=dict(zip(SOURCES, [36, 11, 54], strict=True)), **budget
            )

    def test_seven_shards_preserve_benchmark_selection(self):
        old = self.plan(total_training_samples=5376000)
        new = self.plan(total_training_shards=7)
        self.assertEqual(old["paths"], new["paths"])
        self.assertEqual(new["selected_records_including_validation"], 7109469 + 12288)

    def test_larger_budgets_extend_prefixes_without_changing_validation(self):
        previous = {source: [] for source in SOURCES}
        for count in range(3, 13):
            plan = self.plan(total_training_shards=count)
            self.assertEqual(sum(len(s["train"]) for s in plan["sources"].values()), count)
            for source, selected in plan["sources"].items():
                self.assertEqual(selected["train"][: len(previous[source])], previous[source])
                self.assertEqual(
                    selected["validation"], release()["sources"][source]["validation"]
                )
                previous[source] = selected["train"]
        for source in SOURCES:
            self.assertEqual(previous[source], release()["sources"][source]["train"])

    def test_invalid_and_ambiguous_budgets_fail(self):
        for budget in [
            {},
            {"total_training_shards": 2},
            {"total_training_shards": 13},
            {"total_training_shards": 7, "total_training_samples": 100},
            {"total_training_samples": 0},
        ]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.plan(**budget)
