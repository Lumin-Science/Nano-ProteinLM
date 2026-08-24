import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nano_protein.evaluate import (
    bootstrap_mean_interval,
    merge_contact_evaluation,
    merge_full_evaluation,
)


class EvaluationContractTests(unittest.TestCase):
    def test_constant_contact_bootstrap_is_exact(self) -> None:
        result = bootstrap_mean_interval(
            np.full(7, 0.25),
            replicates=200,
            seed=20260820,
        )
        self.assertEqual(result["replicates"], 200)
        self.assertEqual(result["unit_count"], 7)
        self.assertEqual(result["confidence_interval_95"], [0.25, 0.25])

    def test_full_evaluation_merge_preserves_contact_shards(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint_sha = "a" * 64
            chains = ["chain-a", "chain-b", "chain-c"]
            ranked = sorted(
                chains,
                key=lambda chain: hashlib.sha256(f"20260820:{chain}".encode()).digest(),
            )
            contact_paths = []
            for shard in range(3):
                path = root / f"contact-{shard}.json"
                row = {
                    "chain_id": ranked[shard],
                    "precision_at_l": 0.5,
                    "random_precision_at_l": 0.1,
                }
                path.write_text(
                    json.dumps(
                        {
                            "checkpoint_sha256": checkpoint_sha,
                            "timing_seconds": {"total": float(shard + 1)},
                            "peak_cuda_memory_bytes": shard + 1,
                            "contact": {
                                "shard_index": shard,
                                "shard_count": 3,
                                "selection_total_chains": 3,
                                "evaluation_chains": 1,
                                "precision_at_l_uncertainty": {"replicates": 0},
                                "selected_C": 1.0,
                                "validation_trace": [{"C": 1.0}],
                                "rows": [row],
                            },
                        }
                    )
                )
                contact_paths.append(path)
            pcore = root / "pcore.json"
            pcore.write_text(
                json.dumps(
                    {
                        "checkpoint": "/checkpoint.pt",
                        "checkpoint_sha256": checkpoint_sha,
                        "checkpoint_training_seconds": 12.0,
                        "validation_mlm": {"sequence_mean_nll": 1.0},
                        "pcore": {"protocol": "pcore-v0.2"},
                        "timing_seconds": {"total": 4.0},
                        "peak_cuda_memory_bytes": 9,
                    }
                )
            )
            report = merge_full_evaluation(
                contact_paths=contact_paths,
                pcore_path=pcore,
                expected_contact_chains=3,
                contact_bootstrap=20,
            )
            self.assertEqual(report["contact"]["evaluation_chains"], 3)
            self.assertEqual(report["contact"]["precision_at_l"], 0.5)
            self.assertEqual(report["timing_seconds"]["parallel_critical_path"], 4.0)

    def test_contact_only_merge_is_exact_across_oversubscribed_shards(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint_sha = "b" * 64
            chains = [f"chain-{index}" for index in range(7)]
            ranked = sorted(
                chains,
                key=lambda chain: hashlib.sha256(f"20260820:{chain}".encode()).digest(),
            )
            contact_paths = []
            shard_count = 4
            for shard in range(shard_count):
                shard_rows = [
                    {
                        "chain_id": chain,
                        "precision_at_l": (position + 1) / 10,
                        "random_precision_at_l": 0.1,
                    }
                    for position, chain in enumerate(ranked)
                    if position % shard_count == shard
                ]
                path = root / f"contact-{shard}.json"
                path.write_text(
                    json.dumps(
                        {
                            "checkpoint_sha256": checkpoint_sha,
                            "contact": {
                                "protocol": "esmc-paper-contact-lite-v1",
                                "shard_index": shard,
                                "shard_count": shard_count,
                                "selection_total_chains": len(chains),
                                "evaluation_chains": len(shard_rows),
                                "precision_at_l_uncertainty": {"replicates": 0},
                                "selected_C": 1.0,
                                "validation_trace": [{"C": 1.0}],
                                "rows": shard_rows,
                            },
                        }
                    )
                )
                contact_paths.append(path)
            receipt = merge_contact_evaluation(
                contact_paths=contact_paths,
                expected_contact_chains=len(chains),
            )
            self.assertEqual(receipt["checkpoint_sha256"], checkpoint_sha)
            self.assertEqual(receipt["evaluation_chains"], len(chains))
            self.assertAlmostEqual(receipt["p_at_l"], 0.4)
            self.assertAlmostEqual(receipt["random_p_at_l"], 0.1)
            self.assertEqual(len(receipt["components"]), shard_count)


if __name__ == "__main__":
    unittest.main()
