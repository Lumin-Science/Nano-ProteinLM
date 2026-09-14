"""Audit released-model shards against the frozen split and preserve compact results."""

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from nanoprotein.data import file_sha256
from nanoprotein.evaluate import bootstrap_mean_interval, merge_contact_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--external-src", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.external_src))
    from autoresearch_esm.paper_contact_runtime import ContactDataset

    dataset = ContactDataset(args.dataset_root)
    expected = sorted(
        dataset.eval_ids, key=lambda c: hashlib.sha256(f"20260820:{c}".encode()).digest()
    )
    assert len(expected) == 20775
    sources = json.loads((args.run_root / "MODEL_SOURCES.json").read_text())
    for family in ("esm2", "e1"):
        directory = args.run_root / family
        paths = [directory / f"shard-{i}.json" for i in range(4)]
        result = merge_contact_evaluation(contact_paths=paths, expected_contact_chains=20775)
        probe = json.loads((directory / "PROBE.json").read_text())
        assert probe["dataset_manifest_sha256"] == dataset.manifest_receipt.manifest_sha256
        assert probe["probe_train_chain_ids"] == dataset.train_ids[:16]
        assert probe["probe_validation_chain_ids"] == dataset.train_ids[16:]
        assert result["checkpoint_sha256"] == sources[family]["files"]["model.safetensors"]
        rows = []
        for path in paths:
            receipt = json.loads(path.read_text())
            assert receipt["contact"]["probe_receipt_sha256"] == file_sha256(
                directory / "PROBE.json"
            )
            rows.extend(receipt["contact"]["rows"])
        rows.sort(
            key=lambda row: hashlib.sha256(f"20260820:{row['chain_id']}".encode()).digest()
        )
        assert [r["chain_id"] for r in rows] == expected
        values = np.array([r["precision_at_l"] for r in rows], dtype=np.float64)
        uncertainty = bootstrap_mean_interval(values, replicates=5000, seed=20260820)
        result.update(
            family=family,
            model_source=sources[family],
            dataset_manifest_sha256=dataset.manifest_receipt.manifest_sha256,
            probe_receipt_sha256=file_sha256(directory / "PROBE.json"),
            p_at_l_95_ci=uncertainty["confidence_interval_95"],
            uncertainty=uncertainty,
            smoke=json.loads((directory / "SMOKE.json").read_text()),
            chain_coverage_verified=True,
            inference_mode="single_sequence_without_retrieval",
        )
        chain_path = directory / "contact-per-chain.tsv.gz"
        with gzip.open(chain_path, "wt") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        result["per_chain_sha256"] = file_sha256(chain_path)
        (directory / "RESULT_VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"VERIFIED {family} P@L={result['p_at_l']:.9f} CI={result['p_at_l_95_ci']}")


if __name__ == "__main__":
    main()
