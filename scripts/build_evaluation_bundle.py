#!/usr/bin/env python3
"""Freeze exact evaluation splits and FASTAs for corpus decontamination."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def file_sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_fasta(path: Path, rows: Iterable[tuple[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="ascii") as handle:
        for digest, sequence in rows:
            handle.write(f">{digest}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
    temporary.replace(path)


class EvaluationBundle:
    def __init__(self) -> None:
        self.sequences: dict[str, str] = {}
        self.memberships: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        self.split_occurrences: Counter[tuple[str, str, str]] = Counter()
        self.split_units: Counter[tuple[str, str, str]] = Counter()

    def add(
        self,
        task: str,
        split: str,
        role: str,
        sequences: Iterable[str],
        units: int | None = None,
    ) -> None:
        values = list(sequences)
        membership = (task, split, role)
        self.split_units[membership] += len(values) if units is None else units
        for sequence in values:
            digest = sequence_sha256(sequence)
            observed = self.sequences.setdefault(digest, sequence)
            if observed != sequence:
                raise ValueError(f"SHA-256 collision for {digest}")
            self.memberships[digest].add(membership)
            self.split_occurrences[membership] += 1

    def rows(self) -> list[tuple[str, str]]:
        return [(digest, self.sequences[digest]) for digest in sorted(self.sequences)]

    def protected_rows(self) -> list[tuple[str, str]]:
        protected_roles = {"validation", "test", "validation_and_test"}
        return [
            (digest, self.sequences[digest])
            for digest in sorted(self.sequences)
            if any(role in protected_roles for _task, _split, role in self.memberships[digest])
        ]

    def split_report(self) -> list[dict[str, object]]:
        unique: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for digest, memberships in self.memberships.items():
            for membership in memberships:
                unique[membership].add(digest)
        return [
            {
                "task": task,
                "split": split,
                "role": role,
                "evaluation_units": self.split_units[(task, split, role)],
                "record_sequence_occurrences": self.split_occurrences[(task, split, role)],
                "unique_sequences": len(unique[(task, split, role)]),
                "unique_residues": sum(
                    len(self.sequences[digest]) for digest in unique[(task, split, role)]
                ),
            }
            for task, split, role in sorted(self.split_occurrences)
        ]


def _one_sequence(records: Iterable[Any]) -> Iterable[str]:
    for record in records:
        yield record.sequence


def _pair_sequences(records: Iterable[Any]) -> Iterable[str]:
    for record in records:
        yield record.sequence_a
        yield record.sequence_b


def _input_paths(pcore_root: Path, contact_root: Path) -> list[Path]:
    processed = pcore_root / "processed"
    paths = [
        processed / "remote_homology" / f"remote_homology_{split}.lmdb" / "data.mdb"
        for split in ("train", "valid", "test_fold_holdout")
    ]
    paths.extend(
        processed / "secondary_structure" / f"secondary_structure_{split}.lmdb" / "data.mdb"
        for split in ("train", "valid", "cb513")
    )
    paths.extend(
        processed / "human_ppi" / f"human_ppi_{split}.lmdb" / "data.mdb"
        for split in ("train", "valid", "test")
    )
    ec = processed / "EnzymeCommission" / "EnzymeCommission"
    paths.extend(
        [
            ec / "nrPDB-EC_sequences.fasta",
            ec / "nrPDB-EC_annot.tsv",
            ec / "nrPDB-EC_train.txt",
            ec / "nrPDB-EC_valid.txt",
            ec / "nrPDB-EC_test.csv",
            pcore_root / "raw" / "deeploc2_multisub_5_partitions_unique.csv",
            pcore_root / "raw" / "flip2_hydro_low_to_high.csv.gz",
            pcore_root / "index.jsonl",
            contact_root / "PDB_CONTACT_DATASET_READY.json",
            contact_root / "CONTACT_MANIFEST.jsonl",
            contact_root / "PAYLOAD_INVENTORY.json",
        ]
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"evaluation inputs are missing: {missing}")
    return paths


def build_bundle(
    *,
    external_src: Path,
    pcore_root: Path,
    contact_root: Path,
    output_root: Path,
) -> dict[str, object]:
    sys.path.insert(0, str(external_src.resolve()))
    try:
        from autoresearch_esm.paper_contact_runtime import ContactDataset
        from autoresearch_esm.pcore_tasks import (
            load_deeploc2,
            load_enzyme_commission,
            load_flip2_hydro,
            load_human_ppi,
            load_remote_homology,
            load_secondary_structure,
        )
    finally:
        sys.path.pop(0)

    output_root.mkdir(parents=True, exist_ok=True)
    bundle = EvaluationBundle()
    processed = pcore_root / "processed"

    split_roles = {
        "train": "probe_fit",
        "valid": "validation",
        "test": "test",
        "test_fold_holdout": "test",
        "cb513": "test",
    }
    for split in ("train", "valid", "test_fold_holdout"):
        bundle.add(
            "remote_homology",
            split,
            split_roles[split],
            _one_sequence(load_remote_homology(processed, split)),
        )
    for split in ("train", "valid", "cb513"):
        bundle.add(
            "secondary_structure",
            split,
            split_roles[split],
            _one_sequence(load_secondary_structure(processed, split)),
        )
    for split in ("train", "valid", "test"):
        records, _labels = load_enzyme_commission(processed, split, test_identity=0.30)
        bundle.add("enzyme_commission", split, split_roles[split], _one_sequence(records))
    for split in ("train", "valid", "test"):
        records = load_human_ppi(processed, split)
        bundle.add(
            "human_ppi",
            split,
            split_roles[split],
            _pair_sequences(records),
            units=len(records),
        )

    deeploc = load_deeploc2(pcore_root / "raw" / "deeploc2_multisub_5_partitions_unique.csv")
    partition_counts = Counter(int(record.target[0]) for record in deeploc)
    for partition in range(5):
        bundle.add(
            "deeploc2",
            f"partition_{partition}",
            "validation_and_test",
            _one_sequence(record for record in deeploc if int(record.target[0]) == partition),
        )
    deeploc_folds = [
        {
            "test_partition": test,
            "validation_partition": (test + 1) % 5,
            "train_partitions": sorted(set(range(5)) - {test, (test + 1) % 5}),
            "counts": {
                "train": sum(
                    partition_counts[index]
                    for index in set(range(5)) - {test, (test + 1) % 5}
                ),
                "validation": partition_counts[(test + 1) % 5],
                "test": partition_counts[test],
            },
        }
        for test in range(5)
    ]

    flip = load_flip2_hydro(pcore_root / "raw" / "flip2_hydro_low_to_high.csv.gz")
    flip_splits = {
        "train": [row for row in flip if row.target[1] == "train" and not row.target[2]],
        "valid": [row for row in flip if row.target[1] == "train" and row.target[2]],
        "test": [row for row in flip if row.target[1] == "test"],
    }
    for split, records in flip_splits.items():
        bundle.add("flip2_hydro", split, split_roles[split], _one_sequence(records))

    contact = ContactDataset(contact_root)
    for index, chain_id in enumerate(contact.train_ids):
        _payload, chain = contact.load_payload(chain_id)
        if index < 16:
            bundle.add("contact", "probe_train", "probe_fit", [chain.sequence])
        else:
            bundle.add("contact", "probe_valid", "validation", [chain.sequence])
    for chain_id in contact.eval_ids:
        _payload, chain = contact.load_payload(chain_id)
        bundle.add("contact", "evaluation", "test", [chain.sequence])

    all_fasta = output_root / "evaluation_all_splits.fasta"
    protected_fasta = output_root / "evaluation_validation_test.fasta"
    membership_path = output_root / "sequence_memberships.jsonl"
    write_fasta(all_fasta, bundle.rows())
    protected_rows = bundle.protected_rows()
    write_fasta(protected_fasta, protected_rows)
    temporary_membership = membership_path.with_suffix(membership_path.suffix + ".partial")
    with temporary_membership.open("w") as handle:
        for digest in sorted(bundle.sequences):
            handle.write(
                json.dumps(
                    {
                        "sha256": digest,
                        "length": len(bundle.sequences[digest]),
                        "memberships": [
                            {"task": task, "split": split, "role": role}
                            for task, split, role in sorted(bundle.memberships[digest])
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    temporary_membership.replace(membership_path)

    input_paths = _input_paths(pcore_root, contact_root)
    report: dict[str, object] = {
        "schema_version": 1,
        "protocol": "protein-evaluation-split-ledger-v1",
        "policy": {
            "probe_fit": (
                "fits linear/contact probe parameters; never supplies a headline score"
            ),
            "validation": "selects probe hyperparameters; never supplies a headline score",
            "test": "supplies the frozen final metric and uncertainty",
            "homology_exclusion_scope": (
                "validation and test are mandatory; all-splits FASTA is also emitted for the "
                "stronger optional screen"
            ),
        },
        "heldout_mlm": {
            "source": "post-exclusion source-stratified 70%-identity representatives",
            "split": "sha256-prefix validation partition",
            "role": "validation only",
            "test_split": None,
            "note": "not part of the external homology query bundle",
        },
        "pcore": {
            "version": "v0.2",
            "tasks": 6,
            "final_score_source": (
                "test split only; DeepLoc2 concatenates five held-out test folds"
            ),
            "deeploc2_folds": deeploc_folds,
        },
        "contact": {
            "protocol": contact.ready["protocol_id"],
            "probe_fit_chains": 16,
            "probe_validation_chains": 4,
            "test_chains": len(contact.eval_ids),
            "manifest_sha256": contact.ready["manifest_sha256"],
        },
        "splits": bundle.split_report(),
        "union": {
            "all_splits_unique_sequences": len(bundle.sequences),
            "all_splits_unique_residues": sum(
                len(value) for value in bundle.sequences.values()
            ),
            "validation_test_unique_sequences": len(protected_rows),
            "validation_test_unique_residues": sum(
                len(sequence) for _digest, sequence in protected_rows
            ),
        },
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in input_paths
        ],
        "artifacts": {
            all_fasta.name: {
                "sha256": file_sha256(all_fasta),
                "sequences": len(bundle.sequences),
            },
            protected_fasta.name: {
                "sha256": file_sha256(protected_fasta),
                "sequences": len(protected_rows),
            },
            membership_path.name: {"sha256": file_sha256(membership_path)},
        },
    }
    write_json(output_root / "EVALUATION_SPLIT_LEDGER.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-src", type=Path, required=True)
    parser.add_argument("--pcore-root", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = build_bundle(
        external_src=args.external_src,
        pcore_root=args.pcore_root,
        contact_root=args.contact_root,
        output_root=args.output_root,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
