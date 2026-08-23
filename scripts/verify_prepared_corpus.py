#!/usr/bin/env python3
"""Independently verify a prepared corpus against its frozen exclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nano_protein.data import INDEX_DTYPE, SOURCES, evaluation_digests, file_sha256


def _digest_lines(path: Path) -> set[bytes]:
    values: set[bytes] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip().removeprefix("sha256_")
            if not value:
                continue
            try:
                raw = bytes.fromhex(value)
            except ValueError as error:
                raise ValueError(f"invalid digest at {path}:{line_number}") from error
            if len(raw) != 32:
                raise ValueError(f"invalid digest at {path}:{line_number}")
            values.add(raw)
    return values


def _count_members(index: np.ndarray, values: set[bytes], *, chunk: int = 250_000) -> int:
    matches = 0
    if not values:
        return matches
    for start in range(0, index.size, chunk):
        digests = index[start : start + chunk]["digest"]
        matches += sum(bytes(value) in values for value in digests)
    return matches


def _split_report(
    root: Path, expected: dict[str, object]
) -> tuple[dict[str, object], np.ndarray]:
    token_path = root / "tokens.bin"
    index_path = root / "index.npy"
    index = np.load(index_path, mmap_mode="r", allow_pickle=False)
    if index.dtype != INDEX_DTYPE:
        raise ValueError(f"unexpected index dtype at {index_path}: {index.dtype}")
    records = int(index.size)
    residues = int(token_path.stat().st_size)
    if records <= 0:
        raise ValueError(f"empty prepared split at {root}")
    last = index[-1]
    extent = int(last["offset"]) + int(last["length"])
    if extent != residues:
        raise ValueError(f"token/index extent mismatch at {root}: {extent} != {residues}")
    report = {
        "records": records,
        "residues": residues,
        "tokens_sha256": file_sha256(token_path),
        "index_sha256": file_sha256(index_path),
    }
    for key, value in report.items():
        if expected.get(key) != value:
            raise ValueError(
                f"manifest mismatch for {root} {key}: {expected.get(key)} != {value}"
            )
    return report, index


def verify(data_root: Path) -> dict[str, object]:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    decontamination = manifest["decontamination"]
    if decontamination.get("homology_exclusion") is not True:
        raise ValueError("prepared corpus does not claim homology exclusion")
    homology_path = Path(decontamination["homology_exclusion_digests"])
    homology_receipt_path = Path(decontamination["homology_exclusion_receipt"])
    homology_receipt = json.loads(homology_receipt_path.read_text())
    if file_sha256(homology_path) != decontamination["homology_exclusion_digests_sha256"]:
        raise ValueError("homology digest file changed after corpus preparation")
    if (
        file_sha256(homology_receipt_path)
        != decontamination["homology_exclusion_receipt_sha256"]
    ):
        raise ValueError("homology receipt changed after corpus preparation")

    exact_hex = evaluation_digests(
        [Path(decontamination["pcore_index"]), Path(decontamination["contact_manifest"])]
    )
    exact = {bytes.fromhex(value) for value in exact_hex}
    homology = _digest_lines(homology_path)
    excluded = exact | homology
    sources: dict[str, object] = {}
    total_records = 0
    total_residues = 0
    for source in SOURCES:
        source_manifest = manifest["sources"][source]
        source_homology = homology_receipt.get("sources", {}).get(source, {})
        screening_coverage = source_homology.get("screening_coverage")
        if isinstance(screening_coverage, dict):
            screened_prefix = int(screening_coverage["original_source_records_scanned"])
            if int(source_manifest["scanned_records"]) > screened_prefix:
                raise ValueError(
                    f"prepared {source} corpus scanned beyond its homology-screened prefix"
                )
        else:
            screened_prefix = None
        train_report, train_index = _split_report(
            data_root / source / "train", source_manifest["train"]
        )
        validation_report, validation_index = _split_report(
            data_root / source / "validation", source_manifest["validation"]
        )
        validation_digests = {bytes(value) for value in validation_index["digest"]}
        train_excluded = _count_members(train_index, excluded)
        validation_excluded = _count_members(validation_index, excluded)
        overlap = _count_members(train_index, validation_digests)
        if train_excluded or validation_excluded or overlap:
            raise ValueError(
                f"corpus verification failed for {source}: train_excluded={train_excluded}, "
                f"validation_excluded={validation_excluded}, overlap={overlap}"
            )
        sources[source] = {
            "train": train_report,
            "validation": validation_report,
            "train_excluded_intersection": train_excluded,
            "validation_excluded_intersection": validation_excluded,
            "train_validation_intersection": overlap,
            "prepared_source_records_scanned": int(source_manifest["scanned_records"]),
            "homology_screened_source_prefix_records": screened_prefix,
        }
        total_records += train_report["records"] + validation_report["records"]
        total_residues += train_report["residues"] + validation_report["residues"]

    return {
        "schema_version": 1,
        "status": "verified",
        "protocol": "prepared-corpus-decontamination-verification-v1",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "homology_exclusion_receipt": str(homology_receipt_path.resolve()),
        "homology_exclusion_receipt_sha256": file_sha256(homology_receipt_path),
        "exact_exclusion_digests": len(exact),
        "homology_exclusion_digests": len(homology),
        "combined_exclusion_digests": len(excluded),
        "total_records": total_records,
        "total_residues": total_residues,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.data_root / "CORPUS_VERIFICATION.json"
    report = verify(args.data_root)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
