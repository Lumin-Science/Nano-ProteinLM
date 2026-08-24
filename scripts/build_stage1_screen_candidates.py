#!/usr/bin/env python3
"""Freeze an eligible source prefix large enough for the 4-hour Stage-1 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from nano_protein.data import (
    SOURCES,
    _header_digest,
    evaluation_digests,
    fasta_records,
    file_sha256,
)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_candidates(
    *,
    cluster_root: Path,
    output_root: Path,
    pcore_index: Path,
    contact_manifest: Path,
    train_candidates: int,
    validation_candidates: int,
    validation_modulus: int = 32,
    validation_bucket: int = 0,
    minimum_length: int = 32,
    maximum_length: int = 16_384,
) -> dict[str, object]:
    if train_candidates <= 0 or validation_candidates <= 0:
        raise ValueError("candidate targets must be positive")
    exact_excluded = evaluation_digests([pcore_index, contact_manifest])
    output_root.mkdir(parents=True, exist_ok=True)
    sources: dict[str, object] = {}
    for source in SOURCES:
        source_root = cluster_root / source
        source_fasta = source_root / "representatives.fasta"
        upstream_verification_path = source_root / "verification.json"
        upstream_verification = json.loads(upstream_verification_path.read_text())
        candidate_path = output_root / f"{source}.fasta"
        temporary = candidate_path.with_suffix(candidate_path.suffix + ".partial")
        counts: Counter[str] = Counter()
        rejected: Counter[str] = Counter()
        scanned = 0
        last_digest: str | None = None
        with temporary.open("w", encoding="ascii") as handle:
            for header, sequence in fasta_records(source_fasta):
                scanned += 1
                digest = _header_digest(header)
                last_digest = digest
                if hashlib.sha256(sequence.encode("ascii")).hexdigest() != digest:
                    raise ValueError(
                        f"sequence/header SHA mismatch in {source_fasta}: {header}"
                    )
                if digest in exact_excluded:
                    rejected["evaluation_exact_match"] += 1
                    continue
                if not minimum_length <= len(sequence) <= maximum_length:
                    rejected["length"] += 1
                    continue
                split = (
                    "validation"
                    if int(digest[:16], 16) % validation_modulus == validation_bucket
                    else "train"
                )
                target = validation_candidates if split == "validation" else train_candidates
                if counts[split] >= target:
                    continue
                handle.write(f">sha256_{digest}\n")
                for start in range(0, len(sequence), 80):
                    handle.write(sequence[start : start + 80] + "\n")
                counts[split] += 1
                if (
                    counts["train"] >= train_candidates
                    and counts["validation"] >= validation_candidates
                ):
                    break
        if counts["train"] != train_candidates or counts["validation"] != validation_candidates:
            raise RuntimeError(f"{source} exhausted before candidate targets: {dict(counts)}")
        temporary.replace(candidate_path)
        verification = {
            "schema_version": 1,
            "status": "complete",
            "protocol": "stage1-screen-candidate-prefix-v1",
            "source": source,
            "clusters": counts["train"] + counts["validation"],
            "representative_fasta": str(candidate_path.resolve()),
            "representative_fasta_sha256": file_sha256(candidate_path),
            "upstream_verification": upstream_verification,
            "upstream_verification_sha256": file_sha256(upstream_verification_path),
            "screening_coverage": {
                "coverage_kind": "complete_eligible_prefix",
                "original_source_fasta": str(source_fasta.resolve()),
                "original_source_records_scanned": scanned,
                "last_original_digest": last_digest,
                "candidate_train_sequences": counts["train"],
                "candidate_validation_sequences": counts["validation"],
                "candidate_total_sequences": counts["train"] + counts["validation"],
                "validation_modulus": validation_modulus,
                "validation_bucket": validation_bucket,
                "minimum_length": minimum_length,
                "maximum_length": maximum_length,
                "rejected": dict(rejected),
            },
        }
        verification_path = output_root / f"{source}.verification.json"
        _write_json(verification_path, verification)
        sources[source] = {
            **verification,
            "verification": str(verification_path.resolve()),
            "verification_sha256": file_sha256(verification_path),
        }

    report = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "stage1-screen-candidate-bundle-v1",
        "selection": (
            "complete SHA-ordered eligible prefix with independent train/validation targets"
        ),
        "pcore_index_sha256": file_sha256(pcore_index),
        "contact_manifest_sha256": file_sha256(contact_manifest),
        "exact_exclusion_digests": len(exact_excluded),
        "sources": sources,
    }
    _write_json(output_root / "CANDIDATE_BUNDLE.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pcore-index", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--train-candidates", type=int, default=5_000_000)
    parser.add_argument("--validation-candidates", type=int, default=32_768)
    args = parser.parse_args()
    report = build_candidates(
        cluster_root=args.cluster_root,
        output_root=args.output_root,
        pcore_index=args.pcore_index,
        contact_manifest=args.contact_manifest,
        train_candidates=args.train_candidates,
        validation_candidates=args.validation_candidates,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
