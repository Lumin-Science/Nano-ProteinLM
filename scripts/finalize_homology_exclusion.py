#!/usr/bin/env python3
"""Validate MMseqs2 hits and freeze training-representative exclusion digests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

PROTOCOL = "mmseqs2-evaluation-homology-exclusion-v1"
PROTECTED_ROLES = {"validation", "test", "validation_and_test"}


def file_sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_digests(path: Path, digests: set[str]) -> str:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("".join(f"{digest}\n" for digest in sorted(digests)))
    temporary.replace(path)
    return file_sha256(path)


def normalized_digest(value: str) -> str:
    digest = value.removeprefix("sha256_")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"invalid sequence identifier {value!r}")
    return digest


def coverage(value: str) -> float:
    observed = float(value)
    return observed / 100.0 if observed > 1.0 else observed


def finalize(
    *,
    ledger_path: Path,
    membership_path: Path,
    hits: dict[str, Path],
    source_receipts: dict[str, Path],
    command_receipt: Path,
    output_root: Path,
    minimum_identity: float,
    minimum_query_coverage: float,
    minimum_target_coverage: float,
    maximum_sequences: int,
) -> dict[str, object]:
    ledger = json.loads(ledger_path.read_text())
    expected_membership = ledger["artifacts"][membership_path.name]["sha256"]
    if file_sha256(membership_path) != expected_membership:
        raise ValueError("evaluation membership digest differs from the split ledger")
    command = json.loads(command_receipt.read_text())
    if command.get("status") != "complete":
        raise ValueError("MMseqs2 command receipt is not complete")

    protected_queries: set[str] = set()
    all_queries: set[str] = set()
    with membership_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            digest = normalized_digest(row["sha256"])
            all_queries.add(digest)
            if any(item["role"] in PROTECTED_ROLES for item in row["memberships"]):
                protected_queries.add(digest)

    all_excluded: set[str] = set()
    protected_excluded: set[str] = set()
    source_reports: dict[str, dict[str, object]] = {}
    for source, path in sorted(hits.items()):
        hit_rows = 0
        query_counts: Counter[str] = Counter()
        targets: set[str] = set()
        protected_targets: set[str] = set()
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 8:
                    raise ValueError(f"unexpected MMseqs2 row at {path}:{line_number}")
                query = normalized_digest(fields[0])
                target = normalized_digest(fields[1])
                identity = float(fields[2]) / 100.0
                query_coverage = coverage(fields[4])
                target_coverage = coverage(fields[5])
                if query not in all_queries:
                    raise ValueError(f"MMseqs2 result contains an unknown query {query}")
                if (
                    identity + 1e-12 < minimum_identity
                    or query_coverage + 1e-12 < minimum_query_coverage
                    or target_coverage + 1e-12 < minimum_target_coverage
                ):
                    raise ValueError(f"MMseqs2 result violates the frozen thresholds at {path}")
                query_counts[query] += 1
                targets.add(target)
                if query in protected_queries:
                    protected_targets.add(target)
                hit_rows += 1
        largest = max(query_counts.values(), default=0)
        if largest >= maximum_sequences:
            raise ValueError(
                f"{source} has a query at the --max-seqs cap; exclusion is incomplete"
            )
        verification = json.loads(source_receipts[source].read_text())
        source_report: dict[str, object] = {
            "hits": str(path.resolve()),
            "hits_sha256": file_sha256(path),
            "alignment_rows": hit_rows,
            "matched_queries": len(query_counts),
            "excluded_all_splits": len(targets),
            "excluded_validation_test": len(protected_targets),
            "maximum_hits_for_one_query": largest,
            "representative_fasta_sha256": verification["representative_fasta_sha256"],
            "representative_sequences": verification["clusters"],
            "source_verification_sha256": file_sha256(source_receipts[source]),
        }
        if isinstance(verification.get("screening_coverage"), dict):
            source_report["screening_coverage"] = verification["screening_coverage"]
        source_reports[source] = source_report
        all_excluded.update(targets)
        protected_excluded.update(protected_targets)

    output_root.mkdir(parents=True, exist_ok=True)
    all_path = output_root / "homology_excluded_all_splits.txt"
    protected_path = output_root / "homology_excluded_validation_test.txt"
    all_sha256 = write_digests(all_path, all_excluded)
    protected_sha256 = write_digests(protected_path, protected_excluded)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "verified",
        "protocol": PROTOCOL,
        "scope_used_for_training": "all evaluation splits",
        "training_reservoir_scope": command.get(
            "training_reservoir_scope", "complete transferred representative FASTAs"
        ),
        "validation_test_scope_is_subset": protected_excluded <= all_excluded,
        "mmseqs_version": command["mmseqs_version"],
        "thresholds": {
            "minimum_sequence_identity": minimum_identity,
            "minimum_query_coverage": minimum_query_coverage,
            "minimum_target_coverage": minimum_target_coverage,
            "coverage_mode": 0,
            "maximum_sequences_per_query": maximum_sequences,
        },
        "evaluation_split_ledger": str(ledger_path.resolve()),
        "evaluation_split_ledger_sha256": file_sha256(ledger_path),
        "evaluation_memberships_sha256": expected_membership,
        "evaluation_all_split_queries": len(all_queries),
        "evaluation_validation_test_queries": len(protected_queries),
        "excluded_training_representatives": len(all_excluded),
        "excluded_validation_test_training_representatives": len(protected_excluded),
        "excluded_digest_file": str(all_path.resolve()),
        "excluded_digest_file_sha256": all_sha256,
        "protected_excluded_digest_file": str(protected_path.resolve()),
        "protected_excluded_digest_file_sha256": protected_sha256,
        "command_receipt": str(command_receipt.resolve()),
        "command_receipt_sha256": file_sha256(command_receipt),
        "sources": source_reports,
    }
    write_json(output_root / "HOMOLOGY_EXCLUSION_VERIFIED.json", report)
    return report


def keyed_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or not path or key in result:
            raise ValueError(f"expected unique NAME=PATH value, found {value!r}")
        result[key] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--memberships", type=Path, required=True)
    parser.add_argument("--hits", action="append", default=[], required=True)
    parser.add_argument("--source-receipt", action="append", default=[], required=True)
    parser.add_argument("--command-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-identity", type=float, default=0.30)
    parser.add_argument("--minimum-query-coverage", type=float, default=0.80)
    parser.add_argument("--minimum-target-coverage", type=float, default=0.80)
    parser.add_argument("--maximum-sequences", type=int, default=1_000_000)
    args = parser.parse_args()
    report = finalize(
        ledger_path=args.ledger,
        membership_path=args.memberships,
        hits=keyed_paths(args.hits),
        source_receipts=keyed_paths(args.source_receipt),
        command_receipt=args.command_receipt,
        output_root=args.output_root,
        minimum_identity=args.minimum_identity,
        minimum_query_coverage=args.minimum_query_coverage,
        minimum_target_coverage=args.minimum_target_coverage,
        maximum_sequences=args.maximum_sequences,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
