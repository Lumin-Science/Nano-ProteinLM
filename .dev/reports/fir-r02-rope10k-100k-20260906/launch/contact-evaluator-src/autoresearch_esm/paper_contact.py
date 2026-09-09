"""Digest-bound scoring for the 2026 ESMC paper contact evaluation.

This module deliberately separates the paper-style long-range contact score
from the older ``ESMStructuralSplitDataset`` contact surrogate used by
UPLM-v1.  The numerical headline is the unweighted mean, across eligible PDB
chains, of top-L precision for C-beta contacts below 8 Angstrom with sequence
separation at least 24 residues.

Model inference and probe fitting are intentionally outside this module.  A
score packet is accepted only after those steps have produced per-chain values
bound to both an immutable evaluation manifest and a model-specific probe
receipt.  This makes aggregation cheap to reproduce and prevents an
approximate data build from being labelled as paper parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_ID = "esmc-paper-contact-v1"
SCHEMA_VERSION = 1
PAPER_DOI = "10.64898/2026.06.03.729735"
PDB_SNAPSHOT = "2024-02-28"
MAXIMUM_RESIDUES = 510
SEQUENCE_SEPARATION = 24
DISTANCE_THRESHOLD_ANGSTROM = 8.0
TRAIN_STRUCTURES = 20
EXPECTED_EVALUATION_CHAINS = 20_775
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20_260_819


@dataclass(frozen=True)
class ChainContactScore:
    """One paper-style evaluation unit after deterministic filtering."""

    chain_id: str
    source_length: int
    evaluated_length: int
    eligible_pair_count: int
    true_long_range_contacts: int
    precision_at_l: float
    random_precision_at_l: float


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _sha256_text(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_chain(
    chain_id: str,
    contact_scores: np.ndarray,
    cb_distances: np.ndarray,
    *,
    source_length: int,
    maximum_residues: int = MAXIMUM_RESIDUES,
    sequence_separation: int = SEQUENCE_SEPARATION,
    distance_threshold_angstrom: float = DISTANCE_THRESHOLD_ANGSTROM,
) -> ChainContactScore | None:
    """Return long-range top-L precision, or ``None`` when the paper filters it.

    ``cb_distances`` must already implement the standard glycine fallback to
    C-alpha.  Non-finite distances are treated as unresolved residue pairs and
    excluded.  Ties in prediction score are resolved by the deterministic
    upper-triangle pair order.
    """

    chain_id = str(chain_id).strip()
    if not chain_id:
        raise ValueError("chain_id must be non-empty")
    source_length = int(source_length)
    maximum_residues = int(maximum_residues)
    sequence_separation = int(sequence_separation)
    distance_threshold_angstrom = _finite(
        "distance_threshold_angstrom", distance_threshold_angstrom
    )
    if source_length <= 0:
        raise ValueError("source_length must be positive")
    if maximum_residues <= 0:
        raise ValueError("maximum_residues must be positive")
    if sequence_separation <= 0:
        raise ValueError("sequence_separation must be positive")
    if distance_threshold_angstrom <= 0:
        raise ValueError("distance_threshold_angstrom must be positive")

    scores = np.asarray(contact_scores, dtype=np.float64)
    distances = np.asarray(cb_distances, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("contact_scores must be a square matrix")
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("cb_distances must be a square matrix")
    evaluated_length = min(
        source_length,
        maximum_residues,
        scores.shape[0],
        distances.shape[0],
    )
    if evaluated_length <= sequence_separation:
        return None

    i, j = np.triu_indices(evaluated_length, k=sequence_separation)
    pair_distances = distances[i, j]
    pair_scores = scores[i, j]
    valid = np.isfinite(pair_distances) & np.isfinite(pair_scores)
    pair_distances = pair_distances[valid]
    pair_scores = pair_scores[valid]
    if pair_scores.size < evaluated_length:
        return None

    labels = pair_distances < distance_threshold_angstrom
    true_contacts = int(labels.sum())
    # The paper excludes structures with fewer than L true contacts.
    if true_contacts < evaluated_length:
        return None

    order = np.argsort(-pair_scores, kind="stable")[:evaluated_length]
    return ChainContactScore(
        chain_id=chain_id,
        source_length=source_length,
        evaluated_length=evaluated_length,
        eligible_pair_count=int(labels.size),
        true_long_range_contacts=true_contacts,
        precision_at_l=float(labels[order].mean()),
        random_precision_at_l=float(labels.mean()),
    )


def _bootstrap_means(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 32,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap values must be finite")
    replicates = int(replicates)
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if chunk_size <= 0:
        raise ValueError("bootstrap chunk_size must be positive")

    rng = np.random.default_rng(int(seed))
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        samples[start:stop] = values[indices].mean(axis=1)
    return samples


def _validated_chain_rows(
    packet: dict[str, Any],
    *,
    expected_evaluation_chains: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"score packet schema_version must be {SCHEMA_VERSION}")
    if packet.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"score packet protocol_id must be {PROTOCOL_ID}")
    for field in ("dataset_manifest_sha256", "probe_receipt_sha256", "weights_sha256"):
        _sha256_text(packet.get(field), field=field)
    if not str(packet.get("model_id", "")).strip():
        raise ValueError("model_id must be non-empty")
    if not str(packet.get("model_revision", "")).strip():
        raise ValueError("model_revision must be non-empty")

    rows = packet.get("chains")
    if not isinstance(rows, list):
        raise TypeError("chains must be a list")
    if len(rows) != int(expected_evaluation_chains):
        raise ValueError(
            "paper-parity chain count mismatch: "
            f"expected={expected_evaluation_chains}, observed={len(rows)}"
        )

    seen: set[str] = set()
    precisions = np.empty(len(rows), dtype=np.float64)
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"chain row {index} must be an object")
        chain_id = str(row.get("chain_id", "")).strip()
        if not chain_id:
            raise ValueError(f"chain row {index} has an empty chain_id")
        if chain_id in seen:
            raise ValueError(f"duplicate chain_id: {chain_id}")
        seen.add(chain_id)
        source_length = int(row.get("source_length", 0))
        evaluated_length = int(row.get("evaluated_length", 0))
        eligible_pairs = int(row.get("eligible_pair_count", 0))
        true_contacts = int(row.get("true_long_range_contacts", 0))
        precision = _finite("precision_at_l", row.get("precision_at_l"))
        random_precision = _finite(
            "random_precision_at_l", row.get("random_precision_at_l")
        )
        if source_length <= 0:
            raise ValueError(f"source_length must be positive for {chain_id}")
        if evaluated_length != min(source_length, MAXIMUM_RESIDUES):
            raise ValueError(f"evaluated_length violates the 510-residue crop for {chain_id}")
        if eligible_pairs < evaluated_length:
            raise ValueError(f"too few eligible pairs for {chain_id}")
        if true_contacts < evaluated_length:
            raise ValueError(f"paper top-L filter was not applied for {chain_id}")
        if not 0.0 <= precision <= 1.0:
            raise ValueError(f"precision_at_l outside [0, 1] for {chain_id}")
        if not 0.0 <= random_precision <= 1.0:
            raise ValueError(f"random_precision_at_l outside [0, 1] for {chain_id}")
        precisions[index] = precision
        normalized.append(
            asdict(
                ChainContactScore(
                    chain_id=chain_id,
                    source_length=source_length,
                    evaluated_length=evaluated_length,
                    eligible_pair_count=eligible_pairs,
                    true_long_range_contacts=true_contacts,
                    precision_at_l=precision,
                    random_precision_at_l=random_precision,
                )
            )
        )
    return normalized, precisions


def score_packet(
    packet: dict[str, Any],
    *,
    expected_evaluation_chains: int = EXPECTED_EVALUATION_CHAINS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate a per-chain packet and produce the primary score report."""

    rows, precisions = _validated_chain_rows(
        packet,
        expected_evaluation_chains=expected_evaluation_chains,
    )
    samples = _bootstrap_means(
        precisions,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    interval = np.quantile(samples, [0.025, 0.975])
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "paper": {
            "doi": PAPER_DOI,
            "pdb_snapshot": PDB_SNAPSHOT,
        },
        "model": {
            "model_id": packet["model_id"],
            "model_revision": packet["model_revision"],
            "weights_sha256": packet["weights_sha256"],
        },
        "provenance": {
            "dataset_manifest_sha256": packet["dataset_manifest_sha256"],
            "probe_receipt_sha256": packet["probe_receipt_sha256"],
        },
        "metric": {
            "name": "long_range_top_l_contact_precision",
            "point_estimate": float(precisions.mean()),
            "evaluation_chains": len(rows),
            "maximum_residues": MAXIMUM_RESIDUES,
            "sequence_separation": SEQUENCE_SEPARATION,
            "distance_threshold_angstrom": DISTANCE_THRESHOLD_ANGSTROM,
            "contact_atom": "C-beta; C-alpha fallback for glycine",
        },
        "bootstrap": {
            "unit": "PDB_chain",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "confidence_interval_95": [float(interval[0]), float(interval[1])],
            "samples": samples.tolist(),
        },
        "chains": rows,
    }


def compare_reports(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compute a paired chain bootstrap for two paper-contact reports."""

    for report, label in ((reference, "reference"), (candidate, "candidate")):
        if report.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{label} report schema changed")
        if report.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{label} report protocol changed")
    if candidate["provenance"]["dataset_manifest_sha256"] != reference["provenance"][
        "dataset_manifest_sha256"
    ]:
        raise ValueError("paired reports use different evaluation manifests")
    if candidate["bootstrap"]["replicates"] != reference["bootstrap"]["replicates"]:
        raise ValueError("paired reports use different bootstrap replicate counts")
    if candidate["bootstrap"]["seed"] != reference["bootstrap"]["seed"]:
        raise ValueError("paired reports use different bootstrap seeds")

    reference_rows = reference.get("chains", [])
    candidate_rows = candidate.get("chains", [])
    reference_contract = [
        (
            row["chain_id"],
            row["source_length"],
            row["evaluated_length"],
            row["eligible_pair_count"],
            row["true_long_range_contacts"],
        )
        for row in reference_rows
    ]
    candidate_contract = [
        (
            row["chain_id"],
            row["source_length"],
            row["evaluated_length"],
            row["eligible_pair_count"],
            row["true_long_range_contacts"],
        )
        for row in candidate_rows
    ]
    if candidate_contract != reference_contract:
        raise ValueError("paired reports do not contain identical ordered chain units")

    reference_values = np.asarray(
        [row["precision_at_l"] for row in reference_rows], dtype=np.float64
    )
    candidate_values = np.asarray(
        [row["precision_at_l"] for row in candidate_rows], dtype=np.float64
    )
    differences = candidate_values - reference_values
    samples = _bootstrap_means(
        differences,
        replicates=int(reference["bootstrap"]["replicates"]),
        seed=int(reference["bootstrap"]["seed"]),
    )
    interval = np.quantile(samples, [0.025, 0.975])
    point_delta = float(candidate_values.mean() - reference_values.mean())
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": f"{PROTOCOL_ID}-paired-comparison",
        "dataset_manifest_sha256": reference["provenance"]["dataset_manifest_sha256"],
        "reference_model": reference["model"],
        "candidate_model": candidate["model"],
        "point_estimate": {
            "reference": float(reference_values.mean()),
            "candidate": float(candidate_values.mean()),
            "candidate_minus_reference": point_delta,
        },
        "paired_bootstrap": {
            "unit": "PDB_chain",
            "replicates": int(reference["bootstrap"]["replicates"]),
            "seed": int(reference["bootstrap"]["seed"]),
            "mean_delta": float(samples.mean()),
            "median_delta": float(np.median(samples)),
            "confidence_interval_95": [float(interval[0]), float(interval[1])],
            "probability_candidate_superior": float(
                np.mean(samples > 0) + 0.5 * np.mean(samples == 0)
            ),
            "interval_excludes_zero": bool(interval[0] > 0 or interval[1] < 0),
            "samples": samples.tolist(),
        },
    }


def _atomic_write_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return _file_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m autoresearch_esm.paper_contact")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--packet", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--reference", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "score":
        result = score_packet(json.loads(args.packet.read_text()))
    else:
        result = compare_reports(
            json.loads(args.reference.read_text()),
            json.loads(args.candidate.read_text()),
        )
    output_sha256 = _atomic_write_json(args.output, result)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": output_sha256}))


if __name__ == "__main__":
    main()
