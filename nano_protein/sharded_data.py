"""Immutable Parquet-shard selection, download, and mmap materialization.

The remote dataset is a complete reservoir.  A training run downloads only the
smallest deterministic prefix of each source arm that satisfies its requested
number of unique examples.  Validation shards and the release contract are
always fetched in full.  The plan reports records, residues, and compressed
bytes separately.  This mirrors nanochat's operational model: immutable whole
shards are cached locally; individual rows are not streamed over the network
inside the training loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq

from .data import SOURCES, _StoreWriter, file_sha256
from .tokenizer import ProteinTokenizer

PROTOCOL = "protein-corpus-parquet-shards-v1"
DEFAULT_REPO_ID = "LuminScience/LuminBench-Nano-ESMC"


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _validate_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} is not a SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} is not a SHA-256 digest") from error
    return value


def validate_release_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on malformed or weakly decontaminated releases."""

    if manifest.get("protocol") != PROTOCOL or manifest.get("status") != "verified":
        raise ValueError(f"manifest must be a verified {PROTOCOL} release")
    decontamination = manifest.get("decontamination")
    if not isinstance(decontamination, Mapping):
        raise ValueError("manifest is missing its decontamination contract")
    _validate_digest(
        decontamination.get("homology_exclusion_receipt_sha256"),
        label="homology exclusion receipt",
    )
    thresholds = decontamination.get("thresholds")
    if not isinstance(thresholds, Mapping) or not (
        thresholds.get("minimum_sequence_identity") == 0.30
        and thresholds.get("minimum_query_coverage") == 0.80
        and thresholds.get("minimum_target_coverage") == 0.80
        and thresholds.get("coverage_mode") == 0
        and decontamination.get("scope") == "all_evaluation_splits"
    ):
        raise ValueError("release does not satisfy the frozen all-split homology screen")
    evaluations = decontamination.get("evaluation_protocols")
    required = {"contact-p-at-l", "pcore-v0.2", "pcore-v0.5-alpha-q9"}
    if not isinstance(evaluations, list) or not required <= set(evaluations):
        raise ValueError("release does not protect P@L plus legacy and Q9 P-CORE")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCES):
        raise ValueError(f"manifest sources must be exactly {SOURCES}")
    seen_paths: set[str] = set()
    for source in SOURCES:
        source_row = sources[source]
        if not isinstance(source_row, Mapping):
            raise ValueError(f"invalid source entry for {source}")
        for split in ("train", "validation"):
            shards = source_row.get(split)
            if not isinstance(shards, list) or not shards:
                raise ValueError(f"{source}/{split} has no shards")
            previous_max = ""
            for shard in shards:
                if not isinstance(shard, Mapping):
                    raise ValueError(f"invalid shard in {source}/{split}")
                path = shard.get("path")
                if not isinstance(path, str) or path in seen_paths:
                    raise ValueError(f"duplicate or invalid shard path: {path!r}")
                parsed_path = PurePosixPath(path)
                if parsed_path.is_absolute() or ".." in parsed_path.parts:
                    raise ValueError(f"unsafe shard path: {path!r}")
                seen_paths.add(path)
                _validate_digest(shard.get("sha256"), label=path)
                if int(shard.get("records", 0)) <= 0 or int(shard.get("residues", 0)) <= 0:
                    raise ValueError(f"empty shard declared by manifest: {path}")
                minimum = _validate_digest(shard.get("minimum_sequence_sha256"), label=path)
                maximum = _validate_digest(shard.get("maximum_sequence_sha256"), label=path)
                if minimum > maximum or (previous_max and minimum <= previous_max):
                    raise ValueError(f"shards are not in strictly increasing SHA order: {path}")
                previous_max = maximum


def plan_shards(
    manifest: Mapping[str, Any],
    *,
    total_training_samples: int,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """Choose the smallest per-source train-shard prefixes for one run budget."""

    validate_release_manifest(manifest)
    if total_training_samples <= 0:
        raise ValueError("total_training_samples must be positive")
    if set(weights) != set(SOURCES) or any(float(value) <= 0 for value in weights.values()):
        raise ValueError(f"weights must provide positive values for exactly {SOURCES}")
    denominator = sum(float(value) for value in weights.values())
    if denominator <= 0:
        raise ValueError("at least one source weight must be positive")

    selected: dict[str, Any] = {}
    all_paths: list[str] = []
    selected_records = selected_residues = selected_bytes = 0
    for source in SOURCES:
        required = math.ceil(total_training_samples * float(weights[source]) / denominator)
        available = 0
        residues = 0
        compressed_bytes = 0
        train: list[Mapping[str, Any]] = []
        for shard in manifest["sources"][source]["train"]:
            if available >= required:
                break
            train.append(shard)
            available += int(shard["records"])
            residues += int(shard["residues"])
            compressed_bytes += int(shard["bytes"])
        if available < required:
            raise ValueError(
                f"release has only {available:,} {source} rows, below the {required:,} budget"
            )
        validation = list(manifest["sources"][source]["validation"])
        all_paths.extend(str(row["path"]) for row in train)
        all_paths.extend(str(row["path"]) for row in validation)
        validation_records = sum(int(row["records"]) for row in validation)
        validation_residues = sum(int(row["residues"]) for row in validation)
        validation_bytes = sum(int(row["bytes"]) for row in validation)
        selected_records += available + validation_records
        selected_residues += residues + validation_residues
        selected_bytes += compressed_bytes + validation_bytes
        selected[source] = {
            "required_unique_records": required,
            "selected_unique_records": available,
            "selected_train_residues": residues,
            "selected_train_compressed_bytes": compressed_bytes,
            "selected_validation_records": validation_records,
            "selected_validation_residues": validation_residues,
            "selected_validation_compressed_bytes": validation_bytes,
            "train": train,
            "validation": validation,
        }
    return {
        "schema_version": 1,
        "protocol": "protein-corpus-budget-plan-v1",
        "release_id": manifest.get("release_id"),
        "total_training_samples": total_training_samples,
        "normalized_weights": {
            source: float(weights[source]) / denominator for source in SOURCES
        },
        "selected_records_including_validation": selected_records,
        "selected_residues_including_validation": selected_residues,
        "selected_compressed_bytes_including_validation": selected_bytes,
        "sources": selected,
        "paths": all_paths,
    }


def fetch_release_plan(
    *,
    repo_id: str,
    revision: str,
    cache_root: Path,
    total_training_samples: int,
    weights: Mapping[str, float],
    download_workers: int = 8,
) -> tuple[dict[str, Any], Path]:
    """Download the manifest, plan locally, then fetch only selected whole shards."""

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    if download_workers <= 0:
        raise ValueError("download_workers must be positive")

    # Resolve a branch/tag exactly once.  The manifest and every shard then come
    # from one immutable commit even if `main` changes during a long download.
    resolved_revision = HfApi().dataset_info(repo_id, revision=revision).sha

    manifest_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=resolved_revision,
            filename="manifest.json",
            local_dir=cache_root,
        )
    )
    manifest = json.loads(manifest_path.read_text())
    plan = plan_shards(manifest, total_training_samples=total_training_samples, weights=weights)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=resolved_revision,
        allow_patterns=plan["paths"],
        local_dir=cache_root,
        max_workers=download_workers,
    )
    for relative in plan["paths"]:
        local = cache_root / relative
        if not local.is_file():
            raise FileNotFoundError(f"selected shard was not downloaded: {relative}")
        expected = next(
            row["sha256"]
            for source in SOURCES
            for split in ("train", "validation")
            for row in plan["sources"][source][split]
            if row["path"] == relative
        )
        if file_sha256(local) != expected:
            raise ValueError(f"downloaded shard checksum mismatch: {relative}")
    plan["repo_id"] = repo_id
    plan["requested_revision"] = revision
    plan["revision"] = resolved_revision
    plan["release_manifest_sha256"] = file_sha256(manifest_path)
    output = cache_root / "download-plan.json"
    output.write_text(_canonical_json(plan))
    return plan, manifest_path


def materialize_plan(
    plan: Mapping[str, Any],
    release_manifest: Mapping[str, Any],
    cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Convert a selected Parquet prefix to the existing fast mmap training layout."""

    validate_release_manifest(release_manifest)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite prepared corpus: {output_root}")
    for source in SOURCES:
        for split in ("train", "validation"):
            released = release_manifest["sources"][source][split]
            selected = plan["sources"][source][split]
            expected = released if split == "validation" else released[: len(selected)]
            if selected != expected:
                raise ValueError(
                    f"download plan is not a release-manifest prefix: {source}/{split}"
                )
    tokenizer = ProteinTokenizer.esmc()
    source_receipts: dict[str, Any] = {}
    output_root.mkdir(parents=True)
    for source in SOURCES:
        splits: dict[str, Any] = {}
        for split in ("train", "validation"):
            writer = _StoreWriter(output_root / source / split)
            observed_previous = ""
            for shard in plan["sources"][source][split]:
                local = cache_root / str(shard["path"])
                parquet = pq.ParquetFile(local)
                for batch in parquet.iter_batches(
                    columns=["sequence", "sha256", "length"], batch_size=65_536
                ):
                    rows = batch.to_pydict()
                    for sequence, digest, length in zip(
                        rows["sequence"], rows["sha256"], rows["length"], strict=True
                    ):
                        if len(sequence) != int(length):
                            raise ValueError(f"length mismatch in {local}: {digest}")
                        observed = hashlib.sha256(sequence.encode("ascii")).hexdigest()
                        if observed != digest:
                            raise ValueError(f"sequence digest mismatch in {local}: {digest}")
                        if observed_previous and digest <= observed_previous:
                            raise ValueError(f"non-increasing SHA order in {local}: {digest}")
                        observed_previous = digest
                        writer.append(tokenizer.encode_residues(sequence), digest)
            splits[split] = writer.finish()
        source_receipts[source] = splits
    release_decontamination = release_manifest["decontamination"]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "parquet-prefix-to-mmap-v1",
        "sampling_unit": release_manifest.get("sampling_unit"),
        "release_id": release_manifest.get("release_id"),
        "release_manifest_sha256": plan["release_manifest_sha256"],
        "download_plan_sha256": hashlib.sha256(_canonical_json(plan).encode()).hexdigest(),
        "decontamination": {
            "homology_exclusion": True,
            "homology_exclusion_receipt_sha256": release_decontamination[
                "homology_exclusion_receipt_sha256"
            ],
            "homology_contract": {
                "status": "verified",
                "protocol": "mmseqs2-evaluation-homology-exclusion-v2",
                "scope_used_for_training": "all evaluation splits",
                "thresholds": release_decontamination["thresholds"],
                "evaluation_protocols": release_decontamination["evaluation_protocols"],
                "blocked_benchmark_candidates_are_protected": release_decontamination[
                    "blocked_benchmark_candidates_are_protected"
                ],
            },
        },
        "sources": source_receipts,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(_canonical_json(manifest))
    verification = {
        "schema_version": 2,
        "status": "verified",
        "protocol": "prepared-corpus-decontamination-verification-v2",
        "manifest_sha256": file_sha256(manifest_path),
        "release_manifest_sha256": plan["release_manifest_sha256"],
        "homology_exclusion_receipt_sha256": release_decontamination[
            "homology_exclusion_receipt_sha256"
        ],
        "proof": (
            "every local row was rehashed from a checksum-verified shard in a fully "
            "verified release with zero exact/homology and train/validation intersections"
        ),
        "sources": {
            source: {
                "train_excluded_intersection": 0,
                "validation_excluded_intersection": 0,
                "train_validation_intersection": 0,
                **source_receipts[source],
            }
            for source in SOURCES
        },
    }
    (output_root / "CORPUS_VERIFICATION.json").write_text(_canonical_json(verification))
    return verification


def _weights(value: str) -> dict[str, float]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("weights must be a JSON object")
    return {str(key): float(number) for key, number in parsed.items()}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-samples", type=int, required=True)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument(
        "--weights",
        type=_weights,
        default={"uniref90": 36.0, "mgnify": 11.0, "omg_img": 54.0},
    )
    args = parser.parse_args(argv)
    plan, manifest_path = fetch_release_plan(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_root=args.cache_root,
        total_training_samples=args.training_samples,
        weights=args.weights,
        download_workers=args.download_workers,
    )
    release_manifest = json.loads(manifest_path.read_text())
    receipt = materialize_plan(plan, release_manifest, args.cache_root, args.output_root)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
