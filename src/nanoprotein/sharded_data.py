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
SCREEN_SEARCH_ORIENTATION = "training-representative-query-vs-evaluation-target"
LEGACY_SEARCH_ORIENTATION = "evaluation-query-vs-training-representative-target"
NORMALIZED_HIT_TABLE_SCHEMA = (
    "evaluation_sha256,training_sha256,pident,alnlen,"
    "evaluation_coverage,training_coverage,evalue,bits"
)
MINIMUM_MMSEQS_SENSITIVITY = 7.5
MINIMUM_ORIENTATION_AUDIT_SAMPLE = 8192


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


def validate_search_contracts(value: object) -> None:
    """Require either one fresh all-split screen or the audited parent+Q9 route."""

    if not isinstance(value, Mapping):
        raise ValueError("release is missing MMseqs search provenance")

    def numeric(row: Mapping[str, Any], key: str) -> float | None:
        observed = row.get(key)
        if isinstance(observed, bool) or not isinstance(observed, int | float):
            return None
        return float(observed)

    def reverse_contract(row: object, *, scope: str) -> bool:
        if not isinstance(row, Mapping):
            return False
        sensitivity = numeric(row, "sensitivity")
        targets = numeric(row, "evaluation_target_sequences")
        audit = row.get("orientation_audit")
        if not isinstance(audit, Mapping):
            return False
        audit_sample = numeric(audit, "minimum_sampled_training_sequences_per_source")
        audit_sources = audit.get("sources")
        audit_is_safe = bool(
            audit.get("protocol") == "mmseqs2-search-orientation-audit-v1"
            and audit.get("all_sources_reverse_recover_every_forward_pair") is True
            and audit_sample is not None
            and audit_sample >= MINIMUM_ORIENTATION_AUDIT_SAMPLE
            and audit_sample.is_integer()
            and isinstance(audit_sources, Mapping)
            and set(audit_sources) == set(SOURCES)
            and all(
                isinstance(audit_sources[source], Mapping)
                and audit_sources[source].get("reverse_recovers_every_forward_pair") is True
                and audit_sources[source].get("forward_only_pairs") == 0
                and numeric(audit_sources[source], "sample_training_sequences") is not None
                and numeric(audit_sources[source], "sample_training_sequences")
                >= MINIMUM_ORIENTATION_AUDIT_SAMPLE
                for source in SOURCES
            )
        )
        return bool(
            row.get("query_scope") == scope
            and row.get("search_orientation") == SCREEN_SEARCH_ORIENTATION
            and row.get("normalized_hit_table_schema") == NORMALIZED_HIT_TABLE_SCHEMA
            and sensitivity is not None
            and sensitivity >= MINIMUM_MMSEQS_SENSITIVITY
            and row.get("maximum_evalue") == 0.001
            and row.get("configured_candidate_cap") == 1_000_000
            and targets is not None
            and 0 < targets < 1_000_000
            and targets.is_integer()
            and row.get("candidate_cap_unreachable") is True
            and audit_is_safe
        )

    def forward_contract(row: object, *, scope: str) -> bool:
        """Validate the safe evaluation-query orientation by its observed cap."""

        if not isinstance(row, Mapping):
            return False
        sensitivity = numeric(row, "sensitivity")
        maximum_emitted = numeric(row, "maximum_emitted_hits_for_one_evaluation_query")
        return bool(
            row.get("query_scope") == scope
            and row.get("search_orientation") == LEGACY_SEARCH_ORIENTATION
            and row.get("normalized_hit_table_schema") == NORMALIZED_HIT_TABLE_SCHEMA
            and sensitivity is not None
            and sensitivity >= MINIMUM_MMSEQS_SENSITIVITY
            and row.get("maximum_evalue") == 0.001
            and row.get("configured_candidate_cap") == 1_000_000
            and maximum_emitted is not None
            and 0 <= maximum_emitted < 1_000_000
            and maximum_emitted.is_integer()
            and row.get("all_emitted_hit_counts_below_cap") is True
        )

    if set(value) == {"all_evaluation_splits"}:
        all_splits = value["all_evaluation_splits"]
        if reverse_contract(all_splits, scope="all-evaluation-splits") or forward_contract(
            all_splits, scope="all-evaluation-splits"
        ):
            return
    if set(value) != {"legacy_parent", "q9_delta"}:
        raise ValueError("release has an unknown MMseqs search-provenance topology")
    legacy = value["legacy_parent"]
    if not isinstance(legacy, Mapping):
        raise ValueError("legacy parent MMseqs search provenance is incomplete")
    legacy_sensitivity = numeric(legacy, "sensitivity")
    maximum_emitted = numeric(legacy, "maximum_emitted_hits_for_one_evaluation_query")
    if not (
        legacy.get("query_scope") == "p-at-l-and-pcore-v0.2-all-splits"
        and legacy.get("search_orientation") == LEGACY_SEARCH_ORIENTATION
        and legacy.get("normalized_hit_table_schema") == NORMALIZED_HIT_TABLE_SCHEMA
        and legacy_sensitivity is not None
        and legacy_sensitivity >= MINIMUM_MMSEQS_SENSITIVITY
        and legacy.get("maximum_evalue") == 0.001
        and legacy.get("configured_candidate_cap") == 1_000_000
        and maximum_emitted is not None
        and 0 <= maximum_emitted < 1_000_000
        and maximum_emitted.is_integer()
        and legacy.get("all_emitted_hit_counts_below_cap") is True
    ):
        raise ValueError("legacy parent MMseqs search provenance is incomplete")
    _validate_digest(
        legacy.get("command_receipt_sha256"), label="legacy MMseqs command receipt"
    )
    _validate_digest(legacy.get("commands_sha256"), label="legacy MMseqs commands ledger")
    q9_delta = value["q9_delta"]
    if not (
        reverse_contract(q9_delta, scope="q9-delta")
        or forward_contract(q9_delta, scope="q9-delta")
    ):
        raise ValueError("Q9 delta MMseqs search provenance is incomplete")


def validate_release_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on malformed or weakly decontaminated releases."""

    if manifest.get("protocol") != PROTOCOL or manifest.get("status") != "verified":
        raise ValueError(f"manifest must be a verified {PROTOCOL} release")
    ownership = manifest.get("global_exact_ownership")
    verification = manifest.get("verification")
    if not (
        isinstance(ownership, Mapping)
        and ownership.get("protocol") == "global-exact-representative-ownership-v1"
        and isinstance(verification, Mapping)
        and verification.get("global_train_exact_duplicate_intersection") == 0
    ):
        raise ValueError("release does not prove globally unique training sequences")
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
        and thresholds.get("maximum_evalue") == 0.001
        and thresholds.get("coverage_mode") == 0
        and decontamination.get("scope") == "all_evaluation_splits"
    ):
        raise ValueError("release does not satisfy the frozen all-split homology screen")
    validate_search_contracts(decontamination.get("search_contracts"))
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
    total_training_samples: int | None = None,
    total_training_shards: int | None = None,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """Choose the smallest per-source train-shard prefixes for one run budget."""

    validate_release_manifest(manifest)
    if (total_training_samples is None) == (total_training_shards is None):
        raise ValueError("choose exactly one training sample budget or shard count")
    if total_training_samples is not None and total_training_samples <= 0:
        raise ValueError("total_training_samples must be positive")
    if set(weights) != set(SOURCES) or any(float(value) <= 0 for value in weights.values()):
        raise ValueError(f"weights must provide positive values for exactly {SOURCES}")
    denominator = sum(float(value) for value in weights.values())
    if denominator <= 0:
        raise ValueError("at least one source weight must be positive")

    shard_counts = dict.fromkeys(SOURCES, 0)
    if total_training_shards is not None:
        maximum = sum(len(manifest["sources"][source]["train"]) for source in SOURCES)
        if not len(SOURCES) <= total_training_shards <= maximum:
            raise ValueError(f"training-shards must be between {len(SOURCES)} and {maximum}")
        records = dict.fromkeys(SOURCES, 0)
        for _ in range(total_training_shards):
            # Extend the source with the least coverage of the requested mixture.
            source = min(
                (s for s in SOURCES if shard_counts[s] < len(manifest["sources"][s]["train"])),
                key=lambda s: records[s] / float(weights[s]),
            )
            shard = manifest["sources"][source]["train"][shard_counts[source]]
            records[source] += int(shard["records"])
            shard_counts[source] += 1

    selected: dict[str, Any] = {}
    all_paths: list[str] = []
    selected_records = selected_residues = selected_bytes = 0
    for source in SOURCES:
        required = (
            math.ceil(total_training_samples * float(weights[source]) / denominator)
            if total_training_samples is not None
            else 0
        )
        available = 0
        residues = 0
        compressed_bytes = 0
        train: list[Mapping[str, Any]] = []
        for shard in manifest["sources"][source]["train"]:
            if (
                len(train) >= shard_counts[source]
                if total_training_shards is not None
                else available >= required
            ):
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
    plan = {
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
    if total_training_shards is not None:
        plan["total_training_shards"] = total_training_shards
    return plan


def fetch_release_plan(
    *,
    repo_id: str,
    revision: str,
    cache_root: Path,
    total_training_samples: int | None = None,
    total_training_shards: int | None = None,
    weights: Mapping[str, float],
    download_workers: int = 8,
    plan_only: bool = False,
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
    plan = plan_shards(
        manifest,
        total_training_samples=total_training_samples,
        total_training_shards=total_training_shards,
        weights=weights,
    )
    plan["repo_id"] = repo_id
    plan["requested_revision"] = revision
    plan["revision"] = resolved_revision
    plan["release_manifest_sha256"] = file_sha256(manifest_path)
    if plan_only:
        return plan, manifest_path
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
    output = cache_root / "download-plan.json"
    output.write_text(_canonical_json(plan))
    return plan, manifest_path


def _materialize_source(args: tuple[str, Mapping[str, Any], Path, Path]) -> tuple[str, dict]:
    """Materialize one source with bounded Parquet batches and fast ASCII tokenization."""
    import numpy as np

    source, selected, cache_root, output_root = args
    tokenizer = ProteinTokenizer.esmc()
    translation = bytes(tokenizer.token_to_id.get(chr(i), tokenizer.unk_id) for i in range(256))
    splits = {}
    for split in ("train", "validation"):
        writer = _StoreWriter(output_root / source / split)
        try:
            observed_previous = ""
            for shard in selected[split]:
                local = cache_root / str(shard["path"])
                if file_sha256(local) != shard["sha256"]:
                    raise ValueError(f"shard checksum mismatch: {local}")
                records = residues = 0
                for batch in pq.ParquetFile(local).iter_batches(
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
                        # Same strip/uppercase/unknown-token semantics as encode_residues,
                        # without a Python generator visit for every residue in a large corpus.
                        encoded = (
                            sequence.strip().upper().encode("ascii").translate(translation)
                        )
                        writer.append(np.frombuffer(encoded, dtype=np.uint8), digest)
                        records += 1
                        residues += int(length)
                if records != shard["records"] or residues != shard["residues"]:
                    raise ValueError(f"shard record/residue count mismatch: {local}")
                print(
                    json.dumps(
                        {
                            "event": "shard_materialized",
                            "source": source,
                            "split": split,
                            "path": shard["path"],
                            "records": records,
                        }
                    ),
                    flush=True,
                )
            splits[split] = writer.finish()
        finally:
            writer.handle.close()
    return source, splits


def materialize_plan(
    plan: Mapping[str, Any],
    release_manifest: Mapping[str, Any],
    cache_root: Path,
    output_root: Path,
    *,
    workers: int = 1,
    reuse_root: Path | None = None,
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
    if workers <= 0:
        raise ValueError("materialization workers must be positive")
    output_root.mkdir(parents=True)
    source_receipts = {}
    if reuse_root is not None:
        import os
        import shutil

        previous_plan = json.loads((reuse_root / "download-plan.json").read_text())
        previous = json.loads((reuse_root / "manifest.json").read_text())
        if previous["release_manifest_sha256"] != plan["release_manifest_sha256"]:
            raise ValueError("reused sources must belong to the same immutable release")
        validate_prepared_plan(previous_plan, reuse_root)
        for source in SOURCES:
            if all(
                previous_plan["sources"][source][split] == plan["sources"][source][split]
                for split in ("train", "validation")
            ):
                shutil.copytree(
                    reuse_root / source, output_root / source, copy_function=os.link
                )
                source_receipts[source] = previous["sources"][source]
    tasks = [
        (source, plan["sources"][source], cache_root, output_root)
        for source in SOURCES
        if source not in source_receipts
    ]
    if workers == 1:
        source_receipts.update(map(_materialize_source, tasks))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(workers, len(SOURCES))) as pool:
            source_receipts.update(pool.map(_materialize_source, tasks))
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
                "search_contracts": release_decontamination["search_contracts"],
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
    (output_root / "download-plan.json").write_text(_canonical_json(plan))
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


def validate_prepared_plan(plan: Mapping[str, Any], output_root: Path) -> None:
    """Reuse only the same verified release and exact requested whole-shard prefix."""
    from .data import TokenStore
    from .train import validate_data_manifest

    manifest = validate_data_manifest(output_root)
    if manifest["release_manifest_sha256"] != plan["release_manifest_sha256"]:
        raise ValueError("existing corpus uses a different release; choose a fresh DATA_ROOT")
    for source in SOURCES:
        for split in ("train", "validation"):
            selected = plan["sources"][source][split]
            expected = sum(int(row["records"]) for row in selected)
            observed = manifest["sources"][source][split]
            if observed["records"] != expected or observed["residues"] != sum(
                int(row["residues"]) for row in selected
            ):
                raise ValueError(
                    "existing corpus differs from requested shard count; "
                    "choose a fresh DATA_ROOT for another selection"
                )
            for filename, key in (
                ("tokens.bin", "tokens_sha256"),
                ("index.npy", "index_sha256"),
            ):
                if file_sha256(output_root / source / split / filename) != observed[key]:
                    raise ValueError(
                        f"prepared store checksum mismatch: {source}/{split}/{filename}"
                    )
            if TokenStore.open(output_root / source / split).index.size != expected:
                raise ValueError(f"incomplete prepared store: {source}/{split}")


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
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--training-samples", type=int)
    budget.add_argument(
        "--training-shards",
        type=int,
        help="total train Parquet shards across all sources; validation is always complete",
    )
    parser.add_argument(
        "--plan-only", action="store_true", help="show counts without fetching shards"
    )
    parser.add_argument(
        "--reuse", action="store_true", help="reuse an identical prepared corpus"
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--materialize-workers", type=int, default=1)
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
        total_training_shards=args.training_shards,
        weights=args.weights,
        download_workers=args.download_workers,
        plan_only=args.plan_only or args.output_root.exists(),
    )
    if args.plan_only:
        print(json.dumps(plan, sort_keys=True))
        return
    if args.output_root.exists():
        if not args.reuse:
            raise FileExistsError(f"refusing to overwrite prepared corpus: {args.output_root}")
        validate_prepared_plan(plan, args.output_root)
        print(f"Reusing verified training and MLM validation data: {args.output_root}")
        return
    release_manifest = json.loads(manifest_path.read_text())
    receipt = materialize_plan(
        plan,
        release_manifest,
        args.cache_root,
        args.output_root,
        workers=args.materialize_workers,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
