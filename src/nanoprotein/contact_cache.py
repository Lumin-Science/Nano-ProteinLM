"""Receipt-bound static scoring cache for the frozen 20,775-chain P@L set."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import file_sha256

CACHE_PROTOCOL = "autoresearch-contact-static-scoring-cache-v1"
VERIFICATION_PROTOCOL = "autoresearch-contact-cache-preflight-v1"


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class CachedContactChain:
    chain_id: str
    sequence: str
    source_length: int
    evaluated_length: int
    pair_count: int
    valid_pair_count: int
    true_long_range_contacts: int
    valid_offset: int
    valid_bytes: int
    contact_offset: int
    contact_bytes: int
    source_payload_sha256: str


def build_contact_scoring_cache(
    *, dataset_root: Path, external_src: Path, output_root: Path, workers: int
) -> dict[str, Any]:
    """Validate source payloads once and freeze only static scoring labels."""

    if output_root.exists():
        raise FileExistsError(f"refusing to reuse contact cache output: {output_root}")
    if workers <= 0:
        raise ValueError("contact cache workers must be positive")
    import sys

    sys.path.insert(0, str(external_src))
    try:
        from autoresearch_esm.paper_contact import (  # type: ignore[import-not-found]
            DISTANCE_THRESHOLD_ANGSTROM,
            SEQUENCE_SEPARATION,
        )
        from autoresearch_esm.paper_contact_runtime import (  # type: ignore[import-not-found]
            ContactDataset,
        )
    finally:
        sys.path.pop(0)
    dataset = ContactDataset(dataset_root)
    output_root.mkdir(parents=True)
    masks_path = output_root / "masks.bin"
    entries_path = output_root / "entries.jsonl"

    def prepare(chain_id: str) -> tuple[dict[str, Any], bytes, bytes]:
        payload, chain = dataset.load_payload(chain_id)
        sequence = chain.sequence[:510]
        source_length = int(payload["source_length"])
        evaluated_length = min(source_length, len(sequence), chain.cb_distances.shape[0])
        i, j = np.triu_indices(evaluated_length, k=SEQUENCE_SEPARATION)
        distances = chain.cb_distances[i, j]
        valid = np.isfinite(distances)
        contacts = valid & (distances < DISTANCE_THRESHOLD_ANGSTROM)
        valid_count = int(valid.sum())
        true_contacts = int(contacts.sum())
        if valid_count < evaluated_length or true_contacts < evaluated_length:
            raise ValueError(f"frozen eligible contact chain became ineligible: {chain_id}")
        valid_packet = np.packbits(valid, bitorder="little").tobytes()
        contact_packet = np.packbits(contacts, bitorder="little").tobytes()
        entry = dataset.entry_by_id[chain_id]
        return (
            {
                "chain_id": chain_id,
                "sequence": sequence,
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                "source_length": source_length,
                "evaluated_length": evaluated_length,
                "pair_count": int(i.size),
                "valid_pair_count": valid_count,
                "true_long_range_contacts": true_contacts,
                "source_payload_sha256": entry.source_payload_sha256,
            },
            valid_packet,
            contact_packet,
        )

    entries: list[dict[str, Any]] = []
    offset = 0
    with (
        masks_path.open("wb") as masks_handle,
        entries_path.open("w") as entries_handle,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        for row, valid_packet, contact_packet in pool.map(prepare, dataset.eval_ids):
            row["valid_offset"] = offset
            row["valid_bytes"] = len(valid_packet)
            masks_handle.write(valid_packet)
            offset += len(valid_packet)
            row["contact_offset"] = offset
            row["contact_bytes"] = len(contact_packet)
            masks_handle.write(contact_packet)
            offset += len(contact_packet)
            entries_handle.write(json.dumps(row, sort_keys=True) + "\n")
            entries.append(row)
    if len(entries) != len(dataset.eval_ids):
        raise RuntimeError("contact cache did not cover every evaluation chain")
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "protocol": CACHE_PROTOCOL,
        "dataset_root": str(dataset.root),
        "dataset_manifest_sha256": dataset.manifest_receipt.manifest_sha256,
        "dataset_ready_sha256": file_sha256(dataset.ready_path),
        "evaluation_chains": len(entries),
        "selection": {
            "maximum_model_residues": 510,
            "sequence_separation": SEQUENCE_SEPARATION,
            "distance_threshold_angstrom": DISTANCE_THRESHOLD_ANGSTROM,
            "unresolved_pairs": "excluded",
            "top_l_tie_break": "stable_upper_triangle_order",
        },
        "entries": {
            "path": entries_path.name,
            "bytes": entries_path.stat().st_size,
            "sha256": file_sha256(entries_path),
        },
        "masks": {
            "path": masks_path.name,
            "bytes": masks_path.stat().st_size,
            "sha256": file_sha256(masks_path),
            "packing": "numpy-packbits-little-valid-then-contact-per-chain",
        },
    }
    receipt_path = output_root / "CONTACT_SCORING_CACHE.json"
    receipt_path.write_text(_canonical_json(receipt))
    return receipt


def verify_contact_scoring_cache(*, cache_root: Path, output: Path) -> dict[str, Any]:
    """Hash the large immutable cache once before parallel evaluator shards."""

    receipt_path = cache_root / "CONTACT_SCORING_CACHE.json"
    receipt = json.loads(receipt_path.read_text())
    if not (
        receipt.get("status") == "complete"
        and receipt.get("protocol") == CACHE_PROTOCOL
        and receipt.get("evaluation_chains") == 20_775
    ):
        raise ValueError("contact scoring cache receipt is incomplete")
    for key in ("entries", "masks"):
        row = receipt[key]
        path = cache_root / row["path"]
        if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
            raise ValueError(f"contact scoring cache artifact changed: {key}")
    verification = {
        "schema_version": 1,
        "status": "verified",
        "protocol": VERIFICATION_PROTOCOL,
        "cache_root": str(cache_root.resolve()),
        "cache_receipt_sha256": file_sha256(receipt_path),
        "dataset_manifest_sha256": receipt["dataset_manifest_sha256"],
        "entries_sha256": receipt["entries"]["sha256"],
        "masks_sha256": receipt["masks"]["sha256"],
    }
    output.write_text(_canonical_json(verification))
    return verification


class ContactScoringCache:
    """Cheap shard-local reader trusted through a just-written preflight receipt."""

    def __init__(
        self,
        *,
        root: Path,
        preflight: Path,
        dataset_manifest_sha256: str,
        expected_chain_ids: list[str],
    ) -> None:
        receipt_path = root / "CONTACT_SCORING_CACHE.json"
        receipt = json.loads(receipt_path.read_text())
        verification = json.loads(preflight.read_text())
        if not (
            verification.get("status") == "verified"
            and verification.get("protocol") == VERIFICATION_PROTOCOL
            and Path(verification.get("cache_root", "")).resolve() == root.resolve()
            and verification.get("cache_receipt_sha256") == file_sha256(receipt_path)
            and verification.get("dataset_manifest_sha256") == dataset_manifest_sha256
            and receipt.get("dataset_manifest_sha256") == dataset_manifest_sha256
            and verification.get("entries_sha256") == receipt["entries"]["sha256"]
            and verification.get("masks_sha256") == receipt["masks"]["sha256"]
        ):
            raise ValueError("contact cache preflight binding changed")
        entries_path = root / receipt["entries"]["path"]
        if file_sha256(entries_path) != receipt["entries"]["sha256"]:
            raise ValueError("contact cache entries changed after preflight")
        rows = [json.loads(line) for line in entries_path.read_text().splitlines()]
        if [row["chain_id"] for row in rows] != expected_chain_ids:
            raise ValueError("contact cache chain order changed")
        self.entries = {
            row["chain_id"]: CachedContactChain(
                **{key: row[key] for key in CachedContactChain.__dataclass_fields__}
            )
            for row in rows
        }
        masks_path = root / receipt["masks"]["path"]
        if masks_path.stat().st_size != receipt["masks"]["bytes"]:
            raise ValueError("contact cache masks changed after preflight")
        self.masks = np.memmap(masks_path, mode="r", dtype=np.uint8)

    def score(self, chain_id: str, scores: np.ndarray) -> dict[str, object]:
        row = self.entries[chain_id]
        i, j = np.triu_indices(row.evaluated_length, k=24)
        if i.size != row.pair_count:
            raise ValueError("contact cache pair cardinality changed")
        valid_packet = self.masks[row.valid_offset : row.valid_offset + row.valid_bytes]
        contact_packet = self.masks[row.contact_offset : row.contact_offset + row.contact_bytes]
        valid = np.unpackbits(valid_packet, bitorder="little", count=row.pair_count).astype(
            bool, copy=False
        )
        contacts = np.unpackbits(
            contact_packet, bitorder="little", count=row.pair_count
        ).astype(bool, copy=False)
        pair_scores = np.asarray(scores[: row.evaluated_length, : row.evaluated_length])[i, j]
        pair_scores = pair_scores[valid]
        labels = contacts[valid]
        if (
            pair_scores.size != row.valid_pair_count
            or int(labels.sum()) != row.true_long_range_contacts
        ):
            raise ValueError("contact cache static scoring counts changed")
        order = np.argsort(-pair_scores, kind="stable")[: row.evaluated_length]
        return {
            "chain_id": chain_id,
            "source_length": row.source_length,
            "evaluated_length": row.evaluated_length,
            "eligible_pair_count": row.valid_pair_count,
            "true_long_range_contacts": row.true_long_range_contacts,
            "precision_at_l": float(labels[order].mean()),
            "random_precision_at_l": float(labels.mean()),
        }
