"""Verified contact populations; scoring mathematics remains in the frozen source bundle."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .data import file_sha256


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value


class ContactDataset:
    def __init__(self, root: Path):
        from autoresearch_esm.paper_contact_data import ContactManifestEntry, verify_contact_manifest
        self.root = root.resolve(strict=True)
        self.ready_path = self.root / "PDB_CONTACT_DATASET_READY.json"
        self.ready = _load_json(self.ready_path)
        if self.ready.get("event") != "paper_contact_dataset_ready":
            raise ValueError("contact dataset readiness event mismatch")
        manifest_path = (self.root / str(self.ready["manifest_path"])).resolve()
        manifest_path.relative_to(self.root)
        self.manifest_receipt = verify_contact_manifest(
            manifest_path, expected_sha256=str(self.ready["manifest_sha256"])
        )
        self.entries: list[ContactManifestEntry] = []
        with manifest_path.open() as handle:
            for line in handle:
                self.entries.append(ContactManifestEntry(**json.loads(line)))
        inventory_path = self.root / "PAYLOAD_INVENTORY.json"
        if file_sha256(inventory_path) != self.ready["payload_inventory_sha256"]:
            raise ValueError("contact payload inventory digest mismatch")
        inventory = json.loads(inventory_path.read_bytes())
        if not isinstance(inventory, list):
            raise TypeError("contact payload inventory must be a list")
        self.payloads = {str(row["chain_id"]): row for row in inventory}
        if len(self.payloads) != len(inventory) or set(self.payloads) != {
            entry.chain_id for entry in self.entries
        }:
            raise ValueError("contact payload inventory does not exactly cover the manifest")
        self.entry_by_id = {entry.chain_id: entry for entry in self.entries}
        if len(self.entry_by_id) != len(self.entries):
            raise ValueError("contact manifest contains duplicate chain IDs")
        self.train_ids = [entry.chain_id for entry in self.entries if entry.role == "train"]
        self.eval_ids = [entry.chain_id for entry in self.entries if entry.role == "eval"]
        expected = 26062 if self.ready.get("population_protocol") == "nanoprotein-contact-pool-v3" else 20775
        if len(self.train_ids) != 20 or len(self.eval_ids) != expected:
            raise ValueError("contact dataset does not have the declared exact train/eval coverage")
        if self.train_ids[:16] != list(self.ready["probe_train_chain_ids"]):
            raise ValueError("probe training chain order changed")
        if self.train_ids[16:] != list(self.ready["probe_validation_chain_ids"]):
            raise ValueError("probe validation chain order changed")

    def load_payload(self, chain_id: str) -> tuple[dict[str, Any], Any]:
        from autoresearch_esm.paper_contact_data import manifest_entry, parse_normalized_chain

        row = self.payloads[chain_id]
        path = (self.root / str(row["path"])).resolve(strict=True)
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"payload escapes contact dataset root: {path}") from error
        digest = file_sha256(path)
        entry = self.entry_by_id[chain_id]
        if digest != row["sha256"] or digest != entry.source_payload_sha256:
            raise ValueError(f"contact source payload digest mismatch: {chain_id}")
        payload = _load_json(path)
        chain = parse_normalized_chain(payload)
        observed = manifest_entry(chain, source_payload_sha256=digest, role=entry.role)
        if observed != entry:
            raise ValueError(f"contact payload differs from manifest: {chain_id}")
        return payload, chain
