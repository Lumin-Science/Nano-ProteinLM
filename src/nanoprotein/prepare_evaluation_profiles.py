"""Install recovered contact representatives and precompute the v3 evaluation masks."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from .data import file_sha256
from .evaluate import write_json
from .evaluation_profiles import PROTOCOL, SEARCH_IDS_SHA256
from .prepared_mlm import MASK_SEEDS, original_examples, prepare_cache, verify_cache
from .tokenizer import ProteinTokenizer


def canonical(value) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()


def install_contact_pool(evaluation_root: Path, recovery_root: Path) -> dict:
    from .contact_dataset import ContactDataset

    sys.path.insert(0, str(evaluation_root / "source"))
    try:
        from autoresearch_esm.paper_contact_data import manifest_entry, parse_normalized_chain
    finally:
        sys.path.pop(0)
    recovery = json.loads((recovery_root / "RECOVERY.json").read_text())
    metadata = [json.loads(line) for line in (recovery_root / "eligible_26082.jsonl").read_text().splitlines()]
    if len(metadata) != 26082 or len({row["chain_id"] for row in metadata}) != 26082:
        raise ValueError("recovery must contain exactly 26082 unique eligible chains")
    search_path = recovery_root / "search_8192.ids"
    if file_sha256(search_path) != SEARCH_IDS_SHA256:
        raise ValueError("search subset changed")
    original = ContactDataset(evaluation_root / "contact")
    if original.manifest_receipt.manifest_sha256 != "c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3":
        raise ValueError("original contact release changed")
    all_ids = {row["chain_id"] for row in metadata}
    eval_ids = sorted(all_ids - set(original.train_ids), key=lambda chain: hashlib.sha256(f"20260820:{chain}".encode()).digest())
    if len(eval_ids) != 26062 or not set(search_path.read_text().splitlines()).issubset(eval_ids):
        raise ValueError("probe/search/evaluation split mismatch")
    output = evaluation_root / "contact-v3"
    if output.exists():
        dataset = ContactDataset(output)
        if dataset.ready.get("population_protocol") != "nanoprotein-contact-pool-v3":
            raise ValueError("existing expanded population has wrong protocol")
        return dataset.ready
    details = {row["chain_id"]: row for row in metadata}
    entries, inventory = {}, {}
    with tempfile.TemporaryDirectory(prefix=".contact-v3-", dir=evaluation_root) as tmp:
        stage = Path(tmp) / "contact-v3"
        (stage / "payloads").mkdir(parents=True)
        for archive in recovery["coordinate_archives"]:
            path = recovery_root / archive["archive"]
            if file_sha256(path) != archive["archive_sha256"]:
                raise ValueError(f"recovery archive changed: {path}")
            with gzip.open(path, "rt") as handle:
                for line in handle:
                    record = json.loads(line)
                    payload = record["payload"]
                    chain_id = payload["chain_id"]
                    if chain_id not in all_ids or chain_id in entries:
                        raise ValueError("unexpected/duplicate recovered chain")
                    encoded = canonical(payload)
                    digest = hashlib.sha256(encoded).hexdigest()
                    if digest != details[chain_id]["source_payload_sha256"]:
                        raise ValueError("recovered payload differs from verified metadata")
                    name = f"payloads/{digest}.json"
                    (stage / name).write_bytes(encoded)
                    role = "train" if chain_id in original.train_ids else "eval"
                    if chain_id in original.entry_by_id:
                        entry = original.entry_by_id[chain_id]
                        if entry.source_payload_sha256 != digest or entry.role != role:
                            raise ValueError("recovered original payload differs from published benchmark")
                    else:
                        entry = manifest_entry(parse_normalized_chain(payload), source_payload_sha256=digest, role=role)
                    entries[chain_id] = asdict(entry)
                    inventory[chain_id] = {"chain_id": chain_id, "path": name, "sha256": digest}
        order = original.train_ids + eval_ids
        if set(entries) != set(order):
            raise ValueError("recovery archives do not cover the population")
        (stage / "CONTACT_MANIFEST.jsonl").write_bytes(b"".join(canonical(entries[chain]) for chain in order))
        write_json(stage / "PAYLOAD_INVENTORY.json", [inventory[chain] for chain in order])
        shutil.copyfile(search_path, stage / "search_8192.ids")
        (stage / "evaluation_26062.ids").write_text("\n".join(eval_ids) + "\n")
        (stage / "probe_20.ids").write_text("\n".join(original.train_ids) + "\n")
        sequence_rows = [{"chain_id": chain, "sequence": details[chain]["sequence"],
                          "sequence_sha256": details[chain]["sequence_sha256"]} for chain in eval_ids]
        (stage / "SEQUENCES.jsonl").write_bytes(b"".join(canonical(row) for row in sequence_rows))
        ready = {
            "event": "paper_contact_dataset_ready", "population_protocol": "nanoprotein-contact-pool-v3",
            "manifest_path": "CONTACT_MANIFEST.jsonl", "manifest_sha256": file_sha256(stage / "CONTACT_MANIFEST.jsonl"),
            "payload_inventory_sha256": file_sha256(stage / "PAYLOAD_INVENTORY.json"),
            "sequences_sha256": file_sha256(stage / "SEQUENCES.jsonl"),
            "evaluation_ids_sha256": file_sha256(stage / "evaluation_26062.ids"),
            "search_ids_sha256": SEARCH_IDS_SHA256, "evaluation_chains": 26062, "probe_chains": 20,
            "probe_train_chain_ids": original.train_ids[:16], "probe_validation_chain_ids": original.train_ids[16:],
            "recovery_receipt_sha256": file_sha256(recovery_root / "RECOVERY.json"),
            "original_manifest_sha256": original.manifest_receipt.manifest_sha256,
            "training_overlap_status": "original_20775_and_20_probes_protected; additional_5287_not_verified_for_homology_exclusion",
        }
        write_json(stage / "PDB_CONTACT_DATASET_READY.json", ready)
        ContactDataset(stage)
        stage.rename(output)
    return ready


def prepare_profiles(data_root: Path, recovery_root: Path | None = None, archive: Path | None = None) -> dict:
    evaluation = data_root / "evaluation"
    contact = evaluation / "contact-v3"
    if not contact.exists():
        if recovery_root is None:
            from .released_contact_v3 import install_released_contact_pool
            install_released_contact_pool(data_root, archive)
        else:
            install_contact_pool(evaluation, recovery_root)
    # Organizer-provided overlap evidence accompanies this data version. It does
    # not alter historical training stores or relabel them as decontaminated.
    audit_source = recovery_root / "TRAINING_OVERLAP_AUDIT.json" if recovery_root else None
    if audit_source is not None and audit_source.exists():
        audit = json.loads(audit_source.read_text())
        if audit.get("status") != "complete" or audit.get("query_chains") != 26082:
            raise ValueError("expanded-pool overlap audit is incomplete")
        destination = contact / "TRAINING_OVERLAP_AUDIT.json"
        shutil.copyfile(audit_source, destination)
        ready_path = contact / "PDB_CONTACT_DATASET_READY.json"
        evidence = json.loads(ready_path.read_text())
        evidence["training_overlap_audit_sha256"] = file_sha256(destination)
        evidence["training_overlap_status"] = "known_exact_training_sequence_overlap; expanded_pool_homology_exclusion_not_established"
        write_json(ready_path, evidence)
    from .contact_dataset import ContactDataset
    sys.path.insert(0, str(evaluation / "source"))
    try:
        dataset = ContactDataset(contact)
    finally:
        sys.path.pop(0)
    ready = dataset.ready
    if ready.get("population_protocol") != "nanoprotein-contact-pool-v3":
        raise ValueError("expanded contact protocol changed")
    for name, key in (("SEQUENCES.jsonl", "sequences_sha256"), ("evaluation_26062.ids", "evaluation_ids_sha256"), ("search_8192.ids", "search_ids_sha256")):
        if file_sha256(contact / name) != ready[key]:
            raise ValueError(f"expanded population artifact changed: {name}")
    output = evaluation / "prepared-v3"
    originals, original_binding = original_examples(data_root / "training")
    if output.exists():
        report = json.loads((output / "PREPARED_INPUTS.json").read_text())
        if report["protocol"] != PROTOCOL or report["original_validation_binding"] != original_binding or report["contact_manifest_sha256"] != ready["manifest_sha256"]:
            raise ValueError("existing prepared inputs have a different population")
        for name, receipt in report["caches"].items():
            verify_cache(output / name, expected=receipt)
        return report
    tokenizer = ProteinTokenizer.esmc()
    sequences = {row["chain_id"]: row for row in map(json.loads, (contact / "SEQUENCES.jsonl").read_text().splitlines())}
    if list(sequences) != dataset.eval_ids:
        raise ValueError("MLM sequences do not cover the contact population in order")
    for chain, row in sequences.items():
        if hashlib.sha256(row["sequence"].encode("ascii")).hexdigest() != dataset.entry_by_id[chain].sequence_sha256:
            raise ValueError("MLM sequence differs from coordinate payload manifest")
    search_ids = (contact / "search_8192.ids").read_text().splitlines()
    def contact_examples(ids):
        return [(chain, tokenizer.encode_residues(sequences[chain]["sequence"]),
                 bytes.fromhex(sequences[chain]["sequence_sha256"])) for chain in ids]
    populations = {"search8192": contact_examples(search_ids), "contact26062": contact_examples(dataset.eval_ids), "original12288": originals}
    with tempfile.TemporaryDirectory(prefix=".prepared-v3-", dir=evaluation) as tmp:
        stage = Path(tmp) / "prepared-v3"
        stage.mkdir()
        caches = {}
        for population, examples in populations.items():
            binding = original_binding if population == "original12288" else {"contact_manifest_sha256": ready["manifest_sha256"]}
            for seed in MASK_SEEDS[:1] if population == "search8192" else MASK_SEEDS:
                name = f"{population}-mask{seed}.npz"
                caches[name] = prepare_cache(stage / name, examples, population=population, seed=seed, binding=binding)
                print(f"Prepared {name}", flush=True)
        report = {"protocol": PROTOCOL, "contact_manifest_sha256": ready["manifest_sha256"],
                  "search_ids_sha256": SEARCH_IDS_SHA256, "original_validation_binding": original_binding,
                  "mask_seeds": list(MASK_SEEDS), "caches": caches}
        write_json(stage / "PREPARED_INPUTS.json", report)
        stage.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--recovered-contact-pool", type=Path)
    parser.add_argument("--contact-v3-archive", type=Path, help="use a downloaded v3 bundle offline")
    args = parser.parse_args()
    report = prepare_profiles(args.data_root, args.recovered_contact_pool, args.contact_v3_archive)
    print(json.dumps({"event": "evaluation_profiles_prepared", "protocol": report["protocol"], "caches": len(report["caches"])}))


if __name__ == "__main__":
    main()
