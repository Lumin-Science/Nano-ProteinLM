"""Install recovered contact representatives and precompute the v3 evaluation masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from .data import file_sha256
from .evaluate import write_json
from .evaluation_profiles import PROTOCOL, SEARCH_IDS_SHA256
from .prepared_mlm import MASK_SEEDS, original_examples, prepare_cache, verify_cache
from .tokenizer import ProteinTokenizer


def canonical(value) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n").encode()


def prepare_profiles(data_root: Path, archive: Path | None = None) -> dict:
    evaluation = data_root / "evaluation"
    contact = evaluation / "contact-v3"
    if not contact.exists():
        from .released_contact_v3 import install_released_contact_pool
        install_released_contact_pool(data_root, archive)
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
    parser.add_argument("--contact-v3-archive", type=Path, help="use a downloaded v3 bundle offline")
    args = parser.parse_args()
    report = prepare_profiles(args.data_root, args.contact_v3_archive)
    print(json.dumps({"event": "evaluation_profiles_prepared", "protocol": report["protocol"], "caches": len(report["caches"])}))


if __name__ == "__main__":
    main()
