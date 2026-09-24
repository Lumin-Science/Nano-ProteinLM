"""Package verified evaluation assets without machine paths or experiment metadata."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

MANIFEST_SHA256 = "c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3"
INVENTORY_SHA256 = "1b73f5f466420c8d0c74be452ebabe46af837482cee357674cad01d99e6f4b70"
SOURCE_MANIFEST_SHA256 = "295d82b3ad1ed95769659bd2ba52902cc8c260aaee06ae3c91c68c472c65e012"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def package(evaluation: Path, output: Path) -> dict:
    contact = evaluation / "contact"
    source = evaluation / "source"
    assert sha256(contact / "CONTACT_MANIFEST.jsonl") == MANIFEST_SHA256
    assert sha256(contact / "PAYLOAD_INVENTORY.json") == INVENTORY_SHA256
    assert sha256(source / "SOURCE_MANIFEST.json") == SOURCE_MANIFEST_SHA256
    ready = json.loads((contact / "PDB_CONTACT_DATASET_READY.json").read_text())
    assert "mmseqs_commands" not in ready, "use the prepared clean evaluation assets"
    inventory = json.loads((contact / "PAYLOAD_INVENTORY.json").read_text())
    assert len(inventory) == 20795
    files = {
        "contact/" + name: contact / name
        for name in [
            "CONTACT_MANIFEST.jsonl",
            "PAYLOAD_INVENTORY.json",
            "PDB_CONTACT_DATASET_READY.json",
        ]
    }
    for row in inventory:
        path = (contact / row["path"]).resolve(strict=True)
        path.relative_to(contact.resolve())
        assert sha256(path) == row["sha256"], row["chain_id"]
        files["contact/" + row["path"]] = path
    source_manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text())
    assert set(source_manifest) == {"files"}
    for name, digest in source_manifest["files"].items():
        path = source / name
        path.resolve(strict=True).relative_to(source.resolve())
        assert sha256(path) == digest, name
        files["source/" + name] = path
    files["source/SOURCE_MANIFEST.json"] = source / "SOURCE_MANIFEST.json"
    files["LICENSE"] = evaluation / "LICENSE"
    files["ATTRIBUTION.md"] = evaluation / "ATTRIBUTION.md"
    for name in ("ATTRIBUTION.md", "contact/PDB_CONTACT_DATASET_READY.json"):
        text = files[name].read_text()
        assert all(term not in text for term in (".dev/reports", "/scratch/", "best-recipe"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, path in sorted(files.items()):
                    info = tarfile.TarInfo(name)
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    receipt = {
        "archive": output.name,
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "files": len(files),
        "uncompressed_bytes": sum(p.stat().st_size for p in files.values()),
        "manifest_sha256": MANIFEST_SHA256,
        "inventory_sha256": INVENTORY_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "probe_fit_chains": 16,
        "probe_validation_chains": 4,
        "evaluation_chains": 20775,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.evaluation_root, args.output), indent=2))
