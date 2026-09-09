"""Package the existing frozen contact payload and evaluator without changing either."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".dev/reports/fir-r02-rope10k-100k-20260906/launch/contact-evaluator-src"
MANIFEST_SHA256 = "c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3"
INVENTORY_SHA256 = "1b73f5f466420c8d0c74be452ebabe46af837482cee357674cad01d99e6f4b70"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def package(contact: Path, output: Path) -> dict:
    assert sha256(contact / "CONTACT_MANIFEST.jsonl") == MANIFEST_SHA256
    assert sha256(contact / "PAYLOAD_INVENTORY.json") == INVENTORY_SHA256
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
    source_manifest = json.loads((SOURCE / "SOURCE_MANIFEST.json").read_text())
    for name, digest in source_manifest["files"].items():
        path = SOURCE / name
        assert sha256(path) == digest, name
        files["source/" + name] = path
    files["source/SOURCE_MANIFEST.json"] = SOURCE / "SOURCE_MANIFEST.json"
    files["LICENSE"] = ROOT / "LICENSE"
    files["ATTRIBUTION.md"] = ROOT / "docs/CONTACT_DATA.md"
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
        "source_manifest_sha256": sha256(SOURCE / "SOURCE_MANIFEST.json"),
        "probe_fit_chains": 16,
        "probe_validation_chains": 4,
        "evaluation_chains": 20775,
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.contact_root, args.output), indent=2))
