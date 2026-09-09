"""Install and verify the frozen contact P@L data and evaluator under DATA_ROOT."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .data import file_sha256

# Immutable evaluation release; independent of the pinned training-data revision.
BUNDLE_URL = (
    "https://huggingface.co/datasets/LuminScience/LuminBench-Nano-ESMC/resolve/"
    "cc548944b9caaf8b4f40ba49b316cc6f477a6031/evaluation/contact-evaluation-v1.tar.gz"
)
BUNDLE_SHA256 = "0a9f19e60643114ee98db1b89d3b9611a9e55f6c876a18b1d951e6f16e389eb6"
MANIFEST_SHA256 = "c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3"
INVENTORY_SHA256 = "1b73f5f466420c8d0c74be452ebabe46af837482cee357674cad01d99e6f4b70"
SOURCE_MANIFEST_SHA256 = "2fad51df6f4f2586b6648b9e4f575058f1999eba60175cba9fd7834f36951e55"


def _inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def verify_evaluation(root: Path) -> dict[str, object]:
    contact = root / "contact"
    source = root / "source"
    for path, digest in [
        (contact / "CONTACT_MANIFEST.jsonl", MANIFEST_SHA256),
        (contact / "PAYLOAD_INVENTORY.json", INVENTORY_SHA256),
        (source / "SOURCE_MANIFEST.json", SOURCE_MANIFEST_SHA256),
    ]:
        if file_sha256(path) != digest:
            raise ValueError(f"frozen evaluation checksum mismatch: {path}")
    sources = json.loads((source / "SOURCE_MANIFEST.json").read_text())
    for name, digest in sources["files"].items():
        if file_sha256(_inside(source, name)) != digest:
            raise ValueError(f"frozen evaluator source changed: {name}")
    ready = json.loads((contact / "PDB_CONTACT_DATASET_READY.json").read_text())
    if ready["manifest_path"] != "CONTACT_MANIFEST.jsonl":
        raise ValueError("unexpected contact manifest path")
    sys.path.insert(0, str(source))
    try:
        from autoresearch_esm.paper_contact_runtime import ContactDataset

        dataset = ContactDataset(contact)
    finally:
        sys.path.pop(0)
    inventory = json.loads((contact / "PAYLOAD_INVENTORY.json").read_text())
    for row in inventory:
        if file_sha256(_inside(contact, row["path"])) != row["sha256"]:
            raise ValueError(f"contact payload checksum mismatch: {row['chain_id']}")
    return {
        "status": "verified",
        "manifest_sha256": MANIFEST_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "probe_fit_chains": len(dataset.train_ids[:16]),
        "probe_validation_chains": len(dataset.train_ids[16:]),
        "evaluation_chains": len(dataset.eval_ids),
    }


def extract_bundle(archive: Path, destination: Path) -> None:
    if file_sha256(archive) != BUNDLE_SHA256:
        raise ValueError(f"contact archive checksum mismatch: {archive}")
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            _inside(destination, member.name)
            if not member.isfile() and not member.isdir():
                raise ValueError(f"unsupported archive member: {member.name}")
        handle.extractall(destination, filter="data")


def prepare_evaluation(data_root: Path, archive: Path | None = None) -> dict[str, object]:
    output = data_root / "evaluation"
    if (output / "contact").exists() and (output / "source").exists():
        return verify_evaluation(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"incomplete evaluation data at {output}; move it aside and retry"
        )
    if archive is None:
        cache = data_root / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / "contact-evaluation-v1.tar.gz"
        if not archive.exists() or file_sha256(archive) != BUNDLE_SHA256:
            temporary = archive.with_suffix(".gz.partial")
            print("Downloading frozen contact P@L data and evaluator", flush=True)
            try:
                with urllib.request.urlopen(BUNDLE_URL, timeout=60) as response:
                    with temporary.open("wb") as handle:
                        shutil.copyfileobj(response, handle, length=8 << 20)
                if file_sha256(temporary) != BUNDLE_SHA256:
                    raise ValueError("downloaded contact archive checksum mismatch")
                temporary.replace(archive)
            finally:
                temporary.unlink(missing_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=data_root) as name:
        staged = Path(name) / "evaluation"
        staged.mkdir()
        extract_bundle(archive, staged)
        receipt = verify_evaluation(staged)
        (staged / "SETUP_VERIFIED.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if output.exists():
            output.rmdir()  # Only an empty destination can reach this point.
        staged.rename(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--archive", type=Path, help="use a previously downloaded frozen bundle"
    )
    args = parser.parse_args()
    print(json.dumps(prepare_evaluation(args.data_root, args.archive), indent=2))


if __name__ == "__main__":
    main()
