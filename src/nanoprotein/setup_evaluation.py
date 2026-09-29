"""Install and verify the frozen contact P@L data and evaluator under DATA_ROOT."""

from __future__ import annotations

import argparse
import hashlib
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
    "5eae416dbb415d2df206b9641dd5ddabe04371dc/evaluation/contact-evaluation-v2.tar.gz"
)
BUNDLE_SHA256 = "5d901bb6fbde8face23be3e7bc84d776d4c73eecee6b5e852f86c3e3c9e82154"
MANIFEST_SHA256 = "c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3"
INVENTORY_SHA256 = "1b73f5f466420c8d0c74be452ebabe46af837482cee357674cad01d99e6f4b70"
SOURCE_MANIFEST_SHA256 = "295d82b3ad1ed95769659bd2ba52902cc8c260aaee06ae3c91c68c472c65e012"
BUNDLE_SOURCE_MANIFEST_SHA256 = SOURCE_MANIFEST_SHA256


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
    ]:
        if file_sha256(path) != digest:
            raise ValueError(f"frozen evaluation checksum mismatch: {path}")
    source_manifest = source / "SOURCE_MANIFEST.json"
    source_manifest_sha = file_sha256(source_manifest)
    if source_manifest_sha != SOURCE_MANIFEST_SHA256:
        raise ValueError(f"frozen evaluation checksum mismatch: {source_manifest}")
    sources = json.loads(source_manifest.read_text())
    for name, digest in sources["files"].items():
        if file_sha256(_inside(source, name)) != digest:
            raise ValueError(f"frozen evaluator source changed: {name}")
    ready = json.loads((contact / "PDB_CONTACT_DATASET_READY.json").read_text())
    if set(ready) - READY_FIELDS:
        raise ValueError("use a fresh evaluation root with this release")
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
        "source_manifest_sha256": source_manifest_sha,
        "probe_fit_chains": len(dataset.train_ids[:16]),
        "probe_validation_chains": len(dataset.train_ids[16:]),
        "evaluation_chains": len(dataset.eval_ids),
    }


READY_FIELDS = {
    "claim_level",
    "cluster_representatives",
    "driver_sha256",
    "eligible_cluster_representatives",
    "evaluation_chain_ids_sha256",
    "evaluation_chains",
    "event",
    "manifest_path",
    "manifest_sha256",
    "mmseqs_cluster_tsv_sha256",
    "parse_failures",
    "payload_inventory_sha256",
    "plan_sha256",
    "probe_train_chain_ids",
    "probe_validation_chain_ids",
    "protein_chains",
    "protocol_id",
    "schema",
    "seed",
    "selection",
    "shards",
    "source_ready_sha256",
    "source_structures",
    "train_chain_ids",
    "train_chains",
}
ATTRIBUTION = (
    "# Contact evaluation attribution\n\n"
    "Structures come from the 2024-02-28 RCSB Protein Data Bank snapshot. "
    "The selection includes 16 probe-fit chains, four probe-validation chains and "
    "20,775 evaluation chains. It follows the published procedure; it is not claimed "
    "to reproduce an unpublished author selection.\n\n"
    "PDB archive data are available under CC0 1.0: "
    "https://www.wwpdb.org/about/usage-policies. Acknowledge the PDB and original "
    "structure authors. Chain identifiers are retained in contact/CONTACT_MANIFEST.jsonl; "
    "PDB entry pages identify the associated publications.\n\n"
    "Berman, H. M. et al. The Protein Data Bank. Nucleic Acids Research 28, 235–242 (2000). "
    "https://doi.org/10.1093/nar/28.1.235\n\n"
    "Evaluator source is covered by the accompanying MIT LICENSE. Numerical sources, "
    "contact manifests and payloads retain their frozen SHA-256 checksums.\n"
)


def extract_bundle(archive: Path, destination: Path) -> None:
    """Install only declared evaluation payloads, code and license from a verified bundle."""
    if file_sha256(archive) != BUNDLE_SHA256:
        raise ValueError(f"contact archive checksum mismatch: {archive}")
    with tarfile.open(archive, "r:gz") as handle:
        members = {}
        for member in handle.getmembers():
            _inside(destination, member.name)
            if not member.isfile() and not member.isdir():
                raise ValueError(f"unsupported archive member: {member.name}")
            if member.name in members:
                raise ValueError(f"duplicate archive member: {member.name}")
            members[member.name] = member

        def read(name: str, expected: str | None = None) -> bytes:
            member = members[name]
            if not member.isfile():
                raise ValueError(f"expected a regular file: {name}")
            with handle.extractfile(member) as stream:
                content = stream.read()
            if expected is not None and hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"bundle member checksum mismatch: {name}")
            return content

        inventory = json.loads(read("contact/PAYLOAD_INVENTORY.json", INVENTORY_SHA256))
        sources = json.loads(read("source/SOURCE_MANIFEST.json", BUNDLE_SOURCE_MANIFEST_SHA256))
        wanted = {
            "LICENSE": None,
            "contact/CONTACT_MANIFEST.jsonl": MANIFEST_SHA256,
            "contact/PAYLOAD_INVENTORY.json": INVENTORY_SHA256,
            **{f"contact/{row['path']}": row["sha256"] for row in inventory},
            **{f"source/{name}": digest for name, digest in sources["files"].items()},
        }
        # Validate every destination before writing any payload.
        for name in wanted:
            _inside(destination, name)
        ready = json.loads(read("contact/PDB_CONTACT_DATASET_READY.json"))
        # Read in physical order: backwards seeks in gzip decompress the prefix again.
        for name in sorted(wanted, key=lambda name: members[name].offset):
            digest = wanted[name]
            target = _inside(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read(name, digest))
        (destination / "source/SOURCE_MANIFEST.json").write_text(
            json.dumps({"files": sources["files"]}, indent=2, sort_keys=True) + "\n"
        )
        (destination / "contact/PDB_CONTACT_DATASET_READY.json").write_text(
            json.dumps(
                {key: value for key, value in ready.items() if key in READY_FIELDS},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (destination / "ATTRIBUTION.md").write_text(ATTRIBUTION)


def prepare_evaluation(data_root: Path, archive: Path | None = None) -> dict[str, object]:
    output = data_root / "evaluation"
    if (output / "contact").exists() and (output / "source").exists():
        return verify_evaluation(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"incomplete evaluation data at {output}; use a fresh DATA_ROOT")
    # Provision before giving agents access; do not retain the transport archive in data/cache.
    with tempfile.TemporaryDirectory(prefix="contact-download-") as download:
        if archive is None:
            archive = Path(download) / "contact.tar.gz"
            print("Downloading frozen contact data and evaluator", flush=True)
            with urllib.request.urlopen(BUNDLE_URL, timeout=60) as response:
                with archive.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=8 << 20)
        data_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=data_root) as name:
            staged = Path(name) / "evaluation"
            extract_bundle(archive, staged)
            receipt = verify_evaluation(staged)
            (staged / "SETUP_VERIFIED.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if output.exists():
                output.rmdir()
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
