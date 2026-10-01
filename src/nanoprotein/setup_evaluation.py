"""Install the pinned contact scoring source, full contact population and fixed MLM masks."""

from __future__ import annotations

import argparse
import json
import shutil
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
SOURCE_MANIFEST_SHA256 = "295d82b3ad1ed95769659bd2ba52902cc8c260aaee06ae3c91c68c472c65e012"
RESEARCH_SOURCE_MANIFEST_SHA256 = (
    "2fad51df6f4f2586b6648b9e4f575058f1999eba60175cba9fd7834f36951e55"
)


def _inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def verify_evaluation(root: Path) -> dict[str, object]:
    source = root / "source"
    manifest = source / "SOURCE_MANIFEST.json"
    digest = file_sha256(manifest)
    if digest not in {SOURCE_MANIFEST_SHA256, RESEARCH_SOURCE_MANIFEST_SHA256}:
        raise ValueError(f"frozen evaluator checksum mismatch: {manifest}")
    sources = json.loads(manifest.read_text())
    for name, expected in sources["files"].items():
        if file_sha256(_inside(source, name)) != expected:
            raise ValueError(f"frozen evaluator source changed: {name}")
    return {"status": "verified", "source_manifest_sha256": digest}


def extract_bundle(archive: Path, destination: Path) -> None:
    if file_sha256(archive) != BUNDLE_SHA256:
        raise ValueError(f"contact archive checksum mismatch: {archive}")
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            _inside(destination, member.name)
            if not member.isfile() and not member.isdir():
                raise ValueError(f"unsupported archive member: {member.name}")
        members = [member for member in handle.getmembers()
                   if Path(member.name).parts and Path(member.name).parts[0] == "source"]
        handle.extractall(destination, members=members, filter="data")


def prepare_evaluation(data_root: Path, archive: Path | None = None) -> dict[str, object]:
    output = data_root / "evaluation"
    if (output / "source").exists():
        return verify_evaluation(output)
    if archive is None:
        cache = data_root / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / "contact-evaluation-v2.tar.gz"
        if not archive.exists() or file_sha256(archive) != BUNDLE_SHA256:
            temporary = archive.with_suffix(".gz.partial")
            print("Downloading frozen contact scoring source", flush=True)
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
        output.mkdir(parents=True, exist_ok=True)
        (staged / "source").rename(output / "source")
        (output / "SOURCE_VERIFIED.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--archive", type=Path, help="use a previously downloaded frozen bundle"
    )
    parser.add_argument("--contact-v3-archive", type=Path, help="use a downloaded v3 bundle offline")
    args = parser.parse_args()
    receipt = prepare_evaluation(args.data_root, args.archive)
    from .prepare_evaluation_profiles import prepare_profiles
    profiles = prepare_profiles(args.data_root, archive=args.contact_v3_archive)
    receipt = {"status": "verified", "protocol": profiles["protocol"], "prepared_caches": len(profiles["caches"])}
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
