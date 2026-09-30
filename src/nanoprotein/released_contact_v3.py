"""Install the verified portable expanded contact archive; retain the v2 source bundle."""
from __future__ import annotations

import sys
import tarfile
import tempfile
from pathlib import Path

from .data import file_sha256

BUNDLE_SHA256 = "7bcfd14c1a57d970ad84b1ac823f524c7e2fb27c7a2a1a30d303d12bddaf4647"
MANIFEST_SHA256 = "35c55cabf6547ef089defb7bb544d6037a84fc60915c17f47d9ed29c1c814977"


def extract_bundle(archive: Path, destination: Path) -> None:
    if file_sha256(archive) != BUNDLE_SHA256:
        raise ValueError("expanded contact archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as handle:
        names = set()
        for member in handle.getmembers():
            path = (destination / member.name).resolve()
            path.relative_to((destination / "contact-v3").resolve())
            if not (member.isfile() or member.isdir()) or member.name in names:
                raise ValueError(f"unsupported or duplicate archive member: {member.name}")
            names.add(member.name)
        handle.extractall(destination, filter="data")


def install_released_contact_pool(data_root: Path, archive: Path | None = None) -> None:
    evaluation = data_root / "evaluation"
    output = evaluation / "contact-v3"
    if output.exists():
        return  # prepare_profiles verifies installed inputs before reuse.
    if archive is None:
        raise FileNotFoundError("Fresh evaluation v3 setup needs --recovered-contact-pool PATH or --contact-v3-archive PATH; the portable v3 data archive is not yet publicly hosted")
    evaluation.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".contact-v3-", dir=evaluation) as temporary:
        stage = Path(temporary)
        extract_bundle(archive, stage)
        contact = stage / "contact-v3"
        if file_sha256(contact / "CONTACT_MANIFEST.jsonl") != MANIFEST_SHA256:
            raise ValueError("expanded contact manifest changed")
        sys.path.insert(0, str(evaluation / "source"))
        try:
            from .contact_dataset import ContactDataset
            dataset = ContactDataset(contact)
        finally:
            sys.path.pop(0)
        for row in dataset.payloads.values():
            path = (contact / row["path"]).resolve()
            path.relative_to(contact.resolve())
            if file_sha256(path) != row["sha256"]:
                raise ValueError(f"expanded contact payload changed: {row['chain_id']}")
        contact.rename(output)
