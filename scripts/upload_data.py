#!/usr/bin/env python3
"""Upload one independently verified full release to its Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from nano_protein.data import file_sha256
from nano_protein.sharded_data import DEFAULT_REPO_ID, validate_release_manifest


def validate_upload_root(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    verification_path = root / "RELEASE_VERIFIED.json"
    required = [
        manifest_path,
        verification_path,
        root / "README.md",
        root / "LICENSE_AND_ATTRIBUTION.md",
        root / "SOURCE_PROVENANCE.json",
        root / "RELEASE_METADATA_VERIFIED.json",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"release metadata is incomplete: {missing}")
    manifest = json.loads(manifest_path.read_text())
    validate_release_manifest(manifest)
    verification = json.loads(verification_path.read_text())
    if not (
        verification.get("status") == "verified"
        and verification.get("protocol") == "protein-corpus-parquet-verification-v1"
        and verification.get("manifest_sha256") == file_sha256(manifest_path)
    ):
        raise ValueError("release verification does not bind the final manifest")
    metadata = json.loads((root / "RELEASE_METADATA_VERIFIED.json").read_text())
    if not (
        metadata.get("status") == "verified"
        and metadata.get("protocol") == "protein-corpus-release-metadata-v1"
        and metadata.get("release_manifest_sha256") == file_sha256(manifest_path)
    ):
        raise ValueError("release metadata receipt does not bind the final manifest")
    for relative, expected in metadata.get("artifacts", {}).items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected.get("sha256"):
            raise ValueError(f"release metadata artifact changed: {relative}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--confirm-public-repo",
        help="must exactly equal --repo-id; prevents an accidental public upload",
    )
    args = parser.parse_args()
    validate_upload_root(args.release_root)
    if args.confirm_public_repo != args.repo_id:
        raise SystemExit(
            "public upload blocked: pass --confirm-public-repo with the exact dataset repo ID"
        )
    if os.environ.get("HF_XET_HIGH_PERFORMANCE") != "1":
        raise SystemExit("set HF_XET_HIGH_PERFORMANCE=1 for the large resumable upload")

    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_large_folder(
        folder_path=args.release_root,
        repo_id=args.repo_id,
        repo_type="dataset",
        num_workers=args.num_workers,
    )
    revision = api.dataset_info(args.repo_id).sha
    print(
        json.dumps({"event": "upload_complete", "repo_id": args.repo_id, "revision": revision})
    )


if __name__ == "__main__":
    main()
