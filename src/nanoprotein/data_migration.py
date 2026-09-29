"""Verify append-only corpus expansion and describe the legacy consumed rows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .data import SOURCES, TokenStore, file_sha256


def legacy_origins(packet):
    runtime = packet.get("runtime_states") or []
    world = int(packet["world_size"])
    if len(runtime) != world or len(packet["train_config"]["stages"]) != 1:
        raise ValueError("migration requires a complete single-stage sampler history")
    stage = packet["train_config"]["stages"][0]["name"]
    origins = {}
    for name in SOURCES:
        first = runtime[0]["batchers"][stage]["samplers"][name]
        cursors = []
        for rank, r in enumerate(runtime):
            batcher = r["batchers"][stage]
            sampler = batcher["samplers"][name]
            if (
                batcher.get("protocol") is not None
                or sampler["epoch"] != 0
                or sampler["rank"] != rank
                or sampler["world_size"] != world
                or sampler["size"] != first["size"]
                or sampler["seed"] != first["seed"]
                or sampler["cursor"] != batcher["source_counts"].get(name, 0)
                or not 0 <= sampler["cursor"] <= len(range(rank, sampler["size"], world))
            ):
                raise ValueError("migration requires verified, non-repeated legacy partitions")
            cursors.append(sampler["cursor"])
        origins[name] = dict(
            size=first["size"], seed=first["seed"], world_size=world, cursors=cursors
        )
    if sum(sum(o["cursors"]) for o in origins.values()) != packet["sequences_seen"]:
        raise ValueError("sampler history differs from checkpoint sequence count")
    return origins


def verify_prefix(old_root: Path, new_root: Path):
    """Compare all old encoded residues and full index entries with the new prefix."""
    old, new = TokenStore.open(old_root), TokenStore.open(new_root)
    if old.index.size > new.index.size or old.tokens.size > new.tokens.size:
        raise ValueError("expanded source is smaller than its parent")
    for i in range(0, old.index.size, 1_000_000):
        end = min(i + 1_000_000, old.index.size)
        if not np.array_equal(old.index[i:end], new.index[i:end]):
            raise ValueError("expanded source changes an existing record identity or extent")
    for i in range(0, old.tokens.size, 8 << 20):
        end = min(i + (8 << 20), old.tokens.size)
        if not np.array_equal(old.tokens[i:end], new.tokens[i:end]):
            raise ValueError("expanded source changes existing encoded residues")
    return {
        "old_records": int(old.index.size),
        "new_records": int(new.index.size),
        "existing_indices_and_tokens_identical": True,
    }


def create_migration(checkpoint: Path, old_root: Path, new_root: Path, *, seed: int):
    from .sharded_data import validate_prepared_plan
    from .train import validate_data_manifest

    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    old, new = validate_data_manifest(old_root), validate_data_manifest(new_root)
    if packet["data_manifest_sha256"] != file_sha256(old_root / "manifest.json"):
        raise ValueError("parent checkpoint does not bind the old corpus")
    if old["release_manifest_sha256"] != new["release_manifest_sha256"]:
        raise ValueError("corpus expansion must use the same verified release")
    for root in (old_root, new_root):
        validate_prepared_plan(json.loads((root / "download-plan.json").read_text()), root)
    origins = legacy_origins(packet)
    proof = {}
    for name in SOURCES:
        if old["sources"][name]["validation"] != new["sources"][name]["validation"]:
            raise ValueError("migration cannot alter validation")
        proof[name] = verify_prefix(old_root / name / "train", new_root / name / "train")
        if proof[name]["old_records"] != origins[name]["size"]:
            raise ValueError("sampler size differs from old corpus")
    return {
        "protocol": "append-only-global-sampler-migration-v1",
        "status": "passed",
        "parent_checkpoint_sha256": file_sha256(checkpoint),
        "parent_step": packet["optimizer_step"],
        "parent_sequences_seen": packet["sequences_seen"],
        "old_manifest_sha256": file_sha256(old_root / "manifest.json"),
        "new_manifest_sha256": file_sha256(new_root / "manifest.json"),
        "release_manifest_sha256": new["release_manifest_sha256"],
        "global_data_seed": seed,
        "origins": origins,
        "prefix_verification": proof,
        "validation_unchanged": True,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_migration(packet, migration, new_manifest_sha256):
    if (
        migration.get("protocol") != "append-only-global-sampler-migration-v1"
        or migration.get("status") != "passed"
        or migration.get("old_manifest_sha256") != packet["data_manifest_sha256"]
        or migration.get("new_manifest_sha256") != new_manifest_sha256
        or migration.get("parent_step") != packet["optimizer_step"]
        or migration.get("parent_sequences_seen") != packet["sequences_seen"]
        or migration.get("origins") != legacy_origins(packet)
        or migration.get("validation_unchanged") is not True
    ):
        raise ValueError("migration does not bind the parent checkpoint and expanded corpus")
    for name in SOURCES:
        proof = migration["prefix_verification"][name]
        if (
            not proof["existing_indices_and_tokens_identical"]
            or proof["old_records"] != migration["origins"][name]["size"]
            or proof["new_records"] < proof["old_records"]
        ):
            raise ValueError("invalid append-only corpus proof")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--old-data-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = create_migration(
        args.checkpoint, args.old_data_root, args.data_root, seed=args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
