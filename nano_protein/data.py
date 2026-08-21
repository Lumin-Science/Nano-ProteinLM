"""Streaming preparation and mmap sampling for cluster-representative FASTAs."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .tokenizer import ProteinTokenizer

SOURCES = ("uniref90", "mgnify", "omg_img")
INDEX_DTYPE = np.dtype([("offset", "<u8"), ("length", "<u4"), ("digest", "S32")])


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def file_sha256(path: Path, *, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    parts: list[str] = []
    with path.open("rt", encoding="ascii", errors="strict") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header = line[1:].strip().split()[0]
                parts = []
            elif header is None:
                raise ValueError(f"sequence before header at {path}:{line_number}")
            else:
                parts.append(line.upper())
    if header is not None:
        yield header, "".join(parts)


def _header_digest(header: str) -> str:
    value = header.removeprefix("sha256_")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"representative FASTA header is not a SHA-256 ID: {header!r}")
    return value


def evaluation_digests(paths: Sequence[Path]) -> set[str]:
    digests: set[str] = set()
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                value = record.get("sha256", record.get("sequence_sha256"))
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"missing sequence digest at {path}:{line_number}")
                digests.add(value)
    return digests


class _StoreWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.token_partial = root / "tokens.bin.partial"
        self.index_partial = root / "index.npy.partial"
        self.handle = self.token_partial.open("wb")
        self.rows: list[tuple[int, int, bytes]] = []
        self.offset = 0

    def append(self, tokens: np.ndarray, digest: str) -> None:
        values = np.asarray(tokens, dtype=np.uint8)
        self.handle.write(values.tobytes(order="C"))
        self.rows.append((self.offset, values.size, bytes.fromhex(digest)))
        self.offset += values.size

    def finish(self) -> dict[str, object]:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        index = np.asarray(self.rows, dtype=INDEX_DTYPE)
        with self.index_partial.open("wb") as handle:
            np.save(handle, index, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        token_path = self.root / "tokens.bin"
        index_path = self.root / "index.npy"
        self.token_partial.replace(token_path)
        self.index_partial.replace(index_path)
        return {
            "records": int(index.size),
            "residues": int(self.offset),
            "tokens_sha256": file_sha256(token_path),
            "index_sha256": file_sha256(index_path),
        }


def prepare_dataset(
    *,
    cluster_root: Path,
    output_root: Path,
    pcore_index: Path,
    contact_manifest: Path,
    train_per_source: int,
    validation_per_source: int,
    validation_modulus: int = 32,
    validation_bucket: int = 0,
    minimum_length: int = 32,
    maximum_length: int = 16_384,
    verify_sequence_hashes: bool = True,
) -> dict[str, object]:
    """Create an order-independent train/validation split with exact eval exclusion.

    The Step-9 representative files contain one sequence per 70%-identity
    cluster and are ordered by SHA-256. A prefix is therefore a deterministic
    pseudorandom representative sample with respect to sequence content.
    """

    if train_per_source <= 0 or validation_per_source <= 0:
        raise ValueError("train and validation targets must be positive")
    if not 0 <= validation_bucket < validation_modulus:
        raise ValueError("invalid validation hash bucket")
    excluded = evaluation_digests([pcore_index, contact_manifest])
    tokenizer = ProteinTokenizer.esmc()
    output_root.mkdir(parents=True, exist_ok=True)
    source_receipts: dict[str, object] = {}
    for source in SOURCES:
        source_root = cluster_root / source
        fasta = source_root / "representatives.fasta"
        verification_path = source_root / "verification.json"
        verification = json.loads(verification_path.read_text())
        writers = {
            "train": _StoreWriter(output_root / source / "train"),
            "validation": _StoreWriter(output_root / source / "validation"),
        }
        counts: Counter[str] = Counter()
        rejected: Counter[str] = Counter()
        scanned = 0
        for header, sequence in fasta_records(fasta):
            scanned += 1
            digest = _header_digest(header)
            sequence_digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            if verify_sequence_hashes and sequence_digest != digest:
                raise ValueError(f"sequence/header SHA mismatch in {fasta}: {header}")
            if digest in excluded:
                rejected["evaluation_exact_match"] += 1
                continue
            if not minimum_length <= len(sequence) <= maximum_length:
                rejected["length"] += 1
                continue
            split = (
                "validation"
                if int(digest[:16], 16) % validation_modulus == validation_bucket
                else "train"
            )
            target = validation_per_source if split == "validation" else train_per_source
            if counts[split] >= target:
                continue
            writers[split].append(tokenizer.encode_residues(sequence), digest)
            counts[split] += 1
            if (
                counts["train"] >= train_per_source
                and counts["validation"] >= validation_per_source
            ):
                break
        if counts["train"] != train_per_source or counts["validation"] != validation_per_source:
            raise RuntimeError(
                f"{source} exhausted before targets: train={counts['train']}, "
                f"validation={counts['validation']}"
            )
        source_receipts[source] = {
            "fasta": str(fasta.resolve()),
            "upstream_verification": verification,
            "scanned_records": scanned,
            "rejected": dict(rejected),
            "train": writers["train"].finish(),
            "validation": writers["validation"].finish(),
        }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "sampling_unit": "one_transferred_representative_per_70pct_identity_cluster",
        "member_within_cluster_sampling_available": False,
        "split": {
            "method": "sha256_prefix_modulus",
            "validation_modulus": validation_modulus,
            "validation_bucket": validation_bucket,
        },
        "decontamination": {
            "method": "exact_normalized_sequence_sha256",
            "excluded_digest_count": len(excluded),
            "pcore_index": str(pcore_index.resolve()),
            "pcore_index_sha256": file_sha256(pcore_index),
            "contact_manifest": str(contact_manifest.resolve()),
            "contact_manifest_sha256": file_sha256(contact_manifest),
            "homology_exclusion": False,
        },
        "filters": {"minimum_length": minimum_length, "maximum_length": maximum_length},
        "sources": source_receipts,
    }
    manifest_path = output_root / "manifest.json"
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(_canonical_json(manifest))
    temporary.replace(manifest_path)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


@dataclass(frozen=True)
class TokenStore:
    tokens: np.memmap
    index: np.ndarray

    @classmethod
    def open(cls, root: Path) -> TokenStore:
        index = np.load(root / "index.npy", mmap_mode="r", allow_pickle=False)
        tokens = np.memmap(root / "tokens.bin", dtype=np.uint8, mode="r")
        if index.dtype != INDEX_DTYPE:
            raise ValueError(f"unexpected mmap index dtype at {root}: {index.dtype}")
        if index.size == 0:
            raise ValueError(f"empty token store: {root}")
        last = index[-1]
        if int(last["offset"]) + int(last["length"]) != tokens.size:
            raise ValueError(f"token/index extent mismatch: {root}")
        return cls(tokens=tokens, index=index)

    def sequence(self, row: int) -> np.ndarray:
        record = self.index[int(row)]
        start = int(record["offset"])
        return self.tokens[start : start + int(record["length"])]


class MixtureBatcher:
    def __init__(
        self,
        root: Path,
        split: str,
        weights: Mapping[str, float],
        *,
        seed: int,
        rank: int = 0,
    ) -> None:
        missing = set(weights) - set(SOURCES)
        if missing:
            raise ValueError(f"unknown mixture sources: {sorted(missing)}")
        self.names = tuple(sorted(weights))
        probabilities = np.asarray([weights[name] for name in self.names], dtype=np.float64)
        if np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("mixture weights must be nonnegative with positive sum")
        self.probabilities = probabilities / probabilities.sum()
        self.stores = {name: TokenStore.open(root / name / split) for name in self.names}
        self.rng = np.random.default_rng(int(seed) + 1_000_003 * int(rank))
        self.source_counts: Counter[str] = Counter()

    def batch(
        self,
        batch_size: int,
        *,
        context_length: int,
        tokenizer: ProteinTokenizer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0 or context_length < 4:
            raise ValueError("invalid batch or context length")
        tokens = np.full((batch_size, context_length), tokenizer.pad_id, dtype=np.int64)
        mask = np.zeros((batch_size, context_length), dtype=np.bool_)
        residue_limit = context_length - 2
        source_rows = self.rng.choice(len(self.names), size=batch_size, p=self.probabilities)
        for row, source_row in enumerate(source_rows):
            source = self.names[int(source_row)]
            store = self.stores[source]
            record = store.sequence(int(self.rng.integers(store.index.size)))
            maximum_offset = max(0, record.size - residue_limit)
            offset = int(self.rng.integers(maximum_offset + 1)) if maximum_offset else 0
            residues = record[offset : offset + residue_limit]
            stop = 1 + residues.size
            tokens[row, 0] = tokenizer.bos_id
            tokens[row, 1:stop] = residues
            tokens[row, stop] = tokenizer.eos_id
            mask[row, : stop + 1] = True
            self.source_counts[source] += 1
        return torch.from_numpy(tokens), torch.from_numpy(mask)
