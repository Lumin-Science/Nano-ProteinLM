"""Verified mmap materialization and sampling for protein training corpora."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .tokenizer import ProteinTokenizer

SOURCES = ("uniref90", "mgnify", "omg_img")
INDEX_DTYPE = np.dtype([("offset", "<u8"), ("length", "<u4"), ("digest", "S32")])


def file_sha256(path: Path, *, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


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


class _ShuffledRows:
    """Deterministic, rank-disjoint shuffled passes through one source store."""

    def __init__(
        self,
        size: int,
        *,
        seed: int,
        rank: int,
        world_size: int,
        allow_resampling: bool = True,
    ) -> None:
        if size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid shuffled-row contract")
        self.size = size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.allow_resampling = allow_resampling
        self.epoch = 0
        self.cursor = 0
        self.rows = self._epoch_rows()

    def _epoch_rows(self) -> np.ndarray:
        permutation = np.random.default_rng(self.seed + self.epoch).permutation(self.size)
        rows = permutation[self.rank :: self.world_size]
        if rows.size == 0:
            raise ValueError("source store is smaller than the distributed world size")
        return rows

    def next(self) -> int:
        if self.cursor == self.rows.size:
            if not self.allow_resampling:
                raise RuntimeError(
                    "training source exhausted its unique rank partition; "
                    "resampling is disabled. "
                    "Prepare more verified data before extending this run."
                )
            self.epoch += 1
            self.cursor = 0
            self.rows = self._epoch_rows()
        row = int(self.rows[self.cursor])
        self.cursor += 1
        return row


class MixtureBatcher:
    def __init__(
        self,
        root: Path,
        split: str,
        weights: Mapping[str, float],
        *,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        allow_resampling: bool = True,
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
        self.row_samplers = {
            name: _ShuffledRows(
                self.stores[name].index.size,
                seed=int(seed)
                + int.from_bytes(hashlib.sha256(name.encode("ascii")).digest()[:8], "big"),
                rank=rank,
                world_size=world_size,
                allow_resampling=allow_resampling,
            )
            for name in self.names
        }
        self.source_counts: Counter[str] = Counter()

    def state_dict(self) -> dict[str, object]:
        """Save sampling progress without storing the large shuffle permutations."""
        return {
            "names": self.names,
            "probabilities": self.probabilities.tolist(),
            "rng": self.rng.bit_generator.state,
            "source_counts": dict(self.source_counts),
            "samplers": {
                name: {
                    key: getattr(sampler, key)
                    for key in ("size", "seed", "rank", "world_size", "epoch", "cursor")
                }
                for name, sampler in self.row_samplers.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            tuple(state["names"]) != self.names
            or state["probabilities"] != self.probabilities.tolist()
        ):
            raise ValueError("checkpoint data mixture differs from the current mixture")
        for name, sampler in self.row_samplers.items():
            saved = state["samplers"][name]
            for key in ("size", "seed", "rank", "world_size"):
                if saved[key] != getattr(sampler, key):
                    raise ValueError(f"checkpoint sampler differs at {name}/{key}")
            sampler.epoch = int(saved["epoch"])
            if sampler.epoch and not sampler.allow_resampling:
                raise ValueError(
                    "checkpoint already repeated data; cannot resume as a no-repeat run"
                )
            sampler.rows = sampler._epoch_rows()
            sampler.cursor = int(saved["cursor"])
            if not 0 <= sampler.cursor <= sampler.rows.size:
                raise ValueError("checkpoint sampler cursor is out of bounds")
        self.rng.bit_generator.state = state["rng"]
        self.source_counts = Counter(state["source_counts"])

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
            record = store.sequence(self.row_samplers[source].next())
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
