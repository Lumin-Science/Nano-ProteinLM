"""Globally ordered source epochs with portable, compact consumption history."""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import TRAINING_SOURCES, TokenStore


def origin_draws(origin):
    if origin is None:
        return 0
    if origin.get("protocol") == "global-unseen-origin-v1":
        return row_state_exposure(origin["state"])["draws"]
    return sum(origin["cursors"])


def row_state_exposure(state):
    """Read exposure without constructing a potentially very large permutation."""
    size, epoch, cursor = int(state["size"]), int(state["epoch"]), int(state["cursor"])
    prior = origin_draws(state.get("origin"))
    if size <= 0 or epoch < 0 or cursor < 0 or cursor > size - (prior if epoch == 0 else 0):
        raise ValueError("invalid global source history")
    if epoch and not state["allow_resampling"]:
        raise ValueError("strict source has repeated an epoch")
    draws = prior + cursor if epoch == 0 else epoch * size + cursor
    return dict(
        draws=draws,
        unique_records_seen=min(draws, size),
        repeated_draws=max(0, draws - size),
        epoch=epoch,
        epoch_cursor=cursor,
        available_records=size,
    )


class GlobalRows:
    """Every rank advances the same source cursor before selecting its local rows.

    A legacy origin reconstructs records already consumed by disjoint rank
    partitions. Only their complement, including appended records, enters the
    first queue. Subsequent epochs cover every source record exactly once.
    """

    def __init__(self, size: int, *, seed: int, allow_resampling: bool, origin=None):
        if size <= 0:
            raise ValueError("source must contain records")
        self.size = int(size)
        self.seed = int(seed)
        self.allow_resampling = bool(allow_resampling)
        self.origin = copy.deepcopy(origin)
        self.consumed_before = 0
        if origin is not None and origin.get("protocol") == "global-unseen-origin-v1":
            previous = origin["state"]
            exposure = row_state_exposure(previous)
            if (
                previous["size"] >= self.size
                or previous["epoch"] != 0
                or exposure["repeated_draws"]
            ):
                raise ValueError("global expansion requires an un-repeated source prefix")
            self.consumed_before = exposure["draws"]
        elif origin is not None:
            old_size = int(origin["size"])
            world = int(origin["world_size"])
            cursors = origin["cursors"]
            if not 0 < old_size <= size or world < 1 or len(cursors) != world:
                raise ValueError("invalid legacy source origin")
            for rank, cursor in enumerate(cursors):
                if not isinstance(cursor, int) or not 0 <= cursor <= len(
                    range(rank, old_size, world)
                ):
                    raise ValueError("invalid legacy rank cursor")
            self.consumed_before = sum(cursors)
        self.epoch = 0
        self.cursor = 0
        self.rows = self._rows()

    def _rows(self):
        if self.epoch == 0 and self.origin:
            o = self.origin
            if o.get("protocol") == "global-unseen-origin-v1":
                state = o["state"]
                previous = GlobalRows(
                    state["size"],
                    seed=state["seed"],
                    allow_resampling=state["allow_resampling"],
                    origin=state["origin"],
                )
                previous.load_state_dict(state)
                rows = np.concatenate(
                    (
                        previous.rows[previous.cursor :],
                        np.arange(state["size"], self.size, dtype=np.int64),
                    )
                )
                np.random.default_rng(self.seed).shuffle(rows)
                return rows
            old = np.random.default_rng(o["seed"]).permutation(o["size"])
            unseen = np.ones(self.size, dtype=np.bool_)
            for rank, cursor in enumerate(o["cursors"]):
                unseen[old[rank :: o["world_size"]][:cursor]] = False
            rows = np.flatnonzero(unseen)
            np.random.default_rng(self.seed).shuffle(rows)
            return rows
        return np.random.default_rng(self.seed + self.epoch).permutation(self.size)

    def take(self, count: int) -> np.ndarray:
        if count < 0:
            raise ValueError("negative source draw count")
        if not self.allow_resampling and count > len(self.rows) - self.cursor:
            raise RuntimeError("training source exhausted; resampling is disabled")
        result = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            if self.cursor == len(self.rows):
                self.epoch += 1
                self.cursor = 0
                self.rows = self._rows()
            length = min(count - filled, len(self.rows) - self.cursor)
            result[filled : filled + length] = self.rows[self.cursor : self.cursor + length]
            self.cursor += length
            filled += length
        return result

    def exposure(self) -> dict[str, int]:
        draws = (
            self.consumed_before + self.cursor
            if self.epoch == 0
            else self.epoch * self.size + self.cursor
        )
        return {
            "draws": draws,
            "unique_records_seen": min(draws, self.size),
            "repeated_draws": max(0, draws - self.size),
            "epoch": self.epoch,
            "epoch_cursor": self.cursor,
            "available_records": self.size,
        }

    def state_dict(self):
        return {
            "size": self.size,
            "seed": self.seed,
            "origin": copy.deepcopy(self.origin),
            "allow_resampling": self.allow_resampling,
            "epoch": self.epoch,
            "cursor": self.cursor,
        }

    def load_state_dict(self, state):
        for key in ("size", "seed", "origin", "allow_resampling"):
            if state[key] != getattr(self, key):
                raise ValueError(f"global source contract differs at {key}")
        epoch, cursor = int(state["epoch"]), int(state["cursor"])
        if epoch < 0 or (epoch and not self.allow_resampling):
            raise ValueError("invalid global source epoch")
        if epoch != self.epoch:
            self.epoch = epoch
            self.rows = self._rows()
        if not 0 <= cursor <= len(self.rows):
            raise ValueError("invalid global source cursor")
        self.cursor = cursor


class GlobalMixtureBatcher:
    """Source selection/record order are independent of the number of GPUs.

    Every rank samples the same global microbatch, then owns a contiguous slice.
    Crop RNG remains rank-local. Checkpoints keep global source RNG/cursors, so
    changing GPU count preserves the exact next record identities and their
    grouping when the global microbatch size stays fixed.
    """

    protocol = "global-mixture-v1"

    def __init__(
        self,
        root: Path,
        split: str,
        weights,
        *,
        seed: int,
        rank: int,
        world_size: int,
        policies,
        origins=None,
        migration=None,
    ):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid global sampler rank")
        if not weights or set(weights) - set(TRAINING_SOURCES) or set(policies) != set(weights):
            raise ValueError("global sampler sources/policies differ")
        if any(v not in ("allow", "error") for v in policies.values()):
            raise ValueError("invalid source resampling policy")
        self.names = tuple(sorted(weights))
        self.probabilities = np.array([weights[k] for k in self.names], dtype=np.float64)
        if not np.isfinite(self.probabilities).all() or np.any(self.probabilities <= 0):
            raise ValueError("invalid source weights")
        self.probabilities /= self.probabilities.sum()
        self.seed, self.rank, self.world_size = int(seed), rank, world_size
        self.stores = {k: TokenStore.open(root / k / split) for k in self.names}
        self.migration = copy.deepcopy(migration)
        self.rng = np.random.default_rng(self.seed + 1_000_003 * rank)
        self.global_rng = np.random.default_rng(self.seed)
        self.row_samplers = {
            name: GlobalRows(
                self.stores[name].index.size,
                seed=self.seed
                + int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big"),
                allow_resampling=policies[name] == "allow",
                origin=(origins or {}).get(name),
            )
            for name in self.names
        }
        self.source_counts = Counter()
        self._partition_accounting()

    def _partition_accounting(self):
        # Historical counts are accounting shares, not claims about physical GPUs.
        self.source_counts = Counter(
            {
                name: sampler.exposure()["draws"] // self.world_size
                + (self.rank < sampler.exposure()["draws"] % self.world_size)
                for name, sampler in self.row_samplers.items()
            }
        )

    def exposure(self):
        return {name: sampler.exposure() for name, sampler in self.row_samplers.items()}

    def state_dict(self):
        return {
            "protocol": self.protocol,
            "names": self.names,
            "probabilities": self.probabilities.tolist(),
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "global_rng": copy.deepcopy(self.global_rng.bit_generator.state),
            "source_counts": dict(self.source_counts),
            "samplers": {k: v.state_dict() for k, v in self.row_samplers.items()},
            "migration": copy.deepcopy(self.migration),
        }

    def load_state_dict(self, state):
        if (
            state.get("protocol") != self.protocol
            or tuple(state["names"]) != self.names
            or state["probabilities"] != self.probabilities.tolist()
            or state["seed"] != self.seed
            or state["migration"] != self.migration
        ):
            raise ValueError("global batcher contract differs")
        for name in self.names:
            self.row_samplers[name].load_state_dict(state["samplers"][name])
        self.global_rng.bit_generator.state = state["global_rng"]
        if state["world_size"] == self.world_size and state["rank"] == self.rank:
            self.rng.bit_generator.state = state["rng"]
            self.source_counts = Counter(state["source_counts"])
        else:
            draws = sum(v["draws"] for v in self.exposure().values())
            self.rng = np.random.default_rng(self.seed + draws + 1_000_003 * self.rank)
            self._partition_accounting()

    def select_rows(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        size = batch_size * self.world_size
        sources = self.global_rng.choice(len(self.names), size=size, p=self.probabilities)
        rows = np.empty(size, dtype=np.int64)
        # Check all strict sources before advancing any source cursor.
        for i, name in enumerate(self.names):
            sampler = self.row_samplers[name]
            if not sampler.allow_resampling and np.count_nonzero(sources == i) > (
                len(sampler.rows) - sampler.cursor
            ):
                raise RuntimeError(f"{name} exhausted; resampling is disabled")
        for i, name in enumerate(self.names):
            positions = sources == i
            rows[positions] = self.row_samplers[name].take(int(positions.sum()))
        start = self.rank * batch_size
        selected = [
            (self.names[int(s)], int(r))
            for s, r in zip(
                sources[start : start + batch_size],
                rows[start : start + batch_size],
                strict=True,
            )
        ]
        self.source_counts.update(name for name, _ in selected)
        return selected

    def batch(self, batch_size: int, *, context_length: int, tokenizer):
        if context_length < 4:
            raise ValueError("context length must be at least four")
        tokens = np.full((batch_size, context_length), tokenizer.pad_id, dtype=np.int64)
        mask = np.zeros((batch_size, context_length), dtype=np.bool_)
        for row, (source, index) in enumerate(self.select_rows(batch_size)):
            record = self.stores[source].sequence(index)
            maximum_offset = max(0, record.size - (context_length - 2))
            offset = int(self.rng.integers(maximum_offset + 1)) if maximum_offset else 0
            residues = record[offset : offset + context_length - 2]
            stop = 1 + residues.size
            tokens[row, 0] = tokenizer.bos_id
            tokens[row, 1:stop] = residues
            tokens[row, stop] = tokenizer.eos_id
            mask[row, : stop + 1] = True
        return torch.from_numpy(tokens), torch.from_numpy(mask)


def portable_batcher_states(packet: dict[str, Any]):
    """Check globally replicated sampler state before repartitioning a checkpoint."""
    runtime = packet.get("runtime_states") or []
    if len(runtime) != int(packet["world_size"]):
        raise ValueError("checkpoint is missing rank runtime states")
    first = runtime[0]["batchers"]
    for state in first.values():
        if state.get("protocol") != GlobalMixtureBatcher.protocol:
            raise ValueError("checkpoint does not use portable global source epochs")
    for rank, rank_state in enumerate(runtime):
        other = rank_state["batchers"]
        if other.keys() != first.keys():
            raise ValueError("checkpoint rank stages differ")
        for stage in first:
            if other[stage]["rank"] != rank or other[stage]["world_size"] != len(runtime):
                raise ValueError("checkpoint sampler rank metadata differs")
            for key in ("samplers", "global_rng", "probabilities", "seed", "migration"):
                if other[stage][key] != first[stage][key]:
                    raise ValueError("checkpoint global sampler states disagree across ranks")
    for stage, state in first.items():
        for source, sampler in state["samplers"].items():
            draws = row_state_exposure(sampler)["draws"]
            counted = sum(r["batchers"][stage]["source_counts"][source] for r in runtime)
            if counted != draws:
                raise ValueError("checkpoint source counts disagree with global cursor")
    return first
