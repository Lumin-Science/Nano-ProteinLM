"""Balance existing masked examples across ranks without changing the batch."""

from __future__ import annotations

import heapq
from collections.abc import Sequence

import torch
import torch.distributed as dist


def balanced_partitions(lengths: Sequence[int], world_size: int) -> list[list[int]]:
    """Assign long sequences first, with exactly the same row count per rank."""
    if world_size < 1 or not lengths or len(lengths) % world_size:
        raise ValueError("balanced batches require equal nonempty rank sizes")
    if any(length < 0 for length in lengths):
        raise ValueError("sequence lengths must be nonnegative")
    count = len(lengths) // world_size
    original = [list(range(rank * count, (rank + 1) * count)) for rank in range(world_size)]
    groups: list[list[int]] = [[] for _ in range(world_size)]
    heap = [(0, 0, rank) for rank in range(world_size)]
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        load, assigned, rank = heapq.heappop(heap)
        groups[rank].append(index)
        if assigned + 1 < count:
            heapq.heappush(heap, (load + lengths[index], assigned + 1, rank))
    if max(sum(lengths[i] for i in group) for group in groups) > max(
        sum(lengths[i] for i in group) for group in original
    ):
        return original
    return groups


@torch.no_grad()
def rebalance_masked_batch(
    corrupted: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Move complete examples, including their original corruption and labels.

    Length is a linear-work proxy for this short-context packed transformer.
    No sequence is split, dropped, resampled, or joined to another sequence.
    Equal row counts preserve the existing DDP sequence-mean loss weighting.
    """
    if not (corrupted.shape == labels.shape == attention_mask.shape) or corrupted.ndim != 2:
        raise ValueError("expected matching batch-by-context tensors")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size == 1:
        return corrupted, labels, attention_mask, {}
    batch, context = corrupted.shape
    packed = torch.stack((corrupted, labels, attention_mask.to(corrupted.dtype)), dim=1)
    gathered = torch.empty(
        (world_size * batch, 3, context), device=packed.device, dtype=packed.dtype
    )
    dist.all_gather_into_tensor(gathered, packed)
    lengths = gathered[:, 2].sum(1).cpu().tolist()
    partitions = balanced_partitions(lengths, world_size)
    indices = torch.tensor(partitions[dist.get_rank()], device=packed.device, dtype=torch.long)
    selected = gathered.index_select(0, indices)
    before = [sum(lengths[rank * batch : (rank + 1) * batch]) for rank in range(world_size)]
    after = [sum(lengths[index] for index in group) for group in partitions]
    statistics = {"rank_tokens_before": before, "rank_tokens_after": after}
    return (
        selected[:, 0].contiguous(),
        selected[:, 1].contiguous(),
        selected[:, 2].bool().contiguous(),
        statistics,
    )
