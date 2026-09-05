"""Prove complete-example preservation and gradient equivalence under DDP."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from nano_protein.batch_balance import balanced_partitions, rebalance_masked_batch
from nano_protein.train import sequence_mean_loss


def _examples():
    lengths = torch.tensor([9, 9, 8, 8, 2, 2, 1, 1])
    corrupted = torch.arange(80).reshape(8, 10) % 16
    corrupted[:, 0] = torch.arange(8)
    attention = torch.arange(10)[None] < lengths[:, None]
    labels = (corrupted + 1).remainder(16).masked_fill(~attention, -100)
    labels[:, 1::2] = -100
    return corrupted, labels, attention


def _model():
    torch.manual_seed(17)
    return torch.nn.Sequential(torch.nn.Embedding(16, 5), torch.nn.Linear(5, 16)).double()


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        full = _examples()
        reference = _model()
        expected_loss = sequence_mean_loss(reference(full[0]), full[1])
        expected_loss.backward()
        model = DDP(_model())
        local = tuple(tensor[rank * 4 : (rank + 1) * 4].clone() for tensor in full)
        rng = torch.get_rng_state()
        corrupted, labels, attention, statistics = rebalance_masked_batch(*local)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        assert corrupted.shape == labels.shape == attention.shape == (4, 10)
        for tensor in (corrupted, labels, attention):
            assert tensor.is_contiguous()
        assert attention.dtype == torch.bool
        for row, original_id in enumerate(corrupted[:, 0].tolist()):
            for observed, original in zip((corrupted, labels, attention), full, strict=True):
                torch.testing.assert_close(observed[row], original[original_id], rtol=0, atol=0)
        all_ids = [torch.empty(4, dtype=torch.long) for _ in range(2)]
        dist.all_gather(all_ids, corrupted[:, 0].contiguous())
        assert sorted(torch.cat(all_ids).tolist()) == list(range(8))
        assert statistics["rank_tokens_before"] == [34, 6]
        assert statistics["rank_tokens_after"] == [20, 20]
        loss = sequence_mean_loss(model(corrupted), labels)
        loss.backward()
        for actual, expected in zip(
            model.module.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual.grad, expected.grad)
    finally:
        dist.destroy_process_group()


class BatchBalanceTests(unittest.TestCase):
    def test_equal_counts_no_duplicates_and_nonincreasing_max_load(self):
        for lengths in (
            [9, 9, 8, 8, 2, 2, 1, 1],
            [0] * 16,
            [1] * 16,
            [512, 511, 64, 4, 8, 9, 20, 21],
        ):
            groups = balanced_partitions(lengths, 4)
            self.assertEqual(
                sorted(i for group in groups for i in group), list(range(len(lengths)))
            )
            count = len(lengths) // 4
            self.assertEqual([len(group) for group in groups], [count] * 4)
            before = max(sum(lengths[i * count : (i + 1) * count]) for i in range(4))
            after = max(sum(lengths[j] for j in group) for group in groups)
            self.assertLessEqual(after, before)
            self.assertEqual(groups, balanced_partitions(lengths, 4))

    def test_single_rank_is_identity(self):
        original = _examples()
        result = rebalance_masked_batch(*original)
        for expected, actual in zip(original, result[:3], strict=True):
            self.assertIs(actual, expected)
        self.assertEqual(result[3], {})

    def test_rejects_bad_partitions(self):
        for lengths, ranks in (([], 4), ([1, 2, 3], 2), ([1], 0), ([-1, 2], 2)):
            with self.assertRaises(ValueError):
                balanced_partitions(lengths, ranks)

    def test_real_ddp_preserves_examples_masks_rng_and_global_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(_worker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
