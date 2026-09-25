"""Prefetching keeps the batch order, rewinds to the first unused batch and raises in order."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nanoprotein.data import BatchPrefetcher, MixtureBatcher, _StoreWriter
from nanoprotein.global_sampling import GlobalMixtureBatcher, row_state_exposure
from nanoprotein.tokenizer import ProteinTokenizer

WEIGHTS = {"uniref90": 0.36, "mgnify": 0.11, "omg_img": 0.54}


class _FailingBatcher:
    def __init__(self, fail_at):
        self.calls = 0
        self.fail_at = fail_at

    def batch(self):
        if self.calls == self.fail_at:
            raise RuntimeError("source exhausted")
        self.calls += 1
        return self.calls

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = state["calls"]


class BatchPrefetchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        # Small sources wrap their epochs within a few batches.
        for source, size in zip(WEIGHTS, (11, 13, 17), strict=True):
            writer = _StoreWriter(self.root / source / "train")
            for i in range(size):
                writer.append(
                    (np.arange(5 + 7 * i) + i) % 20 + 4,
                    hashlib.sha256(f"{source}/{i}".encode()).hexdigest(),
                )
            writer.finish()
        tokenizer = ProteinTokenizer.esmc()
        self.arguments = dict(batch_size=4, context_length=24, tokenizer=tokenizer)

    def batchers(self, kind):
        if kind == "rank":
            return [MixtureBatcher(self.root, "train", WEIGHTS, seed=5) for _ in range(2)]
        policies = dict.fromkeys(WEIGHTS, "allow")
        return [
            GlobalMixtureBatcher(
                self.root, "train", WEIGHTS, seed=5, rank=0, world_size=1, policies=policies
            )
            for _ in range(2)
        ]

    def assert_same_batches(self, feed, reference, count):
        for _ in range(count):
            expected = reference.batch(**self.arguments)
            for got, want in zip(feed.batch(), expected, strict=True):
                self.assertTrue(torch.equal(got, want))

    def test_prefetched_batches_match_direct_batches_and_rewind_exactly(self):
        for kind in ("rank", "global"):
            with self.subTest(kind=kind):
                batcher, reference = self.batchers(kind)
                feed = BatchPrefetcher(batcher, 3, **self.arguments)
                self.assert_same_batches(feed, reference, 7)
                self.assertEqual(feed.state, reference.state_dict())
                if kind == "global":
                    self.assertEqual(
                        {k: row_state_exposure(v) for k, v in feed.state["samplers"].items()},
                        reference.exposure(),
                    )
                feed.rewind()
                self.assertEqual(batcher.state_dict(), reference.state_dict())
                self.assert_same_batches(feed, reference, 5)

    def test_errors_arrive_after_the_batches_before_them(self):
        batcher = _FailingBatcher(fail_at=2)
        feed = BatchPrefetcher(batcher, 2)
        self.assertEqual([feed.batch(), feed.batch()], [1, 2])
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            feed.batch()
        self.assertEqual(feed.state, {"calls": 2})
        self.assertEqual(batcher.calls, 2)

    def test_depth_zero_reads_directly(self):
        batcher, reference = self.batchers("rank")
        feed = BatchPrefetcher(batcher, 0, **self.arguments)
        self.assert_same_batches(feed, reference, 3)
        self.assertEqual(feed.state, reference.state_dict())
        with self.assertRaises(ValueError):
            BatchPrefetcher(batcher, -1)


if __name__ == "__main__":
    unittest.main()
