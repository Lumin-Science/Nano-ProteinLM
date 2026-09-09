import json
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanoprotein.periodic_evaluation import run_periodic_evaluation


def _worker(rank, directory, fail):
    root = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=2),
    )
    os.environ["RANK"] = str(rank)
    try:
        rng = torch.get_rng_state()
        command = [
            sys.executable,
            "-c",
            "import os,sys,time; assert 'RANK' not in os.environ; "
            "time.sleep(3); sys.exit(" + str(7 if fail else 0) + ")",
        ]
        try:
            elapsed = run_periodic_evaluation(
                command,
                checkpoint=root / "checkpoint.pt",
                output_root=root,
                optimizer_step=10,
                device=torch.device("cpu"),
                poll_seconds=0.1,
            )
            assert not fail and elapsed >= 3
        except RuntimeError as error:
            assert fail and "code 7" in str(error)
        assert torch.equal(rng, torch.get_rng_state())
        value = torch.tensor(rank + 1)
        dist.all_reduce(value)
        assert value.item() == 3
    finally:
        dist.destroy_process_group()


class PeriodicEvaluationTests(unittest.TestCase):
    def test_all_ranks_wait_beyond_collective_timeout_and_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(directory, False), nprocs=2, join=True)
            receipt = json.loads((Path(directory) / "PERIODIC_EVALUATION.json").read_text())
            self.assertEqual(receipt["status"], "passed")

    def test_evaluator_failure_is_propagated_to_every_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(directory, True), nprocs=2, join=True)
            receipt = json.loads((Path(directory) / "PERIODIC_EVALUATION.json").read_text())
            self.assertEqual(receipt["status"], "failed")


if __name__ == "__main__":
    unittest.main()
