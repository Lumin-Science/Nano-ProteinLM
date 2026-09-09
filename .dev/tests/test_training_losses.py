"""Check loss semantics, including unequal target counts under real CPU DDP."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from nanoprotein.train import sequence_mean_loss, training_losses


def _inputs():
    inputs = torch.arange(36, dtype=torch.float64).reshape(4, 3, 3) / 36
    labels = torch.tensor([[0, -100, -100], [-100, -100, -100], [1, 2, 3], [0, 1, -100]])
    return inputs, labels


def _model():
    model = torch.nn.Linear(3, 4, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.arange(12, dtype=torch.float64).reshape(4, 3) / 12)
    return model


def _pooled_loss(logits, labels):
    # Independent explicit per-sequence reference (no implementation helper).
    total = logits.sum() * 0
    denominator = 0.0
    for row in range(labels.shape[0]):
        chosen = labels[row] != -100
        count = int(chosen.sum())
        if not count:
            continue
        weight = count**0.5
        log_p = logits[row, chosen].log_softmax(-1)
        row_loss = -log_p.gather(1, labels[row, chosen, None]).mean()
        total = total + weight * row_loss
        denominator += weight
    return total / max(denominator, 1)


def _distributed_worker(rank, rendezvous, empty_rank):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        inputs, labels = _inputs()
        if empty_rank:
            labels[:2] = -100
        reference = _model()
        expected = _pooled_loss(reference(inputs), labels)
        expected.backward()
        model = DDP(_model())
        loss, _, logged_objective = training_losses(
            model(inputs[rank * 2 : rank * 2 + 2]),
            labels[rank * 2 : rank * 2 + 2],
            reduction="sqrt_mask_count",
        )
        loss.backward()
        torch.testing.assert_close(model.module.weight.grad, reference.weight.grad)
        torch.testing.assert_close(logged_objective, expected.detach())
    finally:
        dist.destroy_process_group()


class TrainingLossTests(unittest.TestCase):
    def test_default_preserves_sequence_mean_value_and_gradient(self):
        inputs, labels = _inputs()
        logits = _model()(inputs).detach().requires_grad_()
        expected = sequence_mean_loss(logits, labels)
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        loss, diagnostic, objective = training_losses(logits, labels)
        torch.testing.assert_close(loss, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], expected_gradient)
        torch.testing.assert_close(diagnostic, expected.detach())
        torch.testing.assert_close(objective, expected.detach())

    def test_sqrt_weighting_matches_global_reference_but_keeps_sequence_diagnostic(self):
        inputs, labels = _inputs()
        logits = _model()(inputs).detach().requires_grad_()
        expected = _pooled_loss(logits, labels)
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        loss, diagnostic, objective = training_losses(
            logits, labels, reduction="sqrt_mask_count"
        )
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], expected_gradient)
        torch.testing.assert_close(objective, expected.detach())
        torch.testing.assert_close(diagnostic, sequence_mean_loss(logits, labels).detach())
        self.assertFalse(torch.isclose(loss, diagnostic))

    def test_no_targets_has_finite_zero_loss_and_gradient(self):
        logits = torch.randn(2, 3, 4, requires_grad=True)
        labels = torch.full((2, 3), -100)
        loss, diagnostic, objective = training_losses(
            logits, labels, reduction="sqrt_mask_count"
        )
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(diagnostic.item(), 0)
        self.assertEqual(objective.item(), 0)
        self.assertEqual(logits.grad.count_nonzero(), 0)

    def test_rejects_unknown_reduction(self):
        with self.assertRaises(ValueError):
            training_losses(
                torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.long), reduction="typo"
            )

    def test_ddp_matches_global_weighted_mean_with_unequal_and_empty_ranks(self):
        for empty_rank in (False, True):
            with (
                self.subTest(empty_rank=empty_rank),
                tempfile.TemporaryDirectory() as directory,
            ):
                rendezvous = (Path(directory) / "rendezvous").as_uri()
                mp.spawn(
                    _distributed_worker, args=(rendezvous, empty_rank), nprocs=2, join=True
                )


if __name__ == "__main__":
    unittest.main()
