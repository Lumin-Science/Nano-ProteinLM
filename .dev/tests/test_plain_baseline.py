"""Baseline parameter count, optimizer, checkpoint and measurement contracts."""

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from nanoprotein.evaluate import load_checkpoint, parse_args
from nanoprotein.model import build_model, count_parameters
from nanoprotein.train import _git_state, build_optimizer, sequence_mean_loss

ROOT = Path(__file__).resolve().parents[2]


class PlainBaselineTests(unittest.TestCase):
    def test_full_model_count_and_recipe(self):
        config = yaml.safe_load((ROOT / "configs/autoresearch/esmc-171m.yaml").read_text())
        with torch.device("meta"):
            model = build_model(config["model"])
        self.assertEqual(count_parameters(model), 170671168)
        self.assertEqual(count_parameters(model), config["expected_parameter_count"])
        self.assertEqual(config["optimizer"], "adamw")
        self.assertEqual(config["learning_rate"], 0.0005)
        stage = config["stages"][0]
        self.assertEqual(4 * stage["micro_batch_size"] * stage["gradient_accumulation"], 256)

    def test_optimizer_update_and_checkpoint_roundtrip(self):
        torch.manual_seed(19)
        model = build_model("tiny", attention_backend="math")
        config = {"learning_rate": 0.0005, "weight_decay": 0.01}
        optimizer = build_optimizer(model, config)
        tokens = torch.tensor([[0, 4, 5, 6, 2], [0, 7, 8, 9, 2]])
        before = model.embedding.weight.detach().clone()
        loss = sequence_mean_loss(model(tokens)["logits"], tokens)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.embedding.weight))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "model_config": model.config.to_dict(),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                path,
            )
            restored, _ = load_checkpoint(path, torch.device("cpu"))
            restored_optimizer = build_optimizer(restored, config)
            restored_optimizer.load_state_dict(optimizer.state_dict())
            self.assertEqual(len(restored_optimizer.state), len(optimizer.state))
        torch.testing.assert_close(restored(tokens)["logits"], model.eval()(tokens)["logits"])

    def test_evaluation_defaults(self):
        args = parse_args(
            ["--checkpoint", "model.pt", "--data-root", "data", "--output-root", "output"]
        )
        self.assertFalse(hasattr(args, "validation_batches"))
        self.assertEqual(args.validation_context, 512)
        self.assertEqual(args.contact_chains, 20775)
        self.assertEqual(args.contact_bootstrap, 5000)

    def test_archive_runtime_does_not_inherit_parent_history(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            self.assertEqual(
                _git_state(Path(directory)), {"git_commit": None, "git_dirty": None}
            )


if __name__ == "__main__":
    unittest.main()
