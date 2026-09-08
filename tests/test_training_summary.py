import json
import statistics
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.summarize_training_runs import summarize


class TrainingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.runs = [Path(self.directory.name) / f"seed-{seed}" for seed in (42, 43)]
        for seed, loss, root in zip((42, 43), (2.4, 2.6), self.runs, strict=True):
            (root / "evaluation").mkdir(parents=True)
            (root / "config.yaml").write_text(
                yaml.safe_dump({"seed": seed, "max_model_tokens": 100})
            )
            self.write(
                root / "TRAINING_COMPLETE.json",
                {
                    "stop_reason": "max_model_tokens",
                    "target_model_tokens": 100,
                    "model_token_budget_reached": True,
                    "model_tokens": 110,
                    "model_token_overrun": 10,
                    "last_optimizer_step_model_tokens": 20,
                    "final_checkpoint": {"sha256": f"checkpoint-{seed}"},
                },
            )
            self.write(
                root / "evaluation/EVALUATION.json",
                {
                    "checkpoint_sha256": f"checkpoint-{seed}",
                    "validation_mlm": {"sequence_mean_nll": loss, "sequences": 4096},
                    "contact": {
                        "evaluation_chains": 20775,
                        "precision_at_l": 0.32,
                        "precision_at_l_uncertainty": {"low": 0.31, "high": 0.33},
                    },
                },
            )

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def test_reports_mean_sd_and_keeps_chain_ci_separate(self):
        report = summarize(self.runs, 4096)
        self.assertEqual(report["metrics"]["validation_loss"]["mean"], 2.5)
        self.assertAlmostEqual(
            report["metrics"]["validation_loss"]["sample_sd"], statistics.stdev((2.4, 2.6))
        )
        self.assertEqual(report["runs"][0]["p_at_l_95_ci"], {"low": 0.31, "high": 0.33})
        self.assertNotIn("accepted", report)

    def test_rejects_missing_repeat_or_incomplete_eval(self):
        with self.assertRaises(ValueError):
            summarize(self.runs[:1], 4096)
        with self.assertRaises(ValueError):
            summarize(self.runs, 32)
        (self.runs[1] / "evaluation/EVALUATION.json").unlink()
        with self.assertRaises(FileNotFoundError):
            summarize(self.runs, 4096)

    def test_rejects_early_token_stop_even_with_evaluation(self):
        path = self.runs[1] / "TRAINING_COMPLETE.json"
        value = json.loads(path.read_text())
        value["stop_reason"] = "walltime"
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, "token budget"):
            summarize(self.runs, 4096)

    def test_rejects_wrong_checkpoint_and_nonfinite_score(self):
        path = self.runs[1] / "evaluation/EVALUATION.json"
        value = json.loads(path.read_text())
        value["checkpoint_sha256"] = "another-checkpoint"
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            summarize(self.runs, 4096)
        value["checkpoint_sha256"] = "checkpoint-43"
        value["validation_mlm"]["sequence_mean_nll"] = float("nan")
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            summarize(self.runs, 4096)

    def test_rejects_recipe_changes_between_seeds(self):
        path = self.runs[1] / "config.yaml"
        config = yaml.safe_load(path.read_text())
        config["learning_rate"] = 1e-3
        path.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "same effective recipe"):
            summarize(self.runs, 4096)


if __name__ == "__main__":
    unittest.main()
