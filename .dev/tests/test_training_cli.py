import copy
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

from nanoprotein.train import resolve_config_overrides

ROOT = Path(__file__).resolve().parents[2]


class TrainingCLITests(unittest.TestCase):
    def print_config(self, recipe, *flags):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nanoprotein.train",
                "--config",
                str(recipe),
                "--print-config",
                *flags,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return yaml.safe_load(result.stdout)

    def test_default_retains_winner_without_its_old_stopping_budget(self):
        default = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
        historical = yaml.safe_load(
            (ROOT / "configs/archive/program2_h100_100k/r10_sqrtloss.yaml").read_text()
        )
        for key in ("max_steps", "schedule_steps", "walltime_seconds"):
            historical.pop(key)
            self.assertNotIn(key, default)
        self.assertEqual(default, historical)

    def test_research_clears_old_caps_and_records_explicit_seed_and_backend(self):
        path = ROOT / "configs/archive/program2_h100_100k/r10_sqrtloss.yaml"
        source = path.read_bytes()
        config = self.print_config(
            path,
            "--seed",
            "43",
            "--attention-backend",
            "flash",
            "--warmup-steps",
            "554",
            "--max-steps",
            "none",
            "--schedule-steps",
            "none",
            "--max-model-tokens",
            "none",
            "--walltime-seconds",
            "3600",
            "--periodic-evaluation-interval",
            "0",
        )
        self.assertEqual(path.read_bytes(), source)
        self.assertEqual(
            (config["seed"], config["warmup_steps"], config["walltime_seconds"]),
            (43, 554, 3600),
        )
        self.assertEqual(config["attention_backend"], "flash")
        for key in ("max_steps", "schedule_steps", "max_model_tokens"):
            self.assertIsNone(config[key])
        self.assertEqual(config["optimizer"], "muon")
        self.assertEqual(config["training_loss_reduction"], "sqrt_mask_count")

    def test_manual_transfer_matches_baseline_and_candidate_execution_settings(self):
        flags = (
            "--seed",
            "42",
            "--attention-backend",
            "flash3",
            "--warmup-steps",
            "1000",
            "--max-steps",
            "none",
            "--schedule-steps",
            "100000",
            "--max-model-tokens",
            "24200224761",
            "--walltime-seconds",
            "57600",
            "--learning-rate",
            "0.0005",
            "--weight-decay",
            "0.01",
            "--micro-batch-size",
            "64",
            "--gradient-accumulation",
            "4",
        )
        baseline = self.print_config(ROOT / "configs/esmc-171m-original.yaml", *flags)
        candidate = self.print_config(ROOT / "configs/default.yaml", *flags)
        for key in (
            "seed",
            "attention_backend",
            "warmup_steps",
            "max_steps",
            "schedule_steps",
            "max_model_tokens",
            "walltime_seconds",
            "learning_rate",
            "weight_decay",
            "stages",
        ):
            self.assertEqual(candidate[key], baseline[key], key)
        self.assertEqual(candidate["muon_attention_lr_scale"], 0.9)
        self.assertEqual(candidate["muon_ffn_lr_scale"], 0.75)
        self.assertEqual(candidate["muon_weight_decay_scale"], 0.75)

    def test_omitted_cli_arguments_preserve_legacy_recipe(self):
        path = ROOT / "configs/esmc-171m-original.yaml"
        self.assertEqual(self.print_config(path), yaml.safe_load(path.read_text()))

    def test_batch_override_rejects_ambiguous_multistage_target(self):
        config = {"stages": [{"name": "stage1"}, {"name": "stage2"}]}
        with self.assertRaisesRegex(ValueError, "single-stage"):
            resolve_config_overrides(config, {"micro_batch_size": 64})

    def test_override_does_not_mutate_nested_source_settings(self):
        config = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
        before = copy.deepcopy(config)
        resolved = resolve_config_overrides(config, {"gradient_accumulation": 8})
        self.assertEqual(config, before)
        self.assertEqual(resolved["stages"][0]["gradient_accumulation"], 8)


if __name__ == "__main__":
    unittest.main()
