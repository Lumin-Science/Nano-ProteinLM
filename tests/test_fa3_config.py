import unittest
from pathlib import Path

import torch
import yaml

from nanoprotein.model import build_model, count_parameters


class FA3ConfigTests(unittest.TestCase):
    def test_h100_preset_preserves_original_recipe(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        original = yaml.safe_load((root / "esmc-171m-original.yaml").read_text())
        hopper = yaml.safe_load(
            (root / "archive/esmc-171m-original-h100-fa3-12h.yaml").read_text()
        )
        expected = dict(
            original,
            attention_backend="flash3",
            walltime_seconds=43200,
            peak_bf16_tflops_per_gpu=989.5,
        )
        self.assertEqual(hopper, expected)
        with torch.device("meta"):
            model = build_model(hopper["model"], attention_backend=hopper["attention_backend"])
        self.assertEqual(count_parameters(model), original["expected_parameter_count"])

    def test_explicit_fa3_does_not_fall_back_on_cpu(self):
        model = build_model("tiny", attention_backend="flash3")
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            model(torch.tensor([[0, 4, 5, 2]]))


if __name__ == "__main__":
    unittest.main()
