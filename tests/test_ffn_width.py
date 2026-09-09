"""Qualify FFN width budgets, backward compatibility, and contact feature layout."""

import tempfile
import unittest
from pathlib import Path

import torch

from nanoprotein.evaluate import load_checkpoint
from nanoprotein.model import (
    ESMCConfig,
    ESMCForMaskedLM,
    count_parameters,
    expected_parameter_count,
)


class FFNWidthTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(22)

    def test_exact_budget_and_frozen_loader_preserves_attention_features(self):
        config = ESMCConfig.esmc_171m(ffn_hidden_dim=1536)
        with torch.device("meta"):
            full = ESMCForMaskedLM(config)
        self.assertEqual(count_parameters(full), 142_359_616)
        self.assertEqual(expected_parameter_count(config), count_parameters(full))
        self.assertEqual(full.blocks[0].ffn.gate_up.weight.shape, (3072, 768))
        self.assertEqual(full.blocks[-1].ffn.down.weight.shape, (768, 1536))
        tiny = ESMCForMaskedLM(ESMCConfig.tiny(ffn_hidden_dim=192, attention_backend="math"))
        self.assertEqual(count_parameters(tiny), expected_parameter_count(tiny.config))
        tokens = torch.tensor([[0, 4, 5, 6, 2], [0, 7, 8, 2, 1]])
        mask = tokens != 1
        expected = tiny(tokens, mask, output_attentions=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {"model_config": tiny.config.to_dict(), "model": tiny.state_dict()}, path
            )
            loaded, _ = load_checkpoint(path, torch.device("cpu"))
        actual = loaded(tokens, mask, output_attentions=True)
        torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)
        self.assertEqual(len(actual["attentions"]), 2)
        for a, b in zip(actual["attentions"], expected["attentions"], strict=True):
            self.assertEqual(a.shape, (2, 2, 5, 5))
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_legacy_config_and_explicit_default_are_bitwise_identical(self):
        config = ESMCConfig.tiny(attention_backend="math")
        base = ESMCForMaskedLM(config)
        rng = torch.get_rng_state()
        torch.manual_seed(22)
        explicit = ESMCForMaskedLM(
            ESMCConfig.tiny(ffn_hidden_dim=512, attention_backend="math")
        )
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in base.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
        legacy = config.to_dict()
        del legacy["ffn_hidden_dim"]
        self.assertEqual(ESMCConfig(**legacy), config)
        for width in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                ESMCConfig.tiny(ffn_hidden_dim=width)


if __name__ == "__main__":
    unittest.main()
