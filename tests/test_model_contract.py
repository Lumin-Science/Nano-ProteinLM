import unittest

import torch

from nano_protein.model import (
    ESMCConfig,
    _apply_rope,
    build_model,
    count_parameters,
    expected_parameter_count,
)


class ModelContractTests(unittest.TestCase):
    def test_released_parameter_counts(self) -> None:
        self.assertEqual(expected_parameter_count(ESMCConfig.esmc_300m()), 332_997_184)
        self.assertEqual(expected_parameter_count(ESMCConfig.esmc_600m()), 575_036_992)

    def test_materialized_tiny_matches_formula(self) -> None:
        model = build_model("tiny", attention_backend="math")
        self.assertEqual(count_parameters(model), expected_parameter_count(model.config))

    def test_rope_preserves_vector_norm(self) -> None:
        generator = torch.Generator().manual_seed(7)
        query = torch.randn(2, 2, 11, 64, generator=generator)
        key = torch.randn(2, 2, 11, 64, generator=generator)
        rotated_query, rotated_key = _apply_rope(query, key)
        torch.testing.assert_close(
            rotated_query.square().sum(dim=-1), query.square().sum(dim=-1)
        )
        torch.testing.assert_close(rotated_key.square().sum(dim=-1), key.square().sum(dim=-1))

    def test_tiny_forward_shapes(self) -> None:
        model = build_model("tiny", attention_backend="math")
        inputs = torch.randint(4, 24, (2, 17))
        mask = torch.ones_like(inputs, dtype=torch.bool)
        output = model(inputs, mask, output_attentions=True, output_hidden_states=True)
        self.assertEqual(output["logits"].shape, (2, 17, 64))
        self.assertEqual(len(output["attentions"]), 2)
        self.assertEqual(output["attentions"][0].shape, (2, 2, 17, 17))


if __name__ == "__main__":
    unittest.main()
