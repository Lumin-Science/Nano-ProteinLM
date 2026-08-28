import unittest

import numpy as np

from nano_protein.evaluate import (
    _score_long_range_pairs_and_digest,
    _score_sparse_long_range_pairs_and_digest,
    _symmetrized_attention_planes_batched,
)


class FastContactScoringTests(unittest.TestCase):
    def test_sparse_scoring_matches_all_channel_reference(self) -> None:
        generator = np.random.default_rng(20260828)
        residue_length = 40
        layers = 3
        heads = 4
        attentions = [
            generator.normal(size=(1, heads, residue_length + 2, residue_length + 2)).astype(
                np.float32
            )
            for _ in range(layers)
        ]
        coefficients = np.asarray(
            [0.0, 0.2, 0.0, -0.4, 0.7, 0.0, 0.0, 0.1, 0.0, -0.3, 0.5, 0.0],
            dtype=np.float64,
        )
        reference, _reference_digest = _score_long_range_pairs_and_digest(
            attentions,
            coefficients,
            -0.15,
            residue_length=residue_length,
            sequence_separation=24,
            attention_planes=_symmetrized_attention_planes_batched,
        )
        sparse, _sparse_digest = _score_sparse_long_range_pairs_and_digest(
            attentions,
            coefficients,
            -0.15,
            residue_length=residue_length,
            sequence_separation=24,
        )
        np.testing.assert_allclose(sparse, reference, rtol=0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
