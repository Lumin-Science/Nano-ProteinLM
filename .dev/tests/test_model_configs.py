import unittest

from nanoprotein.model import ESMCConfig, expected_parameter_count


class ESMCScaleConfigTests(unittest.TestCase):
    def test_family_scaled_171m_shape_and_parameter_count(self) -> None:
        config = ESMCConfig.esmc_171m()

        self.assertEqual(config.d_model, 768)
        self.assertEqual(config.n_heads, 12)
        self.assertEqual(config.n_layers, 24)
        self.assertEqual(config.head_dim, 64)
        self.assertEqual(expected_parameter_count(config), 170_671_168)

    def test_released_scale_parameter_counts_remain_stable(self) -> None:
        self.assertEqual(expected_parameter_count(ESMCConfig.esmc_300m()), 332_997_184)
        self.assertEqual(expected_parameter_count(ESMCConfig.esmc_600m()), 575_036_992)


if __name__ == "__main__":
    unittest.main()
