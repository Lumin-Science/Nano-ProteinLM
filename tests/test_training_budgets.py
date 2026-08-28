import unittest
from pathlib import Path

import yaml

from nano_protein.train import resolve_step_budgets


class TrainingBudgetTests(unittest.TestCase):
    def test_defaults_to_wall_clock_only(self) -> None:
        self.assertEqual(resolve_step_budgets({}), (None, None))

    def test_legacy_max_steps_controls_stop_and_schedule(self) -> None:
        self.assertEqual(resolve_step_budgets({"max_steps": 100}), (100, 100))

    def test_schedule_can_finish_before_nonbinding_stop_cap(self) -> None:
        self.assertEqual(
            resolve_step_budgets({"max_steps": 120, "schedule_steps": 100}),
            (120, 100),
        )

    def test_schedule_can_be_used_without_a_step_stop(self) -> None:
        self.assertEqual(resolve_step_budgets({"schedule_steps": 100}), (None, 100))

    def test_rejects_schedule_beyond_stop_cap(self) -> None:
        with self.assertRaises(ValueError):
            resolve_step_budgets({"max_steps": 100, "schedule_steps": 101})

    def test_public_best_alias_matches_canonical_incumbent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "configs/esmc-300m-current-best.yaml").open() as handle:
            public_best = yaml.safe_load(handle)
        with (root / "configs/autoresearch_300m_4xa100_1h.yaml").open() as handle:
            campaign_incumbent = yaml.safe_load(handle)
        self.assertEqual(public_best, campaign_incumbent)


if __name__ == "__main__":
    unittest.main()
