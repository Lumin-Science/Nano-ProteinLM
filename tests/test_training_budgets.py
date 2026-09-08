import unittest
from pathlib import Path

import yaml

from nano_protein.train import resolve_step_budgets, resolve_token_budget, training_stop_reason


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

    def test_token_limit_is_optional_and_requires_an_exact_positive_count(self):
        self.assertIsNone(resolve_token_budget({}))
        config = {"stages": [{"name": "stage1"}], "max_model_tokens": 24200224761}
        self.assertEqual(resolve_token_budget(config), 24200224761)
        for value in (0, -1, True, 1.5, "100"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_token_budget({**config, "max_model_tokens": value})

    def test_token_limit_does_not_silently_change_a_multistage_or_decay_schedule(self):
        config = {"stages": [{"name": "stage1"}], "max_model_tokens": 100}
        for change in (
            {"stages": [{"name": "stage1"}, {"name": "stage2"}]},
            {"stage1_cooldown_fraction": 0.1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                resolve_token_budget({**config, **change})

    def test_stop_uses_first_completed_global_update_including_accumulation(self):
        # Per-rank, per-microbatch non-padding counts: sum before checking the endpoint.
        updates = (((9, 11), (12, 8)), ((10, 12), (9, 11)), ((11, 10), (12, 9)))
        for target, expected_step in ((40, 1), (41, 2), (82, 2), (83, 3)):
            with self.subTest(target=target):
                total = 0
                step = 0
                reason = None
                while reason is None:
                    last_update = sum(sum(rank) for rank in updates[step])
                    total += last_update
                    step += 1
                    reason = training_stop_reason(
                        model_tokens=total,
                        max_model_tokens=target,
                        optimizer_step=step,
                        max_steps=None,
                        training_seconds=step,
                        walltime_seconds=10,
                    )
                self.assertEqual((reason, step), ("max_model_tokens", expected_step))
                self.assertGreaterEqual(total, target)
                self.assertLess(total - target, last_update)

    def test_early_caps_are_not_reported_as_token_completion(self):
        base = dict(
            model_tokens=99,
            max_model_tokens=100,
            optimizer_step=2,
            max_steps=None,
            training_seconds=3,
            walltime_seconds=10,
        )
        self.assertIsNone(training_stop_reason(**base))
        self.assertEqual(training_stop_reason(**{**base, "max_steps": 2}), "max_steps")
        self.assertEqual(training_stop_reason(**{**base, "walltime_seconds": 3}), "walltime")
        self.assertEqual(
            training_stop_reason(
                **{**base, "model_tokens": 100, "max_steps": 2, "walltime_seconds": 3}
            ),
            "max_model_tokens",
        )

    def test_legacy_step_and_time_endpoint_precedence_is_unchanged(self):
        base = dict(
            model_tokens=1000,
            max_model_tokens=None,
            optimizer_step=2,
            max_steps=None,
            training_seconds=3,
            walltime_seconds=3,
        )
        self.assertEqual(training_stop_reason(**base), "walltime")
        self.assertEqual(training_stop_reason(**{**base, "max_steps": 2}), "max_steps")

    def test_public_best_alias_matches_canonical_incumbent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (root / "configs/esmc-300m-current-best.yaml").open() as handle:
            public_best = yaml.safe_load(handle)
        with (root / "configs/autoresearch_300m_4xa100_1h.yaml").open() as handle:
            campaign_incumbent = yaml.safe_load(handle)
        self.assertEqual(public_best, campaign_incumbent)


if __name__ == "__main__":
    unittest.main()
