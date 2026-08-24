import unittest

from nano_protein.schedule import Stage, stage_for_progress, stage_for_time, wsd_multiplier


class ScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        mixture = {"uniref90": 1.0}
        self.stages = (
            Stage("stage1", 512, 2, 1, mixture),
            Stage("stage2", 2048, 1, 2, mixture),
        )

    def test_two_to_one_time_split(self) -> None:
        stage, progress = stage_for_time(
            1199, walltime_seconds=1800, stage1_fraction=2 / 3, stages=self.stages
        )
        self.assertEqual(stage.name, "stage1")
        self.assertGreater(progress, 0.99)
        stage, progress = stage_for_time(
            1500, walltime_seconds=1800, stage1_fraction=2 / 3, stages=self.stages
        )
        self.assertEqual(stage.name, "stage2")
        self.assertAlmostEqual(progress, 0.5)

    def test_one_stage_uses_full_progress_interval(self) -> None:
        stage, progress = stage_for_progress(
            0.42,
            stage1_fraction=2 / 3,
            stages=(self.stages[0],),
        )
        self.assertEqual(stage.name, "stage1")
        self.assertAlmostEqual(progress, 0.42)

    def test_wsd(self) -> None:
        self.assertAlmostEqual(
            wsd_multiplier(
                optimizer_step=5,
                warmup_steps=10,
                stage_name="stage1",
                stage_progress=0.2,
            ),
            0.5,
        )
        self.assertAlmostEqual(
            wsd_multiplier(
                optimizer_step=20,
                warmup_steps=10,
                stage_name="stage2",
                stage_progress=1.0,
            ),
            0.1,
        )

    def test_one_stage_optional_linear_cooldown(self) -> None:
        arguments = {
            "optimizer_step": 100,
            "warmup_steps": 10,
            "stage_name": "stage1",
            "minimum_ratio": 0.1,
            "stage1_cooldown_fraction": 0.2,
        }
        self.assertAlmostEqual(wsd_multiplier(stage_progress=0.79, **arguments), 1.0)
        self.assertAlmostEqual(wsd_multiplier(stage_progress=0.80, **arguments), 1.0)
        self.assertAlmostEqual(wsd_multiplier(stage_progress=0.90, **arguments), 0.55)
        self.assertAlmostEqual(wsd_multiplier(stage_progress=1.00, **arguments), 0.1)

    def test_rejects_invalid_one_stage_cooldown(self) -> None:
        for fraction in (-0.1, 1.0):
            with self.subTest(fraction=fraction):
                with self.assertRaisesRegex(ValueError, "cooldown fraction"):
                    wsd_multiplier(
                        optimizer_step=20,
                        warmup_steps=10,
                        stage_name="stage1",
                        stage_progress=0.5,
                        stage1_cooldown_fraction=fraction,
                    )


if __name__ == "__main__":
    unittest.main()
