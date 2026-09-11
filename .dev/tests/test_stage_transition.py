"""Stage transitions preserve learned state, consumed identities and the decay clock."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nanoprotein.data import _StoreWriter
from nanoprotein.data_budget import data_coverage
from nanoprotein.global_sampling import (
    GlobalMixtureBatcher,
    GlobalRows,
    portable_batcher_states,
    row_state_exposure,
)
from nanoprotein.resume import validate_resume
from nanoprotein.schedule import continuation_progress, wsd_multiplier
from nanoprotein.stage_transition import transition_runtime, validate_transition_config
from nanoprotein.train import resolve_step_budgets


class StageTransitionTests(unittest.TestCase):
    def test_nested_expansion_excludes_every_consumed_identity_and_restores(self):
        origin = dict(size=41, seed=19, world_size=4, cursors=[10, 3, 7, 0])
        permutation = np.random.default_rng(19).permutation(41)
        seen = {
            int(x) for rank, n in enumerate(origin["cursors"]) for x in permutation[rank::4][:n]
        }
        first = GlobalRows(83, seed=73, allow_resampling=False, origin=origin)
        seen.update(first.take(50))
        second = GlobalRows(
            131,
            seed=73,
            allow_resampling=False,
            origin=dict(protocol="global-unseen-origin-v1", state=first.state_dict()),
        )
        self.assertEqual(second.exposure()["draws"], len(seen))
        seen.update(second.take(9))
        restored = GlobalRows(131, seed=73, allow_resampling=False, origin=second.origin)
        restored.load_state_dict(second.state_dict())
        count = 131 - len(seen)
        tail = second.take(count)
        np.testing.assert_array_equal(tail, restored.take(count))
        self.assertEqual(set(tail), set(range(131)) - seen)
        self.assertEqual(len(tail), len(set(tail)))
        self.assertEqual(row_state_exposure(second.state_dict())["unique_records_seen"], 131)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            second.take(1)
        third = GlobalRows(
            170,
            seed=73,
            allow_resampling=False,
            origin=dict(protocol="global-unseen-origin-v1", state=second.state_dict()),
        )
        self.assertEqual(set(third.take(39)), set(range(131, 170)))

    def test_mixture_change_preserves_repeated_source_cursors_and_gpu_portability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sizes = dict(uniref90=13, mgnify=1000, omg_img=19)
            for source, size in sizes.items():
                writer = _StoreWriter(root / source / "train")
                for i in range(size):
                    writer.append(
                        np.arange(4, 12, dtype=np.uint8),
                        hashlib.sha256(f"{source}/{i}".encode()).hexdigest(),
                    )
                writer.finish()
            policies = dict(uniref90="allow", mgnify="error", omg_img="allow")
            oldweights = dict(uniref90=0.36, mgnify=0.11, omg_img=0.54)
            batchers = [
                GlobalMixtureBatcher(
                    root, "train", oldweights, seed=11, rank=r, world_size=8, policies=policies
                )
                for r in range(8)
            ]
            for _ in range(10):
                for b in batchers:
                    b.select_rows(4)
            old = dict(
                max_steps=10,
                schedule_steps=10,
                expected_world_size=8,
                data_sampler="global",
                data_resampling="per_source",
                data_source_resampling=policies,
                learning_rate=0.0005,
                weight_decay=0.01,
                stages=[
                    dict(
                        name="stage1",
                        context_length=512,
                        micro_batch_size=4,
                        gradient_accumulation=1,
                        mixture=oldweights,
                    )
                ],
            )
            packet = dict(
                train_config=old,
                optimizer_step=10,
                world_size=8,
                sequences_seen=320,
                runtime_states=[dict(batchers=dict(stage1=b.state_dict())) for b in batchers],
                data_manifest_sha256="old",
            )
            new = copy.deepcopy(old)
            new.update(max_steps=30, schedule_steps=30, schedule_start_step=10)
            new["stages"][0].update(
                name="stage2",
                context_length=2048,
                mixture=dict(uniref90=0.63, mgnify=0.06, omg_img=0.31),
            )
            validate_transition_config(packet, new)
            changed = copy.deepcopy(new)
            changed["learning_rate"] *= 2
            with self.assertRaisesRegex(ValueError, "protected"):
                validate_transition_config(packet, changed)
            with self.assertRaisesRegex(ValueError, "recipe"):
                validate_resume(packet, new, world_size=8, data_manifest_sha256="old")
            runtime = transition_runtime(packet, new, sizes, dict(protocol="test-transition"))
            transformed = dict(
                packet, train_config=new, runtime_states=runtime, data_manifest_sha256="new"
            )
            portable_batcher_states(transformed)
            groups = []
            for world in (8, 4):
                group = []
                for rank in range(world):
                    state = runtime[rank if world == 8 else 0]["batchers"]["stage2"]
                    b = GlobalMixtureBatcher(
                        root,
                        "train",
                        new["stages"][0]["mixture"],
                        seed=11,
                        rank=rank,
                        world_size=world,
                        policies=policies,
                        migration=state["migration"],
                    )
                    b.load_state_dict(state)
                    group.append(b)
                groups.append(group)
            self.assertEqual(groups[0][0].exposure(), batchers[0].exposure())
            for _ in range(6):
                a = [row for b in groups[0] for row in b.select_rows(4)]
                b = [row for b in groups[1] for row in b.select_rows(8)]
                self.assertEqual(a, b)
            for source in sizes:
                self.assertEqual(
                    sum(b.source_counts[source] for b in groups[1]),
                    groups[1][0].exposure()[source]["draws"],
                )
            resumed = dict(transformed, optimizer_step=20)
            validate_resume(resumed, new, world_size=8, data_manifest_sha256="new")
            bad = copy.deepcopy(new)
            bad["schedule_steps"] = 40
            with self.assertRaisesRegex(ValueError, "decaying"):
                validate_resume(resumed, bad, world_size=8, data_manifest_sha256="new")

    def test_capacity_checks_unused_records_and_only_remaining_updates(self):
        config = dict(
            max_steps=110,
            schedule_start_step=100,
            data_resampling="error",
            stages=[
                dict(
                    name="stage2",
                    micro_batch_size=4,
                    gradient_accumulation=1,
                    mixture=dict(mgnify=1),
                )
            ],
        )
        manifest = dict(sources=dict(mgnify=dict(train=dict(records=100))))
        exposure = dict(mgnify=dict(unique_records_seen=80))
        with self.assertRaisesRegex(ValueError, "20 unused"):
            data_coverage(
                config, manifest, world_size=1, resume_step=100, source_exposure=exposure
            )
        manifest["sources"]["mgnify"]["train"]["records"] = 150
        result = data_coverage(
            config, manifest, world_size=1, resume_step=100, source_exposure=exposure
        )
        self.assertEqual(result["stages"][0]["steps"], 10)
        self.assertEqual(result["sources"]["mgnify"]["unused_records"], 70)
        result = data_coverage(
            config, manifest, world_size=1, resume_step=105, source_exposure=exposure
        )
        self.assertEqual(result["stages"][0]["steps"], 5)

    def test_decay_starts_at_transition_and_does_not_restart_on_resume(self):
        self.assertEqual(
            resolve_step_budgets(
                dict(max_steps=400200, schedule_steps=700000, schedule_start_step=400000)
            ),
            (400200, 700000),
        )
        for step, expected in ((400000, 1.0), (550000, 0.55), (700000, 0.1)):
            progress = continuation_progress(step, 700000, 400000)
            multiplier = wsd_multiplier(
                optimizer_step=step + 1,
                warmup_steps=1000,
                stage_name="stage2",
                stage_progress=progress,
            )
            self.assertAlmostEqual(multiplier, expected)
        with self.assertRaises(ValueError):
            continuation_progress(399999, 700000, 400000)


if __name__ == "__main__":
    unittest.main()
