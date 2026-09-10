"""Consumption conservation across data expansion, source epochs and GPU layouts."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nanoprotein.data import MixtureBatcher, _StoreWriter
from nanoprotein.data_budget import data_coverage
from nanoprotein.data_migration import legacy_origins, validate_migration, verify_prefix
from nanoprotein.global_sampling import (
    GlobalMixtureBatcher,
    GlobalRows,
    portable_batcher_states,
)
from nanoprotein.resume import validate_resume
from nanoprotein.tokenizer import ProteinTokenizer


class GlobalSamplingTests(unittest.TestCase):
    def test_legacy_unequal_partitions_and_added_records_complete_before_reuse(self):
        origin = dict(size=41, seed=19, world_size=4, cursors=[10, 3, 7, 0])
        old = np.random.default_rng(19).permutation(41)
        consumed = {int(x) for r, n in enumerate(origin["cursors"]) for x in old[r::4][:n]}
        sampler = GlobalRows(63, seed=73, allow_resampling=True, origin=origin)
        remaining = sampler.take(63 - len(consumed))
        self.assertEqual(set(remaining), set(range(63)) - consumed)
        self.assertEqual(len(remaining), len(set(remaining)))
        self.assertEqual(sampler.exposure()["repeated_draws"], 0)
        second = sampler.take(63)
        self.assertEqual(set(second), set(range(63)))
        self.assertEqual(sampler.exposure()["unique_records_seen"], 63)
        self.assertEqual(sampler.exposure()["repeated_draws"], 63)
        restored = GlobalRows(63, seed=73, allow_resampling=True, origin=origin)
        restored.load_state_dict(sampler.state_dict())
        np.testing.assert_array_equal(sampler.take(100), restored.take(100))

    def test_strict_source_exhaustion_is_atomic(self):
        sampler = GlobalRows(7, seed=1, allow_resampling=False)
        sampler.take(5)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            sampler.take(3)
        self.assertEqual(sampler.cursor, 5)
        sampler.take(2)
        with self.assertRaises(RuntimeError):
            sampler.take(1)
        self.assertEqual(sampler.epoch, 0)

    def test_complete_legacy_source_can_start_second_epoch_only_when_allowed(self):
        origin = dict(size=7, seed=1, world_size=4, cursors=[2, 2, 2, 1])
        s = GlobalRows(7, seed=3, allow_resampling=True, origin=origin)
        self.assertEqual(set(s.take(7)), set(range(7)))
        self.assertEqual(s.exposure()["draws"], 14)
        strict = GlobalRows(7, seed=3, allow_resampling=False, origin=origin)
        with self.assertRaises(RuntimeError):
            strict.take(1)

    def make_data(self, root, sizes):
        for source, size in sizes.items():
            writer = _StoreWriter(root / source / "train")
            for i in range(size):
                writer.append(
                    np.arange(4, 40, dtype=np.uint8),
                    hashlib.sha256(f"{source}/{i}".encode()).hexdigest(),
                )
            writer.finish()

    def test_four_eight_four_layout_keeps_next_global_records_and_epoch_boundaries(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_data(root, {"uniref90": 43, "mgnify": 3000, "omg_img": 67})
            weights = {"uniref90": 0.36, "mgnify": 0.11, "omg_img": 0.54}
            policies = {"uniref90": "allow", "mgnify": "error", "omg_img": "allow"}

            def group(world, states=None):
                result = [
                    GlobalMixtureBatcher(
                        root,
                        "train",
                        weights,
                        seed=11,
                        rank=r,
                        world_size=world,
                        policies=policies,
                    )
                    for r in range(world)
                ]
                if states:
                    portable_batcher_states(
                        {
                            "world_size": len(states),
                            "runtime_states": [{"batchers": {"stage1": s}} for s in states],
                        }
                    )
                    for b in result:
                        b.load_state_dict(states[b.rank] if len(states) == world else states[0])
                return result

            def take(batchers):
                return [row for b in batchers for row in b.select_rows(32 // len(batchers))]

            original = group(4)
            for _ in range(3):
                take(original)
            switched = group(8, [b.state_dict() for b in original])
            for _ in range(6):
                self.assertEqual(take(original), take(switched))
            switched = group(4, [b.state_dict() for b in switched])
            for _ in range(6):
                self.assertEqual(take(original), take(switched))
            self.assertEqual(original[0].exposure(), switched[0].exposure())
            for name in weights:
                self.assertEqual(
                    sum(b.source_counts[name] for b in switched),
                    switched[0].exposure()[name]["draws"],
                )
            self.assertEqual(switched[0].exposure()["mgnify"]["repeated_draws"], 0)
            broken = [b.state_dict() for b in switched]
            broken[1]["samplers"]["uniref90"]["cursor"] += 1
            with self.assertRaisesRegex(ValueError, "disagree"):
                group(8, broken)

    def test_same_layout_restores_next_source_crop_and_tensor(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_data(root, {"uniref90": 43})
            kw = dict(seed=11, rank=0, world_size=1, policies={"uniref90": "allow"})
            b = GlobalMixtureBatcher(root, "train", {"uniref90": 1}, **kw)
            tok = ProteinTokenizer.esmc()
            b.batch(17, context_length=12, tokenizer=tok)
            other = GlobalMixtureBatcher(root, "train", {"uniref90": 1}, **kw)
            other.load_state_dict(b.state_dict())
            for _ in range(4):
                for a, c in zip(
                    b.batch(17, context_length=12, tokenizer=tok),
                    other.batch(17, context_length=12, tokenizer=tok),
                    strict=True,
                ):
                    torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_prefix_check_rejects_changed_identity_or_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_data(root / "old", {"uniref90": 3})
            self.make_data(root / "new", {"uniref90": 5})
            old, new = root / "old/uniref90/train", root / "new/uniref90/train"
            self.assertTrue(verify_prefix(old, new)["existing_indices_and_tokens_identical"])
            with (new / "tokens.bin").open("r+b") as f:
                f.write(b"\x01")
            with self.assertRaisesRegex(ValueError, "residues"):
                verify_prefix(old, new)

    def test_migration_binds_real_legacy_cursor_history_and_preserves_recipe(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sizes = dict.fromkeys(("uniref90", "mgnify", "omg_img"), 103)
            self.make_data(root, sizes)
            weights = {"uniref90": 0.36, "mgnify": 0.11, "omg_img": 0.54}
            states = []
            for rank in range(4):
                b = MixtureBatcher(
                    root,
                    "train",
                    weights,
                    seed=5,
                    rank=rank,
                    world_size=4,
                    allow_resampling=False,
                )
                b.batch(4, context_length=12, tokenizer=ProteinTokenizer.esmc())
                states.append({"batchers": {"stage1": b.state_dict()}})
            old = {
                "max_steps": 1,
                "schedule_steps": 1,
                "data_resampling": "error",
                "learning_rate": 5e-4,
                "stages": [
                    {
                        "name": "stage1",
                        "micro_batch_size": 4,
                        "gradient_accumulation": 1,
                        "mixture": weights,
                    }
                ],
            }
            packet = {
                "world_size": 4,
                "optimizer_step": 1,
                "sequences_seen": 16,
                "runtime_states": states,
                "train_config": old,
                "data_manifest_sha256": "old",
            }
            origins = legacy_origins(packet)
            migration = {
                "protocol": "append-only-global-sampler-migration-v1",
                "status": "passed",
                "parent_step": 1,
                "parent_sequences_seen": 16,
                "old_manifest_sha256": "old",
                "new_manifest_sha256": "new",
                "origins": origins,
                "validation_unchanged": True,
                "prefix_verification": {
                    k: {
                        "old_records": 103,
                        "new_records": 200,
                        "existing_indices_and_tokens_identical": True,
                    }
                    for k in sizes
                },
            }
            new = copy.deepcopy(old)
            new.update(
                max_steps=10,
                schedule_steps=10,
                data_sampler="global",
                data_resampling="per_source",
                data_source_resampling={
                    "uniref90": "allow",
                    "mgnify": "error",
                    "omg_img": "allow",
                },
            )
            new["stages"][0]["micro_batch_size"] = 2
            validate_resume(
                packet, new, world_size=8, data_manifest_sha256="new", migration=migration
            )
            with self.assertRaisesRegex(ValueError, "manifest"):
                validate_resume(packet, new, world_size=8, data_manifest_sha256="new")
            changed = copy.deepcopy(new)
            changed["learning_rate"] = 1e-3
            with self.assertRaisesRegex(ValueError, "recipe"):
                validate_resume(
                    packet,
                    changed,
                    world_size=8,
                    data_manifest_sha256="new",
                    migration=migration,
                )
            bad = copy.deepcopy(migration)
            bad["origins"]["mgnify"]["cursors"][0] += 1
            with self.assertRaises(ValueError):
                validate_migration(packet, bad, "new")

    def test_400k_budget_allows_only_declared_repeated_sources(self):
        config = {
            "max_steps": 400000,
            "data_resampling": "per_source",
            "data_sampler": "global",
            "data_source_resampling": {
                "uniref90": "allow",
                "mgnify": "error",
                "omg_img": "allow",
            },
            "stages": [
                {
                    "name": "stage1",
                    "micro_batch_size": 64,
                    "gradient_accumulation": 4,
                    "mixture": {"uniref90": 0.36, "mgnify": 0.11, "omg_img": 0.54},
                }
            ],
        }
        manifest = {
            "sources": {
                k: {"train": {"records": n}}
                for k, n in {
                    "uniref90": 74175974,
                    "mgnify": 90495061,
                    "omg_img": 262845186,
                }.items()
            }
        }
        receipt = data_coverage(config, manifest, world_size=8)
        self.assertEqual(receipt["status"], "passed")
        self.assertFalse(receipt["sources"]["uniref90"]["sufficient"])
        manifest["sources"]["mgnify"]["train"]["records"] = 22984503
        with self.assertRaisesRegex(ValueError, "mgnify"):
            data_coverage(config, manifest, world_size=8)


if __name__ == "__main__":
    unittest.main()
