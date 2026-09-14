"""Resume must preserve the next optimizer update and the sampled data stream."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nanoprotein.data import MixtureBatcher, _StoreWriter
from nanoprotein.model import build_model
from nanoprotein.resume import validate_resume
from nanoprotein.tokenizer import ProteinTokenizer
from nanoprotein.train import _OptimizerBundle, build_optimizer


class ResumeTests(unittest.TestCase):
    def test_optimizer_bundle_restores_next_update_and_live_parameter_groups(self):
        config = {"optimizer": "muon", "learning_rate": 5e-4, "weight_decay": 0.01}
        model = build_model("tiny", attention_backend="math")
        opt = build_optimizer(model, config)
        self.assertIsInstance(opt, _OptimizerBundle)
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 0.02
        opt.step()
        restored = build_model("tiny", attention_backend="math")
        restored.load_state_dict(model.state_dict())
        other = build_optimizer(restored, config)
        other.load_state_dict(copy.deepcopy(opt.state_dict()))
        for bundle in (opt, other):
            for group in bundle.param_groups:
                group["lr"] = 1e-4
            for _, child in bundle.named_optimizers:
                self.assertTrue(all(g["lr"] == 1e-4 for g in child.param_groups))
        for a, b in zip(model.parameters(), restored.parameters(), strict=True):
            a.grad = torch.ones_like(a) * 0.03
            b.grad = a.grad.clone()
        opt.step()
        other.step()
        for a, b in zip(model.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_adamw_restores_moments_step_and_next_update(self):
        model = torch.nn.Linear(3, 2)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        model(torch.ones(4, 3)).square().mean().backward()
        opt.step()
        other_model = copy.deepcopy(model)
        other = torch.optim.AdamW(other_model.parameters(), lr=0.001)
        other.load_state_dict(copy.deepcopy(opt.state_dict()))
        for m, o in ((model, opt), (other_model, other)):
            o.zero_grad()
            m(torch.full((4, 3), 2.0)).square().mean().backward()
            o.step()
        for a, b in zip(model.parameters(), other_model.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(all(float(s["step"]) == 2 for s in other.state.values()))

    def test_batcher_restores_next_crop_source_rng_and_epoch_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("uniref90", "mgnify"):
                writer = _StoreWriter(root / name / "train")
                for i in range(9):
                    writer.append(
                        np.arange(4, 30, dtype=np.uint8),
                        hashlib.sha256(f"{name}{i}".encode()).hexdigest(),
                    )
                writer.finish()
            args = (root, "train", {"uniref90": 0.7, "mgnify": 0.3})
            original = MixtureBatcher(*args, seed=71, rank=1, world_size=2)
            tokenizer = ProteinTokenizer.esmc()
            original.batch(17, context_length=12, tokenizer=tokenizer)
            restored = MixtureBatcher(*args, seed=71, rank=1, world_size=2)
            restored.load_state_dict(original.state_dict())
            for _ in range(3):
                a = original.batch(17, context_length=12, tokenizer=tokenizer)
                b = restored.batch(17, context_length=12, tokenizer=tokenizer)
                for x, y in zip(a, b, strict=True):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertEqual(original.source_counts, restored.source_counts)

    def test_eight_to_four_gpu_resume_keeps_global_batch_and_training_recipe(self):
        config = {
            "optimizer": "adamw",
            "learning_rate": 0.0005,
            "max_steps": 100000,
            "schedule_steps": 100000,
            "stages": [
                {
                    "name": "stage1",
                    "context_length": 512,
                    "micro_batch_size": 64,
                    "gradient_accumulation": 4,
                }
            ],
        }
        packet = {
            "train_config": config,
            "world_size": 8,
            "optimizer_step": 100000,
            "data_manifest_sha256": "same-data",
        }
        new = copy.deepcopy(config)
        new.update(max_steps=200000, schedule_steps=200000, stop_at_unix_time=2000000000)
        new["stages"][0]["gradient_accumulation"] = 8
        validate_resume(packet, new, world_size=4, data_manifest_sha256="same-data")
        with self.assertRaisesRegex(ValueError, "batch"):
            validate_resume(packet, new, world_size=8, data_manifest_sha256="same-data")
        with self.assertRaisesRegex(ValueError, "manifest"):
            validate_resume(packet, new, world_size=4, data_manifest_sha256="different-data")
        new["learning_rate"] = 0.001
        with self.assertRaisesRegex(ValueError, "recipe"):
            validate_resume(packet, new, world_size=4, data_manifest_sha256="same-data")

    def test_token_endpoint_can_extend_with_a_changed_gpu_layout(self):
        config = {
            "max_model_tokens": 1000,
            "schedule_steps": 100000,
            "stages": [
                {
                    "name": "stage1",
                    "context_length": 512,
                    "micro_batch_size": 64,
                    "gradient_accumulation": 4,
                }
            ],
        }
        packet = {
            "train_config": config,
            "world_size": 8,
            "optimizer_step": 3,
            "model_tokens": 1010,
            "data_manifest_sha256": "same-data",
        }
        new = copy.deepcopy(config)
        new["max_model_tokens"] = 2000
        new["stages"][0]["gradient_accumulation"] = 8
        validate_resume(packet, new, world_size=4, data_manifest_sha256="same-data")
        for endpoint in (1000, 1010):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, "tokens"):
                validate_resume(
                    packet,
                    {**new, "max_model_tokens": endpoint},
                    world_size=4,
                    data_manifest_sha256="same-data",
                )
        with self.assertRaisesRegex(ValueError, "saved step"):
            validate_resume(
                packet, {**new, "max_steps": 3}, world_size=4, data_manifest_sha256="same-data"
            )


if __name__ == "__main__":
    unittest.main()
