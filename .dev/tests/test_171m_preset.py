import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from nanoprotein.evaluate import load_checkpoint
from nanoprotein.model import build_model, count_parameters
from nanoprotein.schedule import wsd_multiplier
from nanoprotein.train import build_optimizer, muon_adamw_parameter_groups


class Retained171MPresetTests(unittest.TestCase):
    @staticmethod
    def preset():
        path = (
            Path(__file__).resolve().parents[2]
            / ".dev/configs/archive/autoresearch_171m_4xl40s_1h.yaml"
        )
        return yaml.safe_load(path.read_text())

    @staticmethod
    def model_options(preset):
        return {
            key: preset[key]
            for key in (
                "rotary_base",
                "learned_residual_routing",
                "transformer_norm",
                "depth_scaled_residual_init",
            )
        }

    def test_full_preset_constructs_with_recorded_parameter_count(self):
        preset = self.preset()
        with torch.device("meta"):
            model = build_model(preset["model"], **self.model_options(preset))
        self.assertEqual(count_parameters(model), preset["expected_parameter_count"])
        self.assertEqual(
            (model.config.n_layers, model.config.d_model, model.config.n_heads),
            (24, 768, 12),
        )

    def test_muon_split_uses_hidden_matrices_only(self) -> None:
        model = build_model(
            "tiny",
            attention_backend="auto",
            learned_residual_routing=True,
            transformer_norm="layernorm",
        )
        named_parameters = dict(model.named_parameters())
        muon_groups, adamw_groups = muon_adamw_parameter_groups(
            model,
            weight_decay=0.1,
            muon_lr_scale=0.8,
            muon_weight_decay_scale=0.75,
        )
        muon_ids = {id(parameter) for group in muon_groups for parameter in group["params"]}
        adamw_ids = {id(parameter) for group in adamw_groups for parameter in group["params"]}
        trainable_ids = {
            id(parameter) for parameter in named_parameters.values() if parameter.requires_grad
        }

        self.assertEqual(muon_ids | adamw_ids, trainable_ids)
        self.assertFalse(muon_ids & adamw_ids)
        self.assertIn(id(named_parameters["blocks.0.attention.qkv.weight"]), muon_ids)
        self.assertIn(id(named_parameters["blocks.1.ffn.down.weight"]), muon_ids)
        self.assertNotIn(id(named_parameters["embedding.weight"]), muon_ids)
        self.assertNotIn(id(named_parameters["head_out.weight"]), muon_ids)
        self.assertEqual(muon_groups[0]["lr_scale"], 0.8)
        self.assertAlmostEqual(muon_groups[0]["weight_decay"], 0.075)
        for name, parameter in named_parameters.items():
            if id(parameter) in muon_ids:
                self.assertTrue(name.startswith("blocks."))
                self.assertEqual(parameter.ndim, 2)

    def test_muon_split_supports_attention_and_ffn_lr_scales(self) -> None:
        model = build_model("tiny", attention_backend="auto")
        muon_groups, _ = muon_adamw_parameter_groups(
            model,
            weight_decay=0.1,
            muon_lr_scale=0.8,
            muon_weight_decay_scale=0.75,
            muon_attention_lr_scale=0.9,
            muon_ffn_lr_scale=0.75,
        )
        self.assertEqual(
            [(group["name"], group["lr_scale"]) for group in muon_groups],
            [("muon_attention", 0.9), ("muon_ffn", 0.75)],
        )
        self.assertTrue(all(group["params"] for group in muon_groups))
        self.assertTrue(
            all(abs(float(group["weight_decay"]) - 0.075) < 1e-12 for group in muon_groups)
        )

    def test_muon_and_adamw_step_and_checkpoint_round_trip(self):
        torch.manual_seed(42)
        preset = self.preset()
        model = build_model("tiny", attention_backend="math", **self.model_options(preset))
        optimizer = build_optimizer(model, preset)
        parameters = dict(model.named_parameters())
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        tokens = torch.tensor([[0, 4, 5, 6, 2], [0, 7, 8, 9, 2]])
        logits = model(tokens)["logits"]
        loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for name in (
            "blocks.0.attention.qkv.weight",
            "blocks.0.ffn.down.weight",
            "embedding.weight",
        ):
            self.assertFalse(torch.equal(before[name], parameters[name]))
        self.assertTrue(all(parameter.grad is None for parameter in parameters.values()))
        states = optimizer.state_dict()["optimizers"]
        self.assertEqual([state["name"] for state in states], ["muon", "adamw"])
        self.assertTrue(all(state["state_dict"]["state"] for state in states))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            torch.save(
                {"model_config": model.config.to_dict(), "model": model.state_dict()},
                checkpoint,
            )
            restored, _ = load_checkpoint(checkpoint, torch.device("cpu"))
        self.assertEqual(restored.config.rotary_base, 20_000.0)
        torch.testing.assert_close(
            restored.blocks[0].attention.rotary.inv_freq,
            20_000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64),
        )
        with torch.no_grad():
            torch.testing.assert_close(
                restored(tokens)["logits"], model.eval()(tokens)["logits"]
            )

    def test_retained_presets_keep_their_distinct_schedules(self):
        root = Path(__file__).resolve().parents[2]
        current_300m = yaml.safe_load(
            (root / ".dev/configs/archive/esmc-300m-current-best.yaml").read_text()
        )
        for preset, expected in ((self.preset(), 1.0), (current_300m, 0.1)):
            with self.subTest(model=preset["model"]):
                value = wsd_multiplier(
                    optimizer_step=10_000,
                    warmup_steps=preset["warmup_steps"],
                    stage_name="stage1",
                    stage_progress=1.0,
                    minimum_ratio=preset.get("minimum_lr_ratio", 0.1),
                    stage1_cooldown_fraction=preset.get("stage1_cooldown_fraction", 0.0),
                )
                self.assertAlmostEqual(value, expected)


if __name__ == "__main__":
    unittest.main()
