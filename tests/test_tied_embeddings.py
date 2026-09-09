"""Check shared embedding gradients, optimizer ownership and frozen checkpoints."""

import tempfile
import unittest
from pathlib import Path

import torch

from nanoprotein.evaluate import load_checkpoint
from nanoprotein.model import (
    ESMCConfig,
    ESMCForMaskedLM,
    count_parameters,
    expected_parameter_count,
)
from nanoprotein.train import build_optimizer


class TiedEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)

    def tiny(self, tied=False):
        return ESMCForMaskedLM(
            ESMCConfig.tiny(
                ffn_hidden_dim=192, attention_backend="math", tie_word_embeddings=tied
            )
        )

    def test_initialization_rng_default_compatibility_and_exact_budget(self):
        base = self.tiny()
        rng = torch.get_rng_state()
        torch.manual_seed(29)
        tied = self.tiny(True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertIs(tied.embedding.weight, tied.head_out.weight)
        self.assertIsNot(base.embedding.weight, base.head_out.weight)
        for key, value in base.state_dict().items():
            if key != "head_out.weight":
                torch.testing.assert_close(value, tied.state_dict()[key], rtol=0, atol=0)
        config = base.config.to_dict()
        del config["tie_word_embeddings"]
        torch.manual_seed(29)
        legacy = ESMCForMaskedLM(ESMCConfig(**config))
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, legacy.state_dict()[key], rtol=0, atol=0)
        full_config = ESMCConfig.esmc_171m(ffn_hidden_dim=1536, tie_word_embeddings=True)
        with torch.device("meta"):
            full = ESMCForMaskedLM(full_config)
        self.assertEqual(count_parameters(full), 142_310_464)
        self.assertEqual(expected_parameter_count(full_config), count_parameters(full))
        self.assertEqual(count_parameters(base) - count_parameters(tied), 64 * 128)
        tied.double()
        self.assertIs(tied.embedding.weight, tied.head_out.weight)

    def test_shared_gradient_equals_sum_of_both_roles(self):
        tied = self.tiny(True).double()
        untied = self.tiny().double()
        untied.load_state_dict(tied.state_dict())
        tokens = torch.tensor([[0, 4, 5, 6, 2], [0, 7, 8, 2, 1]])
        mask = tokens != 1
        outputs = []
        for model in (tied, untied):
            logits = model(tokens, mask)["logits"]
            # A vocabulary-wide loss exercises input lookup and output classifier roles.
            logits[mask].square().mean().backward()
            outputs.append(logits)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        self.assertGreater(untied.embedding.weight.grad.norm().item(), 0)
        self.assertGreater(untied.head_out.weight.grad.norm().item(), 0)
        torch.testing.assert_close(
            tied.embedding.weight.grad,
            untied.embedding.weight.grad + untied.head_out.weight.grad,
            rtol=1e-11,
            atol=1e-12,
        )
        for name, parameter in tied.named_parameters():
            if name != "embedding.weight":
                torch.testing.assert_close(
                    parameter.grad,
                    dict(untied.named_parameters())[name].grad,
                    rtol=1e-11,
                    atol=1e-12,
                )

    def test_adamw_owns_shared_weight_once_and_applies_one_update(self):
        model = self.tiny(True)
        config = {"optimizer": "muon", "learning_rate": 0.000326599, "weight_decay": 0.0183712}
        optimizer = build_optimizer(model, config)
        muon, adamw = [o for _, o in optimizer.named_optimizers]
        shared = model.embedding.weight
        params = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(params), len({id(p) for p in params}))
        self.assertEqual(sum(p is shared for p in params), 1)
        self.assertFalse(any(p is shared for g in muon.param_groups for p in g["params"]))
        self.assertTrue(any(p is shared for g in adamw.param_groups for p in g["params"]))
        reference = torch.nn.Parameter(shared.detach().clone())
        reference_optimizer = torch.optim.AdamW(
            [reference],
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=True,
        )
        for _ in range(3):
            gradient = torch.randn_like(shared)
            shared.grad = gradient.clone()
            reference.grad = gradient.clone()
            optimizer.step()
            reference_optimizer.step()
            torch.testing.assert_close(shared, reference, rtol=0, atol=0)
            for key, value in adamw.state[shared].items():
                torch.testing.assert_close(
                    value, reference_optimizer.state[reference][key], rtol=0, atol=0
                )
        self.assertEqual(len(adamw.state), 1)
        self.assertIs(model.embedding.weight, model.head_out.weight)

    def test_frozen_loader_preserves_sharing_logits_and_attention_features(self):
        model = self.tiny(True)
        tokens = torch.tensor([[0, 4, 5, 6, 2], [0, 7, 8, 2, 1]])
        mask = tokens != 1
        expected = model(tokens, mask, output_attentions=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tied.pt"
            torch.save(
                {"model_config": model.config.to_dict(), "model": model.state_dict()}, path
            )
            loaded, saved = load_checkpoint(path, torch.device("cpu"))
        self.assertIs(loaded.embedding.weight, loaded.head_out.weight)
        self.assertEqual(
            saved["model"]["embedding.weight"].data_ptr(),
            saved["model"]["head_out.weight"].data_ptr(),
        )
        self.assertEqual(count_parameters(loaded), expected_parameter_count(model.config))
        actual = loaded(tokens, mask, output_attentions=True)
        torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)
        self.assertEqual(len(actual["attentions"]), 2)
        for a, b in zip(actual["attentions"], expected["attentions"], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
