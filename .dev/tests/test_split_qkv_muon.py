"""Verify actual aliased updates against independent ordinary PyTorch Muon steps."""

import copy
import io
import unittest

import torch

from nanoprotein.model import ESMCConfig, ESMCForMaskedLM, count_parameters
from nanoprotein.resume import validate_resume
from nanoprotein.split_qkv_muon import SplitQKVMuon
from nanoprotein.train import build_optimizer


class SplitMuonTests(unittest.TestCase):
    def compare_reference(self, device):
        torch.manual_seed(3101)
        for nesterov in [False, True]:
            for adjustment in [None, "match_rms_adamw"]:
                parent = torch.nn.Parameter(torch.randn(24, 8, device=device))
                other = torch.nn.Parameter(torch.randn(8, 8, device=device))
                refs = [torch.nn.Parameter(x.detach().clone()) for x in parent.chunk(3)]
                extra = torch.nn.Parameter(other.detach().clone())
                kw = dict(
                    lr=0.00045,
                    weight_decay=0.0075,
                    momentum=0.95,
                    nesterov=nesterov,
                    ns_steps=5,
                    adjust_lr_fn=adjustment,
                )
                opt = SplitQKVMuon(
                    [{"params": [parent, other], "name": "attention"}],
                    qkv_parameters=[parent],
                    **kw,
                )
                independent = torch.optim.Muon([*refs, extra], **kw)
                for step in range(5):
                    lr = 0.00045 * (step + 1) / 5
                    opt.param_groups[0]["lr"] = independent.param_groups[0]["lr"] = lr
                    parent.grad = None if step == 2 else torch.randn_like(parent)
                    other.grad = torch.randn_like(other)
                    for i, p in enumerate(refs):
                        p.grad = (
                            None if parent.grad is None else parent.grad.chunk(3)[i].clone()
                        )
                    extra.grad = other.grad.clone()
                    before = None if parent.grad is None else parent.grad.clone()
                    opt.step()
                    independent.step()
                    torch.testing.assert_close(parent, torch.cat(refs), rtol=0, atol=0)
                    torch.testing.assert_close(other, extra, rtol=0, atol=0)
                    if before is not None:
                        torch.testing.assert_close(parent.grad, before, rtol=0, atol=0)
                    for view, p in zip(opt.qkv_views[0][1], refs, strict=True):
                        torch.testing.assert_close(
                            opt.state[view]["momentum_buffer"],
                            independent.state[p]["momentum_buffer"],
                            rtol=0,
                            atol=0,
                        )
                    opt.zero_grad(set_to_none=step % 2 == 0)
                    independent.zero_grad(set_to_none=step % 2 == 0)
                    if step % 2 == 0:
                        self.assertIsNone(parent.grad)
                    elif parent.grad is not None:
                        self.assertEqual(parent.grad.count_nonzero(), 0)

    def test_independent_standard_muon_reference_cpu(self):
        self.compare_reference("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_independent_standard_muon_reference_cuda(self):
        self.compare_reference("cuda")

    def test_storage_ownership_validation_and_missing_gradients(self):
        p = torch.nn.Parameter(torch.randn(12, 4))
        q = torch.nn.Parameter(torch.randn(4, 4))
        opt = SplitQKVMuon([{"params": [p, q]}], qkv_parameters=[p], lr=0.01)
        views = opt.qkv_views[0][1]
        self.assertEqual(
            sum(v.numel() for v in opt.param_groups[0]["params"]), p.numel() + q.numel()
        )
        for i, v in enumerate(views):
            self.assertFalse(isinstance(v, torch.nn.Parameter))
            self.assertFalse(v.requires_grad)
            self.assertEqual(v.untyped_storage().data_ptr(), p.untyped_storage().data_ptr())
            self.assertEqual(v.storage_offset(), p.storage_offset() + i * 16)
            self.assertEqual(v.shape, (4, 4))
        before = p.clone()
        opt.step()
        torch.testing.assert_close(p, before, rtol=0, atol=0)
        self.assertEqual(len(opt.state), 0)
        for selected in [[], [p, p], [torch.nn.Parameter(torch.randn(12, 4))]]:
            with self.assertRaises(ValueError):
                SplitQKVMuon([{"params": [p, q]}], qkv_parameters=selected)
        with self.assertRaises(ValueError):
            SplitQKVMuon([{"params": [q]}], qkv_parameters=[q])
        bad = torch.nn.Parameter(torch.randn(4, 12).T)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            SplitQKVMuon([{"params": [bad]}], qkv_parameters=[bad])

    def test_gradient_accumulation_clipping_zeroing_and_closure(self):
        torch.manual_seed(3102)
        p = torch.nn.Parameter(torch.randn(12, 4))
        opt = SplitQKVMuon([{"params": [p]}], qkv_parameters=[p], lr=0.001)
        x = torch.randn(5, 4)
        y = torch.randn(5, 12)

        def loss():
            return ((x @ p.T - y) ** 2).mean()

        loss().backward()
        single = p.grad.clone()
        loss().backward()
        torch.testing.assert_close(p.grad, 2 * single, rtol=0, atol=0)
        torch.nn.utils.clip_grad_norm_([p], 0.1)
        clipped = p.grad.clone()
        opt.step()
        for v, g in zip(opt.qkv_views[0][1], clipped.chunk(3), strict=True):
            torch.testing.assert_close(v.grad, g, rtol=0, atol=0)
        for mode in [True, False, True]:
            opt.zero_grad(set_to_none=mode)
            loss().backward()
            actual = p.grad.clone()
            expected = torch.autograd.grad(loss(), p)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            opt.step()
        calls = []

        def closure():
            calls.append(True)
            opt.zero_grad()
            value = loss()
            value.backward()
            return value

        result = opt.step(closure)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.isfinite(result))

    def test_configuration_default_exact_and_model_training_resume(self):
        torch.manual_seed(3103)
        cfg = ESMCConfig.tiny(
            attention_backend="math",
            transformer_norm="rmsnorm",
        )
        model = ESMCForMaskedLM(cfg)
        old = copy.deepcopy(model)
        new = copy.deepcopy(model)
        recipe = {
            "optimizer": "muon",
            "learning_rate": 0.0005,
            "weight_decay": 0.01,
            "muon_attention_lr_scale": 0.9,
            "muon_ffn_lr_scale": 0.75,
            "muon_adjust_lr_fn": "match_rms_adamw",
            "max_steps": 20,
            "stages": [{"micro_batch_size": 2, "gradient_accumulation": 1}],
        }
        for bad in [1, 0, "true", None]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                build_optimizer(model, recipe | {"muon_split_qkv": bad})
        with self.assertRaisesRegex(ValueError, "requires"):
            build_optimizer(model, recipe | {"optimizer": "adamw", "muon_split_qkv": True})
        opt0 = build_optimizer(model, recipe)
        opt1 = build_optimizer(old, recipe | {"muon_split_qkv": False})
        opt = build_optimizer(new, recipe | {"muon_split_qkv": True})
        self.assertIs(type(opt0.named_optimizers[0][1]), torch.optim.Muon)
        self.assertIs(type(opt.named_optimizers[0][1]), SplitQKVMuon)
        self.assertEqual(count_parameters(model), count_parameters(new))
        self.assertEqual(model.state_dict().keys(), new.state_dict().keys())
        ids = torch.tensor([[0, 4, 32, 7, 2], [0, 32, 5, 2, 1]])
        mask = ids.ne(1)
        for _ in range(3):
            for m, o in [(model, opt0), (old, opt1), (new, opt)]:
                o.zero_grad()
                m(ids, mask)["logits"].square().mean().backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                o.step()
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, old.state_dict()[k], rtol=0, atol=0)
        self.assertGreater(
            (new.blocks[0].attention.qkv.weight - model.blocks[0].attention.qkv.weight).norm(),
            0,
        )
        resumed = ESMCForMaskedLM(cfg)
        resumed.load_state_dict(new.state_dict())
        ropt = build_optimizer(resumed, recipe | {"muon_split_qkv": True})
        buffer = io.BytesIO()
        torch.save(opt.state_dict(), buffer)
        buffer.seek(0)
        ropt.load_state_dict(torch.load(buffer, weights_only=False))
        for m, o in [(new, opt), (resumed, ropt)]:
            o.zero_grad()
            m(ids, mask)["logits"].square().mean().backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            o.step()
        for k, v in new.state_dict().items():
            torch.testing.assert_close(v, resumed.state_dict()[k], rtol=0, atol=0)
        for parent, views in opt.named_optimizers[0][1].qkv_views:
            self.assertIsNotNone(parent.grad)
            self.assertTrue(
                all(
                    torch.isfinite(opt.named_optimizers[0][1].state[v]["momentum_buffer"]).all()
                    for v in views
                )
            )
        saved = recipe | {"muon_split_qkv": True}
        packet = {
            "train_config": saved,
            "world_size": 1,
            "optimizer_step": 3,
            "data_manifest_sha256": "same",
        }
        validate_resume(packet, saved, world_size=1, data_manifest_sha256="same")
        with self.assertRaisesRegex(ValueError, "recipe"):
            validate_resume(packet, recipe, world_size=1, data_manifest_sha256="same")


if __name__ == "__main__":
    unittest.main()
