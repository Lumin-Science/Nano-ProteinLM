"""Evaluate released ESM-2 and Profluent-E1 with the existing frozen contact probe.

Optional dependencies are installed separately. E1 also requires its official
source on PYTHONPATH. This adapter supports single proteins without retrieval.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from nanoprotein.tokenizer import ESMC_SEQUENCE_VOCAB, ProteinTokenizer


def single_sequence_probabilities(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Recover unmasked probabilities from E1's already rotated B,L,H,D Q/K."""
    if query.ndim != 4 or key.ndim != 4 or query.shape[:2] != key.shape[:2]:
        raise ValueError("contact extraction requires equal-length, uncached Q/K")
    if query.shape[0] != 1 or query.shape[2] % key.shape[2]:
        raise ValueError("contact extraction requires one protein and valid GQA heads")
    with torch.autocast(query.device.type, enabled=False):
        q = query.float().transpose(1, 2)
        k = key.float().transpose(1, 2).repeat_interleave(query.shape[2] // key.shape[2], 1)
        return (q @ k.transpose(-1, -2) / query.shape[-1] ** 0.5).softmax(-1)


class ReleasedContactAdapter(nn.Module):
    """Present official attention maps to the unchanged NanoProteinLM evaluator."""

    def __init__(self, family: str, model_dir: Path, device: torch.device):
        super().__init__()
        self.family = family
        self.capture = True
        self.verify_context = False
        self.context_errors: list[float] = []
        self.maps: list[torch.Tensor] = []
        self.processed = 0
        if family == "esm2":
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self.native = (
                AutoModelForMaskedLM.from_pretrained(
                    model_dir, attn_implementation="eager", local_files_only=True
                )
                .to(device)
                .eval()
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        elif family == "e1":
            from E1.batch_preparer import E1BatchPreparer
            from E1.modeling import E1ForMaskedLM

            self.native = E1ForMaskedLM.from_pretrained(model_dir).to(device).eval()
            self.preparer = E1BatchPreparer()
            for layer in self.native.model.layers:
                attention = layer.norm_attn_norm.self_attn
                original = attention._attn

                def capture_attention(*args, _original=original, **kwargs):
                    result = _original(*args, **kwargs)
                    if self.capture:
                        ids = kwargs["sequence_ids"]
                        if ids.shape[0] != 1 or not bool((ids == 0).all()):
                            raise ValueError(
                                "E1 contact adapter excludes padding and retrieval"
                            )
                        probabilities = single_sequence_probabilities(
                            kwargs["query_states"], kwargs["key_states"]
                        )
                        self.maps.append(probabilities)
                        if self.verify_context:
                            with torch.autocast(ids.device.type, enabled=False):
                                value = kwargs["val_states"].float().transpose(1, 2)
                                value = value.repeat_interleave(
                                    probabilities.shape[1] // value.shape[1], 1
                                )
                                expected = (probabilities @ value).transpose(1, 2)
                                expected = expected.reshape_as(result[0])
                                error = (expected - result[0].float()).norm() / expected.norm()
                                self.context_errors.append(float(error))
                    return result

                attention._attn = capture_attention
        else:
            raise ValueError(f"unknown released model: {family}")
        self.config = SimpleNamespace(
            n_layers=self.native.config.num_hidden_layers,
            n_heads=self.native.config.num_attention_heads,
        )
        self.eval()

    def native_forward(self, sequence: str):
        device = next(self.native.parameters()).device
        if self.family == "esm2":
            batch = self.tokenizer(sequence, return_tensors="pt").to(device)
            if batch["input_ids"].shape[1] != len(sequence) + 2:
                raise ValueError("ESM-2 residue tokenization changed")
            result = self.native(**batch, output_attentions=True)
            self.maps = list(result.attentions)
            return result
        batch = self.preparer.get_batch_kwargs([sequence], device=device)
        if batch["input_ids"].shape[1] != len(sequence) + 4:
            raise ValueError("E1 residue tokenization changed")
        self.maps = []
        return self.native(
            **{
                k: batch[k]
                for k in (
                    "input_ids",
                    "within_seq_position_ids",
                    "global_position_ids",
                    "sequence_ids",
                )
            },
            use_cache=False,
            output_attentions=False,
        )

    def forward(self, input_ids, attention_mask, output_attentions=True):
        if not output_attentions or input_ids.shape[0] != 1 or not bool(attention_mask.all()):
            raise ValueError("released contact adapter requires one unpadded protein")
        ids = input_ids[0].tolist()
        if ids[0] != 0 or ids[-1] != 2 or any(i < 4 or i > 30 for i in ids[1:-1]):
            raise ValueError("unexpected contact sequence tokens")
        sequence = "".join(ESMC_SEQUENCE_VOCAB[i] for i in ids[1:-1])
        self.native_forward(sequence)
        if self.family == "e1":
            # E1: BOS, '1', residues, '2', EOS. The frozen evaluator trims one
            # boundary token per side; retain the residue block without rescaling.
            maps = tuple(F.pad(a[:, :, 2:-2, 2:-2], (1, 1, 1, 1)) for a in self.maps)
        else:
            maps = tuple(self.maps)
        self.processed += 1
        if self.processed % 100 == 0:
            print(f"PROGRESS {self.family} chains={self.processed}", flush=True)
        return {"attentions": maps}

    def smoke(self, sequence: str) -> dict:
        self.verify_context = True
        self.capture = True
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            captured = self.native_forward(sequence).logits.clone()
            maps = self.maps
            self.capture = False
            native = self.native_forward(sequence).logits
        self.capture = True
        self.verify_context = False
        if not torch.equal(captured, native):
            raise ValueError("attention capture changed official model logits")
        if len(maps) != self.config.n_layers:
            raise ValueError("missing attention layers")
        for value in maps:
            if not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite attention probabilities")
            if not torch.allclose(
                value.float().sum(-1), torch.ones_like(value[..., 0]).float(), atol=0.01
            ):
                raise ValueError("attention probabilities do not sum to one")
        if self.context_errors and max(self.context_errors) > 0.02:
            raise ValueError("recovered E1 attention disagrees with native context by >2% L2")
        ids, mask = ProteinTokenizer.esmc().encode_batch(
            [sequence], max_length=len(sequence) + 2
        )
        device = next(self.native.parameters()).device
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            exposed = self(ids.to(device), mask.to(device))["attentions"]
        offset = 2 if self.family == "e1" else 1
        for actual, original in zip(exposed, maps, strict=True):
            if not torch.equal(
                actual[:, :, 1:-1, 1:-1], original[:, :, offset:-offset, offset:-offset]
            ):
                raise ValueError(
                    "adapter changed residue attention coordinates or probabilities"
                )
        return {
            "family": self.family,
            "sequence_length": len(sequence),
            "logits_identical_with_capture": True,
            "residue_attention_identity_verified": True,
            "layers": self.config.n_layers,
            "heads": self.config.n_heads,
            "native_context_relative_l2_errors": self.context_errors,
            "actual_parameters": sum(p.numel() for p in self.native.parameters()),
        }


def main():
    from nanoprotein.data import file_sha256
    from nanoprotein.evaluate import fit_contact_probe_receipt, run_contact_lite

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "fit", "score"))
    parser.add_argument("--family", choices=("esm2", "e1"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--external-src", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-receipt", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    model = ReleasedContactAdapter(args.family, args.model_dir, device)
    digest = file_sha256(args.model_dir / "model.safetensors")
    if args.mode == "smoke":
        result = model.smoke("M" + "ACDEFGHIKLMNPQRSTVWY" * 4)
        result["full_length_case"] = model.smoke(("ACDEFGHIKLMNPQRSTVWYXBUZO" * 22)[:510])
    else:
        common = dict(
            checkpoint_sha256=digest,
            dataset_root=args.dataset_root,
            external_src=args.external_src,
            device=device,
        )
        if args.mode == "fit":
            result = fit_contact_probe_receipt(model, **common)
        else:
            contact = run_contact_lite(
                model,
                **common,
                evaluation_chains=20775,
                bootstrap=0,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                probe_receipt=args.probe_receipt,
            )
            result = {"checkpoint_sha256": digest, "contact": contact}
    result["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"COMPLETED {args.family} {args.mode} {args.output}", flush=True)


if __name__ == "__main__":
    main()
