#!/usr/bin/env python3
"""Fail-fast qualification of the uv-locked CUDA training environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from nano_protein.model import _varlen_flash_attention, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpus", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"expected Python 3.11, found {sys.version.split()[0]}")
    if torch.__version__.split("+")[0] != "2.13.0":
        raise RuntimeError(f"expected Torch 2.13.0, found {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != args.require_gpus:
        raise RuntimeError(
            f"expected {args.require_gpus} visible GPUs, found {torch.cuda.device_count()}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the visible GPUs do not support bfloat16")

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(20260821)
    query = torch.randn(
        2, 2, 11, 64, generator=generator, device=device, dtype=torch.bfloat16
    ).requires_grad_()
    key = torch.randn(
        2, 2, 11, 64, generator=generator, device=device, dtype=torch.bfloat16
    ).requires_grad_()
    value = torch.randn(
        2, 2, 11, 64, generator=generator, device=device, dtype=torch.bfloat16
    ).requires_grad_()
    attention_mask = torch.tensor([[True] * 11, [True] * 5 + [False] * 6], device=device)
    output = _varlen_flash_attention(query, key, value, attention_mask)
    output.float().sum().backward()
    if not torch.isfinite(query.grad).all():
        raise RuntimeError("variable-length FlashAttention produced non-finite gradients")
    if output[1, :, 5:].count_nonzero().item() != 0:
        raise RuntimeError("variable-length FlashAttention wrote into padded positions")

    model = build_model("tiny").to(device).train()
    input_ids = torch.randint(4, 24, (2, 17), generator=generator, device=device)
    model_mask = torch.tensor([[True] * 17, [True] * 9 + [False] * 8], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(input_ids, model_mask)
        loss = result["logits"][model_mask].float().square().mean()
    loss.backward()
    if not torch.isfinite(model.embedding.weight.grad).all():
        raise RuntimeError("packed transformer produced non-finite gradients")

    receipt = {
        "status": "qualified",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "visible_gpus": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(device),
        "attention": "aten::_flash_attention_forward(varlen)",
        "forward_backward": True,
        "packed_transformer_forward_backward": True,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
