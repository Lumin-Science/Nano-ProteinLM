"""Audit the same-layout data stream and quantify floating-point trajectory drift."""

import json
import sys
from pathlib import Path

import torch

base, resumed, output = map(Path, sys.argv[1:])
a = torch.load(base, map_location="cpu", weights_only=False)
b = torch.load(resumed, map_location="cpu", weights_only=False)
assert a["optimizer_step"] == b["optimizer_step"] == 200
for key in ("model_tokens", "filled_residues", "sequences_seen"):
    assert a[key] == b[key]
for x, y in zip(a["runtime_states"], b["runtime_states"], strict=True):
    assert x["batchers"] == y["batchers"]
    assert torch.equal(x["torch_rng"], y["torch_rng"])
    assert torch.equal(x["cuda_rng"], y["cuda_rng"])
max_abs = max((a["model"][key] - b["model"][key]).abs().max().item() for key in a["model"])
assert all(torch.isfinite(value).all().item() for value in b["model"].values())
# DDP bucket construction can change reduction ordering after process restart.
# Exact sampling and saved-state restoration are required; bitwise weights are reported.
receipt = dict(
    status="passed",
    sampler_and_mask_rng_exact=True,
    model_bitwise_equal=max_abs == 0,
    model_max_absolute_difference=max_abs,
)
output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
