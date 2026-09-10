"""Verify a full portable training checkpoint and optionally restore its optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import file_sha256
from .global_sampling import portable_batcher_states
from .model import build_model, count_parameters


def assert_finite(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all().item():
            raise ValueError("nonfinite checkpoint tensor")
    elif isinstance(value, dict):
        for v in value.values():
            assert_finite(v)
    elif isinstance(value, list | tuple):
        for v in value:
            assert_finite(v)


def assert_exact(a, b):
    if torch.is_tensor(a):
        if not torch.equal(a, b):
            raise ValueError("checkpoint tensor differs after restoration")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("checkpoint state keys differ")
        for k in a:
            assert_exact(a[k], b[k])
    elif isinstance(a, list | tuple):
        if len(a) != len(b):
            raise ValueError("checkpoint state lengths differ")
        for x, y in zip(a, b, strict=True):
            assert_exact(x, y)
    elif a != b:
        raise ValueError("checkpoint state value differs")


def audit_checkpoint(checkpoint: Path, data_root: Path, *, expected_step=None, restore=False):
    from .train import build_optimizer, validate_data_manifest

    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = packet["train_config"]
    manifest = validate_data_manifest(data_root)
    if packet["data_manifest_sha256"] != file_sha256(data_root / "manifest.json"):
        raise ValueError("checkpoint and data manifest differ")
    step = int(packet["optimizer_step"])
    if expected_step is not None and step != expected_step:
        raise ValueError("checkpoint step differs from requested endpoint")
    stages = config["stages"]
    if len(stages) != 1 or config.get("data_sampler") != "global":
        raise ValueError("checkpoint audit requires single-stage global source sampling")
    stage = stages[0]
    batch = stage["micro_batch_size"] * stage["gradient_accumulation"] * packet["world_size"]
    if packet["sequences_seen"] != batch * step:
        raise ValueError("sequence count differs from step budget")
    if packet["optimizer_layout"] != "replicated_ddp_full_state":
        raise ValueError("checkpoint is missing full replicated optimizer state")
    states = portable_batcher_states(packet)
    exposure = {}
    for name, s in states[stage["name"]]["samplers"].items():
        size = manifest["sources"][name]["train"]["records"]
        if s["size"] != size or not 0 <= s["cursor"] <= size:
            raise ValueError("source cursor/size differs from data")
        prior = sum((s.get("origin") or {}).get("cursors", []))
        if s["epoch"] < 0 or (s["epoch"] == 0 and s["cursor"] > size - prior):
            raise ValueError("invalid source epoch/cursor")
        policy = config["data_source_resampling"][name]
        if s["allow_resampling"] != (policy == "allow") or (policy == "error" and s["epoch"]):
            raise ValueError("source violated configured repeat policy")
        draws = prior + s["cursor"] if s["epoch"] == 0 else s["epoch"] * size + s["cursor"]
        exposure[name] = {
            "draws": draws,
            "unique_records_seen": min(draws, size),
            "repeated_draws": max(0, draws - size),
            "epoch": s["epoch"],
        }
    if sum(v["draws"] for v in exposure.values()) != packet["sequences_seen"]:
        raise ValueError("global source history differs from training count")
    assert_finite(packet["model"])
    assert_finite(packet["optimizer"])
    if restore:
        options = {
            k: config[k]
            for k in (
                "learned_residual_routing",
                "transformer_norm",
                "depth_scaled_residual_init",
                "rotary_base",
                "ffn_hidden_dim",
                "tie_word_embeddings",
            )
            if k in config
        }
        model = build_model(config["model"], attention_backend="math", **options)
        model.load_state_dict(packet["model"], strict=True)
        optimizer = build_optimizer(model, config)
        optimizer.load_state_dict(packet["optimizer"])
        assert_exact(packet["model"], model.state_dict())
        assert_exact(packet["optimizer"], optimizer.state_dict())
        if count_parameters(model) != packet["parameter_count"]:
            raise ValueError("restored model size differs")
    return {
        "status": "passed",
        "scope": "checkpoint_at_step",
        "checkpoint_sha256": file_sha256(checkpoint),
        "optimizer_steps": step,
        "sequences_seen": packet["sequences_seen"],
        "model_tokens": packet["model_tokens"],
        "world_size": packet["world_size"],
        "parameter_count": packet["parameter_count"],
        "full_training_complete": step >= config["max_steps"],
        "finite_model_and_optimizer": True,
        "model_and_optimizer_roundtrip_exact": restore,
        "optimizer_layout": packet["optimizer_layout"],
        "source_exposure_global": exposure,
        "global_sampler_state_verified": True,
        "data_manifest_sha256": packet["data_manifest_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--restore-optimizer", action="store_true")
    args = parser.parse_args()
    receipt = audit_checkpoint(
        args.checkpoint,
        args.data_root,
        expected_step=args.expected_step,
        restore=args.restore_optimizer,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
