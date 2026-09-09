"""Portable replicated-DDP checkpoints and explicit data-stream continuation."""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def capture_runtime(batchers: dict[str, Any], *, data_seed: int) -> list[dict[str, Any]]:
    """Called by every rank at an optimizer boundary; model/optimizer stay replicated."""
    state = {
        "data_seed": data_seed,
        "batchers": {name: batcher.state_dict() for name, batcher in batchers.items()},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if not dist.is_initialized():
        return [state]
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, state)
    return states


def restore_runtime(state: dict[str, Any], batchers: dict[str, Any]) -> None:
    if state["batchers"].keys() != batchers.keys():
        raise ValueError("checkpoint stage names differ")
    for name, batcher in batchers.items():
        batcher.load_state_dict(state["batchers"][name])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def validate_resume(
    packet: dict[str, Any],
    config: dict[str, Any],
    *,
    world_size: int,
    data_manifest_sha256: str,
) -> None:
    """Allow extending the budget and changing GPU layout, preserving the recipe."""
    if packet.get("data_manifest_sha256") != data_manifest_sha256:
        raise ValueError("resume requires a matching checkpoint data-manifest hash")
    old = copy.deepcopy(packet["train_config"])
    new = copy.deepcopy(config)
    # Historical checkpoints implicitly allowed sampler wraps. They can still
    # continue when the caller explicitly chooses that same legacy behavior.
    old.setdefault("data_resampling", "allow")
    new.setdefault("data_resampling", old["data_resampling"])
    old_world = int(packet["world_size"])
    if len(old["stages"]) != len(new["stages"]):
        raise ValueError("resume cannot change the stage schedule")
    for previous, current in zip(old["stages"], new["stages"], strict=True):
        old_batch = (
            previous.pop("micro_batch_size") * previous.pop("gradient_accumulation") * old_world
        )
        new_batch = (
            current.pop("micro_batch_size") * current.pop("gradient_accumulation") * world_size
        )
        if old_batch != new_batch:
            raise ValueError("resume must preserve the global sequence batch size")
    for key in (
        "max_steps",
        "max_model_tokens",
        "schedule_steps",
        "walltime_seconds",
        "log_interval",
        "checkpoint_interval",
        "expected_world_size",
        "periodic_evaluation_interval",
        "periodic_evaluation_command",
        "data_capacity_headroom",
    ):
        old.pop(key, None)
        new.pop(key, None)
    if old != new:
        raise ValueError("resume config changes the model, optimizer, or data/loss recipe")
    step = int(packet["optimizer_step"])
    step_limit = config.get("max_steps")
    token_limit = config.get("max_model_tokens")
    if step_limit is None and token_limit is None:
        raise ValueError("resume requires a total max_steps or max_model_tokens endpoint")
    if step_limit is not None and int(step_limit) <= step:
        raise ValueError(
            "resume max_steps is the total endpoint and must exceed the saved step"
        )
    if token_limit is not None and int(token_limit) <= int(packet["model_tokens"]):
        raise ValueError(
            "resume max_model_tokens is the total endpoint and must exceed the saved tokens"
        )
    # Extending a constant Stage-1 schedule does not move warmup or a decay boundary.
    old_schedule = packet["train_config"].get(
        "schedule_steps", packet["train_config"].get("max_steps")
    )
    new_schedule = config.get("schedule_steps", config.get("max_steps"))
    if old_schedule != new_schedule and (
        len(config["stages"]) != 1
        or config["stages"][0]["name"] != "stage1"
        or config.get("stage1_cooldown_fraction", 0) != 0
    ):
        raise ValueError(
            "changing a decaying or multistage schedule requires an explicit new training plan"
        )
