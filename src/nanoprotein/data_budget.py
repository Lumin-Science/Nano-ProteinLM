"""Check source coverage before spending GPU time on a larger training budget."""

from __future__ import annotations

import math
from typing import Any


def data_coverage(
    config: dict[str, Any], manifest: dict[str, Any], *, world_size: int
) -> dict[str, Any]:
    """Expected per-source draws, with staging headroom and a runtime exhaustion guard.

    Headroom is not a probabilistic guarantee: source selection is stochastic.
    The sampler's refusal to wrap provides the actual no-repeat guarantee.
    Counts refer to records within this verified, globally deduplicated release.
    """
    policy = config.get("data_resampling", "error")
    if policy not in {"error", "allow", "per_source"}:
        raise ValueError("data_resampling must be error, allow, or per_source")
    source_policies = config.get("data_source_resampling", {})
    if policy == "per_source" and (
        config.get("data_sampler") != "global"
        or set(source_policies) != {s for stage in config["stages"] for s in stage["mixture"]}
        or any(p not in {"error", "allow"} for p in source_policies.values())
    ):
        raise ValueError("per_source policy requires a global sampler and every source policy")
    headroom = float(config.get("data_capacity_headroom", 0.01))
    if not math.isfinite(headroom) or headroom < 0:
        raise ValueError("data_capacity_headroom must be finite and nonnegative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    stages = config["stages"]
    if not 1 <= len(stages) <= 2 or len({s["name"] for s in stages}) != len(stages):
        raise ValueError("expected one or two uniquely named stages")
    max_steps = config.get("max_steps")
    stage_steps: list[int | None] = [None] * len(stages)
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if len(stages) == 1:
            stage_steps = [max_steps]
        else:
            schedule = int(config.get("schedule_steps", max_steps))
            fraction = float(config.get("stage1_fraction", 2 / 3))
            if schedule <= 0 or not 0 < fraction < 1:
                raise ValueError("invalid two-stage step schedule")
            first = min(max_steps, math.ceil(schedule * fraction))
            stage_steps = [first, max_steps - first]
    expected: dict[str, float] = {}
    stage_budgets = []
    for stage, steps in zip(stages, stage_steps, strict=True):
        batch = (
            int(stage["micro_batch_size"]) * int(stage["gradient_accumulation"]) * world_size
        )
        if batch <= 0:
            raise ValueError("global batch must be positive")
        weights = {k: float(v) for k, v in stage["mixture"].items()}
        if (
            any(not math.isfinite(v) or v < 0 for v in weights.values())
            or sum(weights.values()) <= 0
        ):
            raise ValueError(
                "mixture weights must be finite, nonnegative and have positive sum"
            )
        draws = batch * steps if steps is not None else None
        stage_budgets.append(
            dict(name=stage["name"], steps=steps, global_batch=batch, draws=draws)
        )
        for source, weight in weights.items():
            expected.setdefault(source, 0.0)
            if draws is not None:
                expected[source] += draws * weight / sum(weights.values())
    sources = {}
    insufficient = []
    for source, draws in expected.items():
        records = int(manifest["sources"][source]["train"]["records"])
        required = math.ceil(draws * (1 + headroom) / world_size) * world_size
        enough = records >= required and records >= world_size
        source_policy = source_policies[source] if policy == "per_source" else policy
        sources[source] = {
            "available_records": records,
            "expected_draws": draws if max_steps is not None else None,
            "required_records_with_headroom": required if max_steps is not None else None,
            "expected_exposures": draws / records
            if max_steps is not None and records
            else None,
            "sufficient": enough if max_steps is not None else None,
            **({"resampling": source_policy} if policy == "per_source" else {}),
        }
        if not enough and (policy != "per_source" or source_policy == "error"):
            insufficient.append(f"{source}: {records:,} available, {required:,} required")
    receipt = {
        "protocol": "training-data-coverage-v1",
        "policy": policy,
        "headroom_fraction": headroom,
        "world_size": world_size,
        "stages": stage_budgets,
        "sources": sources,
        "status": "insufficient"
        if insufficient
        else ("passed" if max_steps is not None else "runtime_guard_only"),
        "scope": (
            "per-run records across all ranks and stages; "
            "draws are not unique proteins when resampling is allowed"
        ),
        "runtime_exhaustion_guard": policy in {"error", "per_source"},
    }
    if insufficient and policy in {"error", "per_source"}:
        raise ValueError(
            "training data capacity is insufficient: "
            + "; ".join(insufficient)
            + ". Prepare more verified shards for the actual global-batch x step budget. "
            "Use data_resampling: allow only for an intentional repeated-data experiment."
        )
    return receipt
