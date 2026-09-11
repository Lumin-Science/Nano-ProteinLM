"""Explicit full-state Stage 1 to Stage 2 transition with append-only data history."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from .checkpoint_audit import assert_exact
from .data import SOURCES, file_sha256
from .data_budget import data_coverage
from .data_migration import verify_prefix
from .global_sampling import portable_batcher_states, row_state_exposure
from .resume import validate_resume
from .sharded_data import validate_prepared_plan


def validate_transition_config(packet, config):
    """Authorize only a new context/mixture/decay, preserving the trained recipe."""
    old, new = copy.deepcopy(packet["train_config"]), copy.deepcopy(config)
    step = int(packet["optimizer_step"])
    if (
        len(old["stages"]) != 1
        or old["stages"][0]["name"] != "stage1"
        or len(new["stages"]) != 1
        or new["stages"][0]["name"] != "stage2"
        or new.get("schedule_start_step") != step
        or old.get("data_sampler") != "global"
        or new.get("data_sampler") != "global"
        or old.get("data_resampling") != "per_source"
        or new.get("data_resampling") != "per_source"
        or not step < int(new.get("schedule_steps", 0))
        or not step < int(new.get("max_steps", 0)) <= int(new["schedule_steps"])
        or old.get("stage1_cooldown_fraction", 0) != 0
    ):
        raise ValueError("transition requires constant Stage 1 to explicitly bounded Stage 2")
    if set(old["stages"][0]["mixture"]) != set(new["stages"][0]["mixture"]):
        raise ValueError("transition cannot change source identities")
    if new["stages"][0]["context_length"] < old["stages"][0]["context_length"]:
        raise ValueError("Stage 2 context must not shrink")
    for stage in (old["stages"][0], new["stages"][0]):
        for key in ("name", "context_length", "mixture"):
            stage.pop(key)
    for cfg in (old, new):
        for key in (
            "schedule_start_step",
            "max_steps",
            "schedule_steps",
            "walltime_seconds",
            "checkpoint_interval",
            "periodic_evaluation_interval",
            "periodic_evaluation_command",
            "log_interval",
            "data_capacity_headroom",
        ):
            cfg.pop(key, None)
    if old != new:
        raise ValueError(
            "transition changes model, optimizer, batch size, or another protected recipe field"
        )
    if int(config.get("expected_world_size", packet["world_size"])) != int(
        packet["world_size"]
    ):
        raise ValueError("transition keeps GPU layout; repartition a later ordinary resume")


def transition_runtime(packet, config, new_sizes, provenance):
    """Keep RNG/accounting; replace only mixture, stage label and expanded queues."""
    validate_transition_config(packet, config)
    states = portable_batcher_states(packet)
    old_name = next(iter(states))
    stage = config["stages"][0]
    first = states[old_name]
    weights = stage["mixture"]
    probabilities = [float(weights[name]) / sum(weights.values()) for name in first["names"]]
    runtime = copy.deepcopy(packet["runtime_states"])
    for rank_state in runtime:
        batcher = rank_state["batchers"].pop(old_name)
        batcher["probabilities"] = probabilities
        batcher["migration"] = copy.deepcopy(provenance)
        for source, previous in list(batcher["samplers"].items()):
            size = int(new_sizes[source])
            if size < previous["size"]:
                raise ValueError("transition cannot shrink a source")
            if size != previous["size"]:
                if previous["epoch"] != 0 or previous["allow_resampling"]:
                    raise ValueError(
                        "expansion requires a strict source before its first repeat"
                    )
                batcher["samplers"][source] = dict(
                    size=size,
                    seed=previous["seed"],
                    allow_resampling=False,
                    epoch=0,
                    cursor=0,
                    origin={
                        "protocol": "global-unseen-origin-v1",
                        "state": copy.deepcopy(previous),
                    },
                )
            assert (
                row_state_exposure(batcher["samplers"][source])["draws"]
                == row_state_exposure(previous)["draws"]
            )
        rank_state["batchers"][stage["name"]] = batcher
    check = dict(packet, runtime_states=runtime)
    portable_batcher_states(check)
    return runtime


def create_transition(checkpoint, config, old_root, new_root, output):
    from .train import validate_data_manifest

    if output.exists():
        raise FileExistsError(output)
    packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_transition_config(packet, config)
    old, new = validate_data_manifest(old_root), validate_data_manifest(new_root)
    if packet["data_manifest_sha256"] != file_sha256(old_root / "manifest.json"):
        raise ValueError("parent checkpoint does not bind the old corpus")
    if old["release_manifest_sha256"] != new["release_manifest_sha256"]:
        raise ValueError("transition requires the same immutable release")
    if old["decontamination"] != new["decontamination"]:
        raise ValueError("transition cannot alter the decontamination contract")
    for root in (old_root, new_root):
        validate_prepared_plan(json.loads((root / "download-plan.json").read_text()), root)
    proof = {}
    original_state = next(iter(portable_batcher_states(packet).values()))
    for source in SOURCES:
        if old["sources"][source]["validation"] != new["sources"][source]["validation"]:
            raise ValueError("transition cannot change validation data")
        proof[source] = verify_prefix(old_root / source / "train", new_root / source / "train")
        if proof[source]["old_records"] != original_state["samplers"][source]["size"]:
            raise ValueError("parent sampler size differs from its prepared source")
    provenance = dict(
        protocol="stage1-to-stage2-full-state-v1",
        parent_checkpoint=str(checkpoint),
        parent_checkpoint_sha256=file_sha256(checkpoint),
        parent_step=packet["optimizer_step"],
        old_manifest_sha256=packet["data_manifest_sha256"],
        new_manifest_sha256=file_sha256(new_root / "manifest.json"),
        prefix_verification=proof,
        validation_unchanged=True,
        schedule_start_step=config["schedule_start_step"],
        schedule_steps=config["schedule_steps"],
    )
    runtime = transition_runtime(
        packet, config, {s: new["sources"][s]["train"]["records"] for s in SOURCES}, provenance
    )
    transformed = dict(
        packet,
        train_config=copy.deepcopy(config),
        stage="stage2",
        data_manifest_sha256=provenance["new_manifest_sha256"],
        runtime_states=runtime,
        stage_transition=provenance,
    )
    exposure = {
        s: row_state_exposure(state)
        for s, state in runtime[0]["batchers"]["stage2"]["samplers"].items()
    }
    coverage = data_coverage(
        config,
        new,
        world_size=packet["world_size"],
        resume_step=packet["optimizer_step"],
        source_exposure=exposure,
    )
    validate_resume(
        transformed,
        config,
        world_size=packet["world_size"],
        data_manifest_sha256=provenance["new_manifest_sha256"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial")
    torch.save(transformed, partial)
    saved = torch.load(partial, map_location="cpu", weights_only=False)
    for key in (
        "model",
        "optimizer",
        "optimizer_step",
        "sequences_seen",
        "model_tokens",
        "filled_residues",
        "training_seconds",
        "model_config",
        "optimizer_layout",
    ):
        assert_exact(packet[key], saved[key])
    for before, after in zip(packet["runtime_states"], saved["runtime_states"], strict=True):
        for key in ("torch_rng", "cuda_rng"):
            assert_exact(before[key], after[key])
        import numpy as np

        np.testing.assert_equal(before["numpy_rng"], after["numpy_rng"])
        assert_exact(before["python_rng"], after["python_rng"])
        assert_exact(before["data_seed"], after["data_seed"])
    portable_batcher_states(saved)
    partial.replace(output)
    result = dict(
        provenance,
        status="passed",
        transition_checkpoint=str(output),
        transition_checkpoint_sha256=file_sha256(output),
        model_optimizer_and_counters_exact=True,
        source_exposure_preserved=exposure,
        data_coverage=coverage,
        verified_utc=datetime.now(timezone.utc).isoformat(),
    )
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "old-data-root", "data-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = create_transition(
        args.checkpoint,
        yaml.safe_load(args.config.read_text()),
        args.old_data_root,
        args.data_root,
        args.output,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
