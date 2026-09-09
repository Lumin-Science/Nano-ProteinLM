#!/usr/bin/env python3
"""Validate and summarize one completed AutoResearch round."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def training_losses(path: Path) -> list[float]:
    losses: list[float] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"expected a JSON object at {path}:{line_number}")
        if value.get("event") != "train":
            continue
        loss = float(value["loss"])
        if not math.isfinite(loss):
            raise ValueError(f"non-finite training loss at {path}:{line_number}")
        losses.append(loss)
    if not losses:
        raise ValueError(f"no training-loss records found in {path}")
    return losses


def execution_seconds(path: Path) -> float:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return float(values["evaluation_wall_seconds"])


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    completion = load_json(root / "TRAINING_COMPLETE.json")
    if completion.get("stop_reason") != "walltime":
        raise ValueError(
            "AutoResearch training must stop on wall time; found "
            f"{completion.get('stop_reason')!r}"
        )

    losses = training_losses(root / "metrics.jsonl")
    final_window = losses[-100:]
    validation_report = load_json(root / "eval-validation" / "EVALUATION.json")
    validation = load_json(root / "eval-validation" / "VALIDATION_MLM.json")
    contact = load_json(root / "eval-p-at-l" / "P_AT_L.json")

    validation_timing = validation_report.get("timing_seconds")
    if not isinstance(validation_timing, dict):
        raise TypeError("validation report has no timing_seconds object")
    validation_seconds = sum(float(value) for value in validation_timing.values())
    contact_seconds = execution_seconds(root / "eval-p-at-l" / "EXECUTION.txt")

    summary = {
        "schema_version": 1,
        "status": "complete",
        "p_at_l": float(contact["p_at_l"]),
        "train_loss": fmean(final_window),
        "train_loss_records": len(final_window),
        "validation_loss": float(validation["sequence_mean_nll"]),
        "training_seconds": float(completion["training_seconds"]),
        "actual_steps": int(completion["optimizer_steps"]),
        "model_tokens_M": int(completion["model_tokens"]) / 1_000_000,
        "num_params_M": int(completion["parameter_count"]) / 1_000_000,
        "peak_vram_mb": int(completion["peak_cuda_memory_bytes"]) / (1024 * 1024),
        "evaluation_seconds": validation_seconds + contact_seconds,
        "validation_evaluation_seconds": validation_seconds,
        "p_at_l_evaluation_seconds": contact_seconds,
        "checkpoint": completion["final_checkpoint"],
    }
    write_json(root / "ROUND_SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
