#!/usr/bin/env python3
"""Read completed training/evaluation records and report seed means and sample SDs.

This utility never launches training, changes a recipe or makes acceptance decisions.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import yaml


def summarize(runs: list[Path], validation_sequences: int) -> dict:
    if len(runs) < 2 or len({path.resolve() for path in runs}) != len(runs):
        raise ValueError("provide at least two distinct completed runs")
    rows = []
    reference_config = None
    seeds = set()
    for root in runs:
        config = yaml.safe_load((root / "config.yaml").read_text())
        seed = config.pop("seed")
        if seed in seeds:
            raise ValueError("each run must have a distinct training seed")
        seeds.add(seed)
        if reference_config is not None and config != reference_config:
            raise ValueError("runs must use the same effective recipe apart from training seed")
        reference_config = config
        completion = json.loads((root / "TRAINING_COMPLETE.json").read_text())
        token_budget = config.get("max_model_tokens")
        if token_budget is not None and (
            completion["stop_reason"] != "max_model_tokens"
            or completion["target_model_tokens"] != token_budget
            or not completion["model_token_budget_reached"]
            or completion["model_tokens"] < token_budget
            or completion["model_token_overrun"] != completion["model_tokens"] - token_budget
            or not 0
            <= completion["model_token_overrun"]
            < completion["last_optimizer_step_model_tokens"]
        ):
            raise ValueError("training did not complete its declared token budget")
        evaluation = json.loads((root / "evaluation/EVALUATION.json").read_text())
        if evaluation["checkpoint_sha256"] != completion["final_checkpoint"]["sha256"]:
            raise ValueError("evaluation does not match the final training checkpoint")
        if evaluation["validation_mlm"]["sequences"] != validation_sequences:
            raise ValueError("incomplete MLM evaluation sample")
        contact = evaluation["contact"]
        if contact["evaluation_chains"] != 20775:
            raise ValueError("incomplete contact evaluation population")
        rows.append(
            {
                "seed": seed,
                "run": str(root.resolve()),
                "validation_loss": evaluation["validation_mlm"]["sequence_mean_nll"],
                "p_at_l": contact["precision_at_l"],
                "p_at_l_95_ci": contact["precision_at_l_uncertainty"],
            }
        )
    metrics = {}
    for name in ("validation_loss", "p_at_l"):
        values = [row[name] for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite {name}")
        metrics[name] = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}
    return {"runs": rows, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--validation-sequences", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.runs, args.validation_sequences)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as handle:
        handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
