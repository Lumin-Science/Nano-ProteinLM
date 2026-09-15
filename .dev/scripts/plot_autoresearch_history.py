#!/usr/bin/env python3
"""Plot the published validation-loss history, checking its seed statistics.

Run with a separate plotting environment (the training lock is unchanged):
    python .dev/scripts/plot_autoresearch_history.py

Requires matplotlib >= 3.9, < 4. Accepts per-run or per-method TSV data.
Writes PNG and SVG to .dev/reports/readme-figures-20260914/ by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[2]


def summarize_runs(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    """Convert a run log to method rows without changing its recorded decisions."""
    groups: dict[str, list[dict[str, str]]] = {}
    run_ids: set[str] = set()
    for row in rows:
        if row["run_id"] in run_ids:
            raise ValueError(f"Duplicate run: {row['run_id']}")
        run_ids.add(row["run_id"])
        groups.setdefault(row["method_id"], []).append(row)
    if not groups:
        raise ValueError("Run log is empty")
    seed_ids = sorted({row["seed"] for row in rows}, key=int)
    seeds = [f"seed{seed}" for seed in seed_ids]

    methods = []
    incumbent_id = ""
    for method_id, runs in groups.items():
        if len(runs) != len(seeds) or {row["seed"] for row in runs} != set(seed_ids):
            raise ValueError(f"Expected one run per matched seed for {method_id}")
        first = runs[0]
        for row in runs:
            if int(row["n_runs"]) != len(runs):
                raise ValueError(f"Incorrect run count for {method_id}")
            if row["status"] != first["status"]:
                raise ValueError(f"Inconsistent decisions for {method_id}")
            for field in ["validation_loss_mean", "validation_loss_std", "delta"]:
                if not math.isclose(
                    float(row[field]), float(first[field]), rel_tol=0, abs_tol=1e-12
                ):
                    raise ValueError(f"Inconsistent {field} for {method_id}")
        methods.append(
            {
                "method_id": method_id,
                "incumbent_before": incumbent_id,
                "mean": first["validation_loss_mean"],
                "sample_sd": first["validation_loss_std"],
                "delta": first["delta"],
                "decision": first["status"],
                **{f"seed{row['seed']}": row["validation_loss"] for row in runs},
            }
        )
        if first["status"] in {"baseline", "keep"}:
            incumbent_id = method_id
    return methods, seeds


def load_history(path: Path) -> tuple[list[dict[str, str]], list[str], list[float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        seeds = [name for name in reader.fieldnames or [] if re.fullmatch(r"seed\d+", name)]
        rows = list(reader)
        if "validation_loss" in (reader.fieldnames or []):
            rows, seeds = summarize_runs(rows)
    if len(seeds) < 2 or not rows or rows[0]["method_id"] != "baseline":
        raise ValueError("Expected a baseline followed by rounds with at least two seeds")

    incumbent = rows[0]
    accepted_means = []
    for index, row in enumerate(rows):
        losses = [float(row[seed]) for seed in seeds]
        mean, sd = float(row["mean"]), float(row["sample_sd"])
        if not all(math.isfinite(value) for value in [*losses, mean, sd]):
            raise ValueError(f"Non-finite loss for {row['method_id']}")
        if not math.isclose(mean, statistics.mean(losses), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Mean does not match seed losses for {row['method_id']}")
        if not math.isclose(sd, statistics.stdev(losses), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"SD does not match seed losses for {row['method_id']}")
        delta = 0.0 if index == 0 else float(incumbent["mean"]) - mean
        if not math.isclose(float(row["delta"]), delta, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Gain does not match the incumbent for {row['method_id']}")
        if index == 0:
            if row["decision"] != "baseline":
                raise ValueError("The first method must have the baseline decision")
        else:
            if not row["method_id"].startswith(f"r{index:02d}_"):
                raise ValueError("Expected consecutive, ordered rounds")
            if row["incumbent_before"] != incumbent["method_id"]:
                raise ValueError(f"Unexpected incumbent for {row['method_id']}")
            decision = "keep" if float(incumbent["mean"]) - mean > sd else "discard"
            if row["decision"] != decision:
                raise ValueError(
                    f"Decision does not match the acceptance rule: {row['method_id']}"
                )
            if decision == "keep":
                incumbent = row
        accepted_means.append(float(incumbent["mean"]))
    return rows, seeds, accepted_means


def plot_history(source: Path, output: Path) -> None:
    rows, seeds, accepted_means = load_history(source)
    rounds = list(range(len(rows)))
    means = [float(row["mean"]) for row in rows]
    deviations = [float(row["sample_sd"]) for row in rows]
    kept = [index for index, row in enumerate(rows) if row["decision"] == "keep"]
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "svg.hashsalt": "nano-protein-autoresearch-history",
        }
    )
    fig, ax = plt.subplots(figsize=(10, 3.9), layout="constrained")
    ax.set_title("AutoResearch progress", fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    trials = ax.errorbar(
        rounds,
        means,
        yerr=deviations,
        fmt="o",
        color="tab:orange",
        markersize=3.5,
        markerfacecolor="white",
        elinewidth=0.9,
        capsize=2,
        capthick=0.9,
        alpha=0.8,
        label="Trial mean ± SD",
        zorder=2,
    )
    (retained,) = ax.step(
        rounds,
        accepted_means,
        where="post",
        color="tab:blue",
        linewidth=1.8,
        label="Retained recipe",
        zorder=3,
    )
    for number, index in enumerate(kept, start=1):
        ax.annotate(
            str(number),
            (index, accepted_means[index]),
            xytext=(5, -15),
            textcoords="offset points",
            color="tab:blue",
            fontsize=10,
            fontweight="medium",
        )
    ax.legend(
        [trials, retained],
        ["Trial mean ± SD", "Retained recipe"],
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    ax.set_xlim(-0.65, rounds[-1] + 0.65)
    lower = min(mean - sd for mean, sd in zip(means, deviations, strict=True))
    upper = max(mean + sd for mean, sd in zip(means, deviations, strict=True))
    ax.set_ylim(lower - 0.004, upper + 0.004)
    ax.set_xticks(sorted({0, *range(10, rounds[-1], 10), rounds[-1]}))
    ax.tick_params(direction="out", length=3, width=0.8)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.set_xlabel("Research round")
    ax.set_ylabel("Validation loss")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, metadata={"Software": "Matplotlib"})
    svg_path = output.with_suffix(".svg")
    fig.savefig(svg_path, metadata={"Date": None})
    svg_text = "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
    svg_path.write_text(svg_text + "\n")
    plt.close(fig)
    print(f"Verified {len(rows)} methods, {len(seeds)} seeds each, {len(kept)} kept changes.")
    print(f"Wrote {output.with_suffix('.png')} and {output.with_suffix('.svg')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / ".dev/reports/program2/runs-through-r38.tsv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".dev/reports/readme-figures-20260914/validation-loss",
    )
    args = parser.parse_args()
    plot_history(args.input, args.output)


if __name__ == "__main__":
    main()
