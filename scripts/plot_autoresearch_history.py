#!/usr/bin/env python3
"""Plot the published validation-loss history, checking its seed statistics.

Run with a separate plotting environment (the training lock is unchanged):
    python scripts/plot_autoresearch_history.py

Requires matplotlib >= 3.9, < 4. Accepts per-run or per-method TSV data.
Writes PNG and SVG to reports/program2/ by default.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[1]
INK = "#233446"
MUTED = "#64748b"
GRAY = "#94a3b8"
TEAL = "#087f70"
AMBER = "#a96613"
ACCEPTED_CHANGES = {
    "r01_muon": (1, "Muon"),
    "r04_batchbalance": (2, "Batch balance"),
    "r10_sqrtloss": (3, "Sqrt loss"),
    "r22_ffn1536": (4, "FFN 1536*"),
    "r29_tied": (5, "Tied embeddings*"),
}


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
    gain = (means[0] - accepted_means[-1]) / means[0] * 100

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.hashsalt": "nano-protein-autoresearch-history",
        }
    )
    fig, ax = plt.subplots(figsize=(11.8, 5.6), facecolor="white")
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.25, top=0.76)
    fig.text(0.085, 0.92, "Autoresearch progress", fontsize=20, weight="bold")
    fig.text(
        0.98,
        0.92,
        f"−{gain:.2f}% loss",
        ha="right",
        fontsize=18,
        weight="bold",
        color=TEAL,
    )
    fig.text(
        0.085,
        0.858,
        f"171M baseline · 4 × L40S · 1 h per seed · {len(seeds)} seeds",
        fontsize=10,
        color=MUTED,
    )

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#eef1f4", linewidth=0.7)
    ax.errorbar(
        rounds,
        means,
        yerr=deviations,
        fmt="o",
        markersize=3.5,
        markerfacecolor="white",
        markeredgecolor=GRAY,
        ecolor="#cbd5e1",
        elinewidth=0.8,
        capsize=1.8,
        capthick=0.8,
        alpha=0.8,
        zorder=2,
    )
    ax.step(rounds, accepted_means, where="post", color=TEAL, linewidth=2.1, zorder=3)
    for index in [0, *kept]:
        change = ACCEPTED_CHANGES.get(rows[index]["method_id"])
        # R29 inherits the narrower FFN: both changes 4 and 5 are ~142M models.
        color = INK if index == 0 else AMBER if change and change[0] >= 4 else TEAL
        ax.errorbar(
            index,
            means[index],
            yerr=deviations[index],
            fmt="o",
            color=color,
            markersize=12 if change else 5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            elinewidth=1,
            capsize=2,
            capthick=1,
            zorder=4,
        )
        if change:
            ax.text(
                index,
                means[index],
                str(change[0]),
                ha="center",
                va="center",
                color="white",
                weight="bold",
                fontsize=8,
                zorder=5,
            )

    ax.legend(
        [
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="white",
                markeredgecolor=GRAY,
                color="#cbd5e1",
                linestyle="none",
                markersize=4,
            ),
            Line2D([], [], color=TEAL, linewidth=2.1),
        ],
        ["Trials ± SD", "Retained recipe"],
        loc="upper right",
        frameon=False,
        fontsize=9,
        ncols=2,
        columnspacing=1.6,
        borderaxespad=0,
    )
    ax.set_xlim(-0.65, rounds[-1] + 0.65)
    lower = min(mean - sd for mean, sd in zip(means, deviations, strict=True))
    upper = max(mean + sd for mean, sd in zip(means, deviations, strict=True))
    ax.set_ylim(lower - 0.004, upper + 0.004)
    ax.set_xticks(sorted({0, *range(10, rounds[-1], 10), rounds[-1]}))
    ax.tick_params(axis="both", length=0, pad=7, labelsize=9)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.set_xlabel("Research round", labelpad=9)
    ax.set_ylabel("Validation loss ↓", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for position, index in enumerate(kept):
        change = ACCEPTED_CHANGES.get(rows[index]["method_id"])
        if change is None:
            continue
        number, label = change
        color = AMBER if number >= 4 else TEAL
        fig.text(
            0.085 + position * 0.186,
            0.098,
            f"{number}  {label}",
            fontsize=10,
            weight="normal",
            color=color,
        )
    fig.text(
        0.085,
        0.044,
        "0 = AdamW baseline",
        fontsize=8.5,
        color=MUTED,
    )
    fig.text(
        0.98,
        0.044,
        "*4–5 use ~142M models",
        ha="right",
        fontsize=8.5,
        color=AMBER,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, metadata={"Software": "Matplotlib"})
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
        "--input", type=Path, default=ROOT / "reports/program2/runs-through-r38.tsv"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/program2/validation-loss"
    )
    args = parser.parse_args()
    plot_history(args.input, args.output)


if __name__ == "__main__":
    main()
