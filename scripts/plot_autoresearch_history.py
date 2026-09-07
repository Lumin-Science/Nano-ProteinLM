#!/usr/bin/env python3
"""Plot the published validation-loss history, checking its seed statistics.

Run with a separate plotting environment (the training lock is unchanged):
    python scripts/plot_autoresearch_history.py

Requires matplotlib >= 3.9, < 4. Writes PNG and SVG beside methods.tsv.
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


def load_history(path: Path) -> tuple[list[dict[str, str]], list[str], list[float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        seeds = [name for name in reader.fieldnames or [] if re.fullmatch(r"seed\d+", name)]
        rows = list(reader)
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
    fig, ax = plt.subplots(figsize=(12.8, 6.5), facecolor="white")
    fig.subplots_adjust(left=0.082, right=0.975, bottom=0.21, top=0.79)
    fig.text(
        0.082,
        0.93,
        f"Validation loss across {len(rows) - 1} autoresearch rounds",
        fontsize=19,
        weight="bold",
    )
    seed_labels = ", ".join(seed.removeprefix("seed") for seed in seeds)
    fig.text(
        0.082,
        0.883,
        f"171M parameter cap · 4 × L40S · 1 hour per seed · seeds {seed_labels}",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.975,
        0.835,
        f"{means[0]:.5f} → {accepted_means[-1]:.5f}  |  {gain:.2f}% lower",
        ha="right",
        fontsize=11,
        weight="bold",
        color=TEAL,
    )

    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e9edf1", linewidth=0.8)
    ax.plot(rounds, means, color=GRAY, alpha=0.6, linewidth=0.85, zorder=2)
    all_methods = ax.errorbar(
        rounds,
        means,
        yerr=deviations,
        fmt="o",
        markersize=3.8,
        markerfacecolor="white",
        markeredgecolor=MUTED,
        ecolor=GRAY,
        elinewidth=0.85,
        capsize=2,
        capthick=0.85,
        zorder=3,
        label="Each recipe: mean ± 1 SD",
    )
    (best_line,) = ax.step(
        rounds,
        accepted_means,
        where="post",
        color=TEAL,
        linewidth=2,
        zorder=4,
        label="Current best accepted",
    )
    for index in [0, *kept]:
        color = INK if index == 0 else TEAL
        ax.errorbar(
            index,
            means[index],
            yerr=deviations[index],
            fmt="o",
            color=color,
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1,
            capsize=2.5,
            capthick=1,
            zorder=5,
        )

    kept_marker = Line2D([], [], color=TEAL, marker="o", linestyle="none", markersize=5)
    ax.legend(
        [all_methods, best_line, kept_marker],
        ["Each recipe: mean ± 1 SD", "Current best accepted", "Kept change"],
        loc="upper right",
        frameon=False,
        fontsize=9,
        ncols=3,
        columnspacing=1.5,
        borderaxespad=0.4,
    )
    ax.set_xlim(-0.65, rounds[-1] + 0.65)
    lower = min(mean - sd for mean, sd in zip(means, deviations, strict=True))
    upper = max(mean + sd for mean, sd in zip(means, deviations, strict=True))
    ax.set_ylim(lower - 0.006, upper + 0.012)
    ax.set_xticks(rounds)
    ax.tick_params(axis="both", length=0, pad=8, labelsize=9)
    for index, label in enumerate(ax.get_xticklabels()):
        if index in kept:
            label.set_color(TEAL)
            label.set_weight("bold")
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.set_xlabel("Autoresearch round (0 = AdamW baseline)", labelpad=13)
    ax.set_ylabel("Held-out MLM loss · lower is better", labelpad=13)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d8e0e7")

    fig.text(
        0.082,
        0.083,
        "Kept: R01 Muon · R04 rank balance · R10 sqrt loss · "
        "R22 FFN 1536 · R29 tied embeddings",
        fontsize=9,
        color=TEAL,
    )
    fig.text(
        0.082,
        0.04,
        "Source: reports/program2/methods.tsv · 2026-09-06 snapshot · "
        f"Sample SD across {len(seeds)} training seeds; 32 fixed validation sequences.",
        fontsize=8.5,
        color=MUTED,
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
    parser.add_argument("--input", type=Path, default=ROOT / "reports/program2/methods.tsv")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/program2/validation-loss"
    )
    args = parser.parse_args()
    plot_history(args.input, args.output)


if __name__ == "__main__":
    main()
