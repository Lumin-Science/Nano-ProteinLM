#!/usr/bin/env python3
"""Generate the measured comparison and two worked recipe illustrations.

Run from an isolated plotting environment with matplotlib >= 3.9, < 4:
    python .dev/scripts/plot_best_recipe_explainer.py

No training dependencies or remote jobs are used or changed.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/figures/best-recipe"
SOURCE = ROOT / ".dev/reports/fir-r02-rope10k-100k-20260906/results.json"
INK, MUTED, TEAL, BLUE = "#223247", "#63758a", "#007f70", "#4676b8"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#d4dce4",
        "svg.hashsalt": "nano-best-recipe",
        "svg.fonttype": "none",
    }
)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(
        OUT / f"{name}.svg", facecolor="white", bbox_inches="tight", metadata={"Date": None}
    )
    plt.close(fig)


def measured_results():
    packet = json.loads(SOURCE.read_text())
    labels = [
        "ESMC-like AdamW baseline",
        "Previous R02 · RoPE20k",
        "1 · R02 RoPE10k",
        "2 · + batch balance",
        "3 · + sqrt loss",
        "4 · + tied embeddings",
    ]
    records = [
        packet["historical_references"]["default"],
        packet["historical_references"]["r02"],
        *[
            packet["completed"][m]
            for m in ("r02_rope10k", "r04_batchbalance", "r10_sqrtloss", "r29_tied")
        ],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.985, top=0.76, bottom=0.18, wspace=0.19)
    fig.suptitle("Setting 3 leads the completed 100k-step comparison", y=0.98, fontsize=16)
    fig.text(
        0.25,
        0.865,
        "4 H100s · batch 1,024 · 102.4M sequences · one training seed per recipe",
        color=MUTED,
        fontsize=10,
    )
    y = np.arange(6)[::-1]
    for ax in axes:
        ax.axhspan(y[4] - 0.4, y[4] + 0.4, color="#e5f3ef", zorder=0)
        ax.grid(axis="x", color="#e9eef2", zorder=0)
        ax.set_ylim(-0.6, 5.6)
        ax.tick_params(axis="y", length=0)
    for i, r in enumerate(records):
        color = TEAL if i == 4 else BLUE if i == 0 else MUTED
        loss = r["validation"]["sequence_mean_nll"]
        p = 100 * r["p_at_l"]
        lo, hi = np.array(r["p_at_l_uncertainty"]["confidence_interval_95"]) * 100
        axes[0].plot(loss, y[i], "o", color=color, markersize=7, zorder=3)
        axes[0].text(loss + 0.0018, y[i], f"{loss:.5f}", va="center", color=color, fontsize=9)
        axes[1].errorbar(
            p,
            y[i],
            xerr=[[p - lo], [hi - p]],
            fmt="o",
            color=color,
            capsize=3,
            markersize=7,
            zorder=3,
        )
        axes[1].text(hi + 0.18, y[i], f"{p:.3f}%", va="center", color=color, fontsize=9)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlim(2.414, 2.489)
    axes[0].set_xlabel("Validation sequence-mean NLL ↓")
    axes[0].set_title("4,096 held-out sequences", fontsize=11, loc="left", pad=12)
    axes[1].set_xlim(25.7, 34.6)
    axes[1].set_xlabel("Long-range contact P@L (%) ↑")
    axes[1].set_title("20,775 chains · 95% bootstrap CI", fontsize=11, loc="left", pad=12)
    fig.text(
        0.25,
        0.025,
        "Axes are zoomed. Contact intervals resample chains; "
        "they do not measure training-seed variation.",
        fontsize=9,
        color=MUTED,
    )
    save(fig, "scaleup-results")
    return [
        {
            "recipe": label,
            "validation_loss": r["validation"]["sequence_mean_nll"],
            "p_at_l": r["p_at_l"],
            "p_at_l_ci95": r["p_at_l_uncertainty"]["confidence_interval_95"],
            "training_seconds": r["training_seconds"],
        }
        for label, r in zip(labels, records, strict=True)
    ]


def balancing_example():
    # Execute the actual pure-Python partitioner without importing Torch into
    # this plotting-only environment. This is repository code, not a copy.
    source = (ROOT / "src/nanoprotein/batch_balance.py").read_text()
    tree = ast.parse(source)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "balanced_partitions"
    )
    namespace = {}
    exec(
        "import heapq\nfrom collections.abc import Sequence\n"
        + ast.get_source_segment(source, node),
        namespace,
    )
    lengths = [512, 480, 448, 416, 128, 96, 64, 32]
    before = [list(range(i * 2, i * 2 + 2)) for i in range(4)]
    after = namespace["balanced_partitions"](lengths, 4)
    assert after == [[0, 7], [1, 6], [2, 5], [3, 4]]
    assert [sum(lengths[j] for j in group) for group in before] == [992, 864, 224, 96]
    assert [sum(lengths[j] for j in group) for group in after] == [544] * 4
    colors = [
        "#225b94",
        "#3874ac",
        "#528aba",
        "#719fc8",
        "#008777",
        "#259c86",
        "#4db39b",
        "#7bcbb4",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.74, bottom=0.30, wspace=0.16)
    fig.suptitle("Batch balance: the same proteins, more even GPU loads", y=0.98, fontsize=16)
    fig.text(
        0.075,
        0.87,
        "Illustration: 8 already-masked examples · 4 GPUs · 2 examples per GPU",
        color=MUTED,
    )
    for ax, groups, title in zip(
        axes,
        [before, after],
        ["Before: original rank assignment", "After: longest-first assignment"],
        strict=True,
    ):
        for rank, group in enumerate(groups):
            left = 0
            for index in group:
                ax.barh(
                    3 - rank,
                    lengths[index],
                    left=left,
                    height=0.59,
                    color=colors[index],
                    edgecolor="white",
                )
                ax.text(
                    left + lengths[index] / 2,
                    3 - rank,
                    chr(65 + index),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                    weight="bold",
                )
                left += lengths[index]
            ax.text(left + 13, 3 - rank, str(left), va="center", fontsize=10, weight="bold")
        ax.axvline(544, color=TEAL, linestyle="--", alpha=0.65, linewidth=1)
        ax.set_xlim(0, 1120)
        ax.set_yticks([3, 2, 1, 0], [f"GPU {i}" for i in range(4)])
        ax.set_title(title, loc="left", fontsize=11, pad=12)
        ax.set_xlabel("Non-padding model tokens, including BOS/EOS")
        ax.grid(axis="x", color="#e9eef2")
        ax.set_axisbelow(True)
    fig.legend(
        [Patch(color=c) for c in colors],
        [f"{chr(65 + i)}: {n}" for i, n in enumerate(lengths)],
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.085),
        frameon=False,
    )
    fig.text(
        0.075,
        0.015,
        "Total stays 2,176 tokens. Maximum rank load: 992 → 544. "
        "This is an illustration, not a measured speedup.",
        color=MUTED,
        fontsize=9,
    )
    save(fig, "batch-balance-example")
    return {"lengths": lengths, "before": before, "after": after, "total_tokens": sum(lengths)}


def loss_example():
    counts = np.array([4, 16, 64], dtype=float)
    losses = np.array([4, 3, 2], dtype=float)
    shares = np.array(
        [np.ones(3) / 3, np.sqrt(counts) / np.sqrt(counts).sum(), counts / counts.sum()]
    )
    objectives = shares @ losses
    assert np.allclose(objectives, [3, 18 / 7, 16 / 7])
    colors = [BLUE, TEAL, "#adb6c1"]
    labels = [
        "Equal proteins (baseline)",
        "√target count (setting 3)",
        "Equal targets (reference only)",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.075, right=0.97, top=0.72, bottom=0.22, wspace=0.65)
    fig.suptitle("Sqrt loss changes how much each protein contributes", y=0.98, fontsize=16)
    fig.text(
        0.075,
        0.87,
        "Worked example: A has 4 targets / loss 4; B has 16 / loss 3; C has 64 / loss 2",
        color=MUTED,
    )
    x = np.arange(3)
    for i, (weights, color, label) in enumerate(zip(shares, colors, labels, strict=True)):
        bars = axes[0].bar(x + (i - 1) * 0.25, weights, width=0.23, color=color, label=label)
        axes[0].bar_label(
            bars, labels=[f"{100 * w:.1f}%" for w in weights], fontsize=8, padding=3
        )
    axes[0].set_xticks(
        x, ["Protein A\n4 targets", "Protein B\n16 targets", "Protein C\n64 targets"]
    )
    axes[0].set_ylim(0, 0.9)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Share of microstep objective")
    axes[0].set_title(
        "Total coefficient on each protein's mean loss", loc="left", fontsize=11, pad=12
    )
    axes[0].grid(axis="y", color="#e9eef2")
    axes[0].set_axisbelow(True)
    axes[1].barh([2, 1, 0], objectives, color=colors, height=0.56)
    for y, v in zip([2, 1, 0], objectives, strict=True):
        axes[1].text(v + 0.06, y, f"{v:.4f}", va="center", fontsize=10)
    axes[1].set_yticks([2, 1, 0], ["Equal proteins", "√target count", "Equal targets"])
    axes[1].set_xlim(0, 3.65)
    axes[1].set_xlabel("Weighted mean loss in this example")
    axes[1].set_title("Same predictions, different objective", loc="left", fontsize=11, pad=12)
    handles = [Patch(color=c, label=label) for c, label in zip(colors, labels, strict=True)]
    fig.legend(
        handles=handles,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.055),
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.075,
        0.012,
        "These are constructed numbers, not measured validation results. "
        "Validation remains an equal-protein mean for every recipe.",
        color=MUTED,
        fontsize=9,
    )
    save(fig, "sqrt-loss-example")
    return {
        "target_counts": counts.tolist(),
        "per_sequence_losses": losses.tolist(),
        "shares": shares.tolist(),
        "objectives": objectives.tolist(),
    }


if __name__ == "__main__":
    data = {
        "result_source": str(SOURCE.relative_to(ROOT)),
        "result_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "measured": measured_results(),
        "illustrative_batch_balance": balancing_example(),
        "illustrative_sqrt_loss": loss_example(),
    }
    (OUT / "figure-data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote three PNG/SVG figures and figure-data.json to {OUT}")
