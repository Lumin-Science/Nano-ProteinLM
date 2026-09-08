"""Render the ten audited baseline checkpoints from learning-curve.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

root = Path(__file__).resolve().parent
data = json.loads((root / "learning-curve.json").read_text())
rows = data["evaluations"]
assert [r["optimizer_steps"] for r in rows] == list(range(10000, 100001, 10000))
assert data["status"] == "complete" and data["global_batch_size"] == 2048
x = np.array([r["optimizer_steps"] / 1000 for r in rows])
loss = np.array([r["validation_loss"] for r in rows])
precision = np.array([r["p_at_l"] * 100 for r in rows])
ci = np.array([r["uncertainty"]["confidence_interval_95"] for r in rows]) * 100
assert np.all(ci[:, 0] <= precision) and np.all(precision <= ci[:, 1])

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#344256",
        "text.color": "#172c43",
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
fig.subplots_adjust(left=0.075, right=0.97, bottom=0.22, top=0.72, wspace=0.23)
fig.text(
    0.075, 0.93, "ESMC-like AdamW baseline · all 10 checkpoints", fontsize=19, weight="bold"
)
fig.text(
    0.075,
    0.87,
    "Nibi · 8 H100s · batch 2,048 · 100k steps · LR 5e−4 · WD 0.01 · warmup 1,000",
    fontsize=11,
    color="#516174",
)
axes[0].plot(x, loss, color="#3465ad", marker="o", markersize=5, linewidth=2)
axes[0].set(title="Validation loss ↓", ylabel="Sequence-mean NLL", ylim=(2.41, 2.61))
axes[1].fill_between(x, ci[:, 0], ci[:, 1], color="#087f83", alpha=0.16, linewidth=0)
axes[1].errorbar(
    x,
    precision,
    yerr=np.array([precision - ci[:, 0], ci[:, 1] - precision]),
    color="#087f83",
    marker="o",
    markersize=5,
    linewidth=2,
    elinewidth=1,
    capsize=3,
)
axes[1].set(title="Long-range contact P@L ↑", ylabel="Precision at L", ylim=(14, 30.5))
axes[1].set_yticks([15, 18, 21, 24, 27, 30])
axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
for ax in axes:
    ax.set(xlabel="Optimizer steps (thousands)", xlim=(6, 104), xticks=x)
    ax.grid(axis="y", color="#e5eaf0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#516174", length=3)
    for spine in ax.spines.values():
        spine.set_color("#ccd5df")
axes[0].annotate(
    f"{loss[-1]:.5f}",
    (x[-1], loss[-1]),
    xytext=(-8, 12),
    textcoords="offset points",
    ha="right",
    color="#3465ad",
    weight="bold",
)
axes[1].annotate(
    f"{precision[-1]:.3f}%",
    (x[-1], precision[-1]),
    xytext=(-8, 13),
    textcoords="offset points",
    ha="right",
    color="#087f83",
    weight="bold",
)
fig.text(
    0.075,
    0.10,
    "Fixed evaluation: 4,096 MLM sequences and 20,775 contact chains. "
    "Points are measured checkpoints.",
    fontsize=10,
    color="#516174",
)
fig.text(
    0.075,
    0.055,
    "P@L band/error bars: 95% chain-bootstrap CI, 5,000 resamples. "
    "One training seed; no validation-loss CI.",
    fontsize=10,
    color="#516174",
)
for suffix in ("png", "svg"):
    fig.savefig(root / f"learning-curve.{suffix}", dpi=180, facecolor="white")
plt.close(fig)
