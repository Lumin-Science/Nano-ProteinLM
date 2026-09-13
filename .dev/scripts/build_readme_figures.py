"""Render README comparisons from curated evidence; requires NumPy and Matplotlib."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".dev/reports/readme-overview-20260913"
data = json.loads((OUT / "comparison-data.json").read_text())
for relative, digest in data["sources"].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#334155",
        "text.color": "#0f172a",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    }
)
TEAL, BLUE, ORANGE = "#0f8b74", "#4266b2", "#bd612b"


def style(ax):
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)


def save(fig, stem):
    for suffix in ("png", "svg"):
        path = OUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


fig, (performance, tokens) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(left=0.075, right=0.975, top=0.79, bottom=0.22, wspace=0.27)
fig.suptitle(
    "Final model: contact prediction and training budget",
    x=0.075,
    ha="left",
    y=0.97,
    fontsize=18,
    fontweight="bold",
)
fig.text(
    0.075,
    0.895,
    "NanoProteinLM: 400k Stage 1 + 300k Stage 2 updates · batch 2,048",
    fontsize=11,
    color="#475569",
)
offsets = {
    "ESMC-300M": (-75, 8),
    "ESMC-600M": (12, 8),
    "ESMC-6B": (-78, -6),
    "NanoProteinLM-171M": (12, 7),
}
for row in data["models"]:
    ours = row["model"].startswith("Nano")
    y = row["p_at_l"] * 100
    low, high = np.array(row["p_at_l_95_ci"]) * 100
    performance.errorbar(
        row["total_nominal_flops"],
        y,
        yerr=[[y - low], [high - y]],
        fmt="D" if ours else "o",
        color=TEAL if ours else BLUE,
        markersize=8,
        capsize=5,
        elinewidth=1.8,
        zorder=4,
    )
    label = "NanoProteinLM 171M" if ours else row["model"]
    performance.annotate(
        f"{label}\n{y:.2f}%",
        (row["total_nominal_flops"], y),
        xytext=offsets[row["model"]],
        textcoords="offset points",
        fontsize=10,
        color=TEAL if ours else BLUE,
    )
performance.set_xscale("log")
performance.set_xlim(1e21, 5e23)
performance.set_ylim(40, 78)
performance.set_ylabel("Contact P@L (%) ↑")
performance.set_xlabel("Estimated training FLOPs · nominal token budget")
style(performance)
rows = [data["models"][-1], *data["models"][:-1]]
x = np.arange(len(rows))
s1 = np.array([r["stage_nominal_tokens"][0] for r in rows]) / 1e12
s2 = np.array([r["stage_nominal_tokens"][1] for r in rows]) / 1e12
tokens.bar(x, s1, width=0.55, label="Stage 1 · context 512", color="#4266b2")
tokens.bar(x, s2, width=0.55, bottom=s1, label="Stage 2 · context 2,048", color="#98c8e6")
for i, total in enumerate(s1 + s2):
    tokens.text(i, total + 0.16, f"{total:.2f}T", ha="center", fontweight="bold")
tokens.set_xticks(x, ["NanoProteinLM\n171M", "ESMC\n300M", "ESMC\n600M", "ESMC\n6B"])
tokens.set_ylim(0, 7.1)
tokens.set_ylabel("Nominal training tokens (trillions)")
tokens.legend(
    frameon=False,
    fontsize=9,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=2,
    columnspacing=1,
)
style(tokens)
fig.text(
    0.075,
    0.025,
    "95% CI: ESMC paper / NanoProteinLM local evaluation. Budgets use "
    "batch × context × steps; FLOPs follow the paper's formula.",
    color="#64748b",
    fontsize=9,
)
save(fig, "released-model-comparison")

curve_path = ROOT / ".dev/reports/nibi-paired-unique-b2048-100k-20260909/learning-curve.json"
curve = json.loads(curve_path.read_text())
assert curve["global_batch"] == 2048 and curve["target_steps"] == 100000
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
fig.subplots_adjust(left=0.075, right=0.96, top=0.73, bottom=0.19, wspace=0.25)
fig.suptitle(
    "Matched 100k-step verification", x=0.075, ha="left", y=0.97, fontsize=18, fontweight="bold"
)
fig.text(
    0.075,
    0.895,
    "Same batch 2,048 · same 48.39B model tokens · four H100s per model · "
    "10k-step evaluation cadence",
    fontsize=10.5,
    color="#475569",
)
for key, label, color in [
    ("baseline", "ESMC-like AdamW", ORANGE),
    ("setting3", "Setting 3 · best recipe", TEAL),
]:
    rows = curve["runs"][key]
    assert [r["optimizer_step"] for r in rows] == list(range(10000, 100001, 10000))
    assert all(r["independently_audited"] for r in rows)
    steps = np.array([r["optimizer_step"] for r in rows]) / 1000
    pal = np.array([r["p_at_l"] for r in rows]) * 100
    ci = np.array([r["confidence_interval_95"] for r in rows]) * 100
    loss = np.array([r["validation_loss"] for r in rows])
    axes[0].fill_between(steps, ci[:, 0], ci[:, 1], color=color, alpha=0.17, linewidth=0)
    axes[0].plot(steps, pal, "o-", color=color, linewidth=2.2, markersize=4, label=label)
    axes[1].plot(steps, loss, "o-", color=color, linewidth=2.2, markersize=4, label=label)
    axes[0].annotate(
        f"{pal[-1]:.2f}%",
        (100, pal[-1]),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        color=color,
        fontweight="bold",
    )
    axes[1].annotate(
        f"{loss[-1]:.4f}",
        (100, loss[-1]),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        color=color,
        fontweight="bold",
    )
for ax in axes:
    style(ax)
    ax.set_xlim(5, 116)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}k"))
    ax.set_xlabel("Optimizer steps")
axes[0].set_ylabel("Contact P@L (%) ↑")
axes[0].set_ylim(10, 40)
axes[1].set_ylabel("MLM validation loss ↓")
axes[1].set_ylim(2.35, 2.62)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.067, 0.855),
    ncol=2,
    fontsize=11,
)
fig.text(
    0.075,
    0.055,
    "One training seed per recipe. P@L shading: 95% chain-bootstrap CI "
    "over 20,775 chains; validation uses 4,096 held-out sequences.",
    color="#64748b",
    fontsize=9,
)
save(fig, "matched-100k-curves")
print(f"Wrote four figures to {OUT}")
