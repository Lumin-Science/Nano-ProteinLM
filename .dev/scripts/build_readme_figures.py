"""Render README comparisons from curated evidence; requires NumPy and Matplotlib."""

import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
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


def trailing_mean(values, window=100):
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


fig, performance = plt.subplots(figsize=(10.8, 5.2))
fig.subplots_adjust(left=0.255, right=0.95, top=0.79, bottom=0.20)
fig.suptitle(
    "Released protein models on our evaluation split",
    x=0.035,
    ha="left",
    y=0.97,
    fontsize=18,
    fontweight="bold",
)
fig.text(
    0.035,
    0.895,
    "Single-sequence contact prediction · same frozen probe protocol · 20,775 chains",
    fontsize=11,
    color="#475569",
)
rows = sorted(data["models"], key=lambda r: r["p_at_l"], reverse=True)
y_positions = np.arange(len(rows))
for i, row in enumerate(rows):
    ours = row["model"].startswith("Nano")
    y = row["p_at_l"] * 100
    interval = row["p_at_l_95_ci"]
    error = None if interval is None else [[y - interval[0] * 100], [interval[1] * 100 - y]]
    performance.barh(i, y, height=0.52, color=TEAL if ours else BLUE, alpha=0.85)
    performance.errorbar(y, i, xerr=error, fmt="none", ecolor="#0f172a", capsize=4)
    performance.text(y + 1.2, i, f"{y:.2f}%", va="center", fontsize=10, fontweight="bold")
labels = [
    "Auto Research Best\n171M · 400k S1 + 300k S2"
    if r["model"].startswith("Nano")
    else r["model"]
    for r in rows
]
performance.set_yticks(y_positions, labels)
performance.set_ylim(len(rows) - 0.5, -0.5)
performance.grid(axis="x", color="#e2e8f0", linewidth=0.7)
performance.set_axisbelow(True)
performance.tick_params(length=0, pad=8)
performance.set_xlim(0, 64)
performance.set_xlabel("Contact P@L (%) ↑ · bars show mean, whiskers 95% CI")
fig.text(
    0.035,
    0.067,
    "ESMC full-split CIs were not recovered.",
    fontsize=8.7,
    color="#64748b",
)
fig.text(
    0.035,
    0.028,
    "All available intervals use 5,000 chain-bootstrap replicates.",
    fontsize=8.7,
    color="#64748b",
)
save(fig, "released-model-comparison")

paired = ROOT / ".dev/reports/nibi-paired-unique-b2048-100k-20260909"
curve = json.loads((paired / "learning-curve.json").read_text())
assert curve["global_batch"] == 2048 and curve["target_steps"] == 100000
fig, (contact, training) = plt.subplots(
    1, 2, figsize=(15.5, 6.3), gridspec_kw={"width_ratios": [1.05, 1]}
)
fig.subplots_adjust(left=0.06, right=0.975, top=0.69, bottom=0.23, wspace=0.45)
validation = contact.twinx()
validation.spines["right"].set_visible(True)
validation.tick_params(length=0, pad=8)
fig.suptitle(
    "171M models: Auto Research improves the training recipe",
    x=0.06,
    ha="left",
    y=0.975,
    fontsize=18,
    fontweight="bold",
)
fig.text(
    0.06,
    0.915,
    "Matched 100k updates · batch 2,048 · 48.39B model tokens · four H100s per model",
    fontsize=11,
    color="#475569",
)
recipes = [
    ("baseline", data["matched_plot"]["labels"]["baseline"], ORANGE),
    ("setting3", data["matched_plot"]["labels"]["setting3"], TEAL),
]
fig.legend(
    [Line2D([0], [0], color=color, lw=3) for _, _, color in recipes],
    [label for _, label, _ in recipes],
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.053, 0.863),
    ncol=2,
    fontsize=11,
)
fig.legend(
    [
        Line2D([0], [0], color="#475569", marker="o", lw=2),
        Line2D([0], [0], color="#475569", marker="s", ls="--", lw=2),
    ],
    ["Contact P@L · left axis ↑", "MLM validation loss · right axis ↓"],
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.053, 0.809),
    ncol=2,
    fontsize=10,
)
contact.set_title("Evaluation · every 10k steps", loc="left", fontsize=11, pad=12)
training.set_title(
    "Training loss · raw logs + 1,000-step mean",
    loc="left",
    fontsize=11,
    pad=12,
)
warmup = training.inset_axes([0.49, 0.55, 0.46, 0.35])
warmup.set_facecolor("#f8fafc")
warmup.set_title("Warmup · first 1,000 steps", fontsize=8, pad=6)
training_stats = {}
for key, _label, color in recipes:
    rows = curve["runs"][key]
    assert [r["optimizer_step"] for r in rows] == list(range(10000, 100001, 10000))
    assert all(r["independently_audited"] for r in rows)
    steps = np.array([r["optimizer_step"] for r in rows]) / 1000
    pal = np.array([r["p_at_l"] for r in rows]) * 100
    ci = np.array([r["confidence_interval_95"] for r in rows]) * 100
    loss = np.array([r["validation_loss"] for r in rows])
    contact.fill_between(steps, ci[:, 0], ci[:, 1], color=color, alpha=0.17, linewidth=0)
    contact.plot(steps, pal, "o-", color=color, linewidth=2.2, markersize=4)
    validation.plot(steps, loss, "s--", color=color, linewidth=2, markersize=4)
    for ax, value, text in [
        (contact, pal[-1], f"{pal[-1]:.2f}%"),
        (validation, loss[-1], f"{loss[-1]:.4f}"),
    ]:
        ax.annotate(
            text,
            (100, value),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=color,
            fontweight="bold",
            fontsize=10,
        )
    with gzip.open(paired / "full" / key / "metrics.jsonl.gz", "rt") as handle:
        records = [json.loads(line) for line in handle]
    logged = [r for r in records if r.get("event") == "train"]
    expected_steps = [1, *range(10, 100001, 10)]
    assert [r["optimizer_step"] for r in logged] == expected_steps
    train_steps = np.array(expected_steps)
    raw = np.array([r["loss"] for r in logged])
    assert np.isfinite(raw).all()
    smoothed = trailing_mean(raw)
    visible = train_steps >= 1000
    training.plot(
        train_steps[visible] / 1000,
        raw[visible],
        color=color,
        alpha=0.17,
        linewidth=0.45,
        rasterized=True,
    )
    training.plot(train_steps[visible] / 1000, smoothed[visible], color=color, linewidth=2)
    training.annotate(
        f"{smoothed[-1]:.4f}",
        (100, smoothed[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        color=color,
        fontweight="bold",
        fontsize=10,
    )
    early = train_steps <= 1000
    warmup.plot(train_steps[early], raw[early], color=color, linewidth=1)
    training_stats[key] = {
        "records": len(logged),
        "first_step": int(train_steps[0]),
        "last_step": int(train_steps[-1]),
        "final_raw_sequence_mean_loss": float(raw[-1]),
        "final_100_record_mean": float(smoothed[-1]),
        "minimum_after_warmup": float(raw[visible].min()),
        "maximum_after_warmup": float(raw[visible].max()),
    }
contact.set_xlim(0, 113)
contact.set_ylim(10, 40)
contact.set_ylabel("Contact P@L (%) ↑", labelpad=9)
validation.set_ylim(2.34, 2.62)
validation.set_ylabel("MLM validation loss ↓", labelpad=10)
training.set_xlim(0, 113)
training.set_ylim(2.28, 2.85)
training.set_ylabel("Training MLM loss ↓", labelpad=9)
training.set_xlabel("Optimizer steps")
for ax in (contact, training):
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}k"))
    style(ax)
contact.set_xlabel("Optimizer steps", labelpad=5)
warmup.set_xlim(0, 1000)
warmup.set_ylim(2.6, 4.5)
warmup.set_xticks([0, 500, 1000])
warmup.set_yticks([3, 4])
warmup.tick_params(labelsize=8, length=0, pad=3)
fig.text(
    0.06,
    0.095,
    "One seed per recipe; all ten evaluations are shown. P@L bands: 95% chain-bootstrap "
    "CI (20,775 chains). Validation: 4,096 sequences.",
    fontsize=9,
    color="#64748b",
)
fig.text(
    0.06,
    0.055,
    "Training traces: 10,001 logged rank-0 sequence-mean losses per recipe; warmup is "
    "shown in the inset. ESMC 171M is our AdamW reproduction.",
    fontsize=9,
    color="#64748b",
)
save(fig, "matched-100k-curves")
(OUT / "training-curve-summary.json").write_text(json.dumps(training_stats, indent=2) + "\n")
print(f"Wrote four figures and training summary to {OUT}")
