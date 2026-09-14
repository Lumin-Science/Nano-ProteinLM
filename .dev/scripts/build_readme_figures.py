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


fig, (performance, tokens) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.subplots_adjust(left=0.075, right=0.975, top=0.79, bottom=0.22, wspace=0.27)
fig.suptitle(
    "Released ESMC references on our evaluation split",
    x=0.075,
    ha="left",
    y=0.97,
    fontsize=18,
    fontweight="bold",
)
fig.text(
    0.075,
    0.895,
    "Final Auto Research model: 171M · 400k Stage 1 + 300k Stage 2 updates",
    fontsize=11,
    color="#475569",
)
offsets = {"ESMC-300M": (-75, -31), "ESMC-600M": (-100, 9), "NanoProteinLM-171M": (12, 6)}
for row in data["models"]:
    ours = row["model"].startswith("Nano")
    y = row["p_at_l"] * 100
    interval = row["p_at_l_95_ci"]
    yerr = None
    if interval is not None:
        low, high = np.array(interval) * 100
        yerr = [[y - low], [high - y]]
    performance.errorbar(
        row["total_nominal_flops"],
        y,
        yerr=yerr,
        fmt="D" if ours else "o",
        color=TEAL if ours else BLUE,
        markersize=8,
        capsize=5,
        elinewidth=1.8,
        zorder=4,
    )
    label = "Auto Research Best 171M" if ours else row["model"]
    performance.annotate(
        f"{label}\n{y:.2f}%",
        (row["total_nominal_flops"], y),
        xytext=offsets[row["model"]],
        textcoords="offset points",
        fontsize=10,
        color=TEAL if ours else BLUE,
    )
performance.set_xscale("log")
performance.set_xlim(1.5e21, 4.2e22)
performance.set_ylim(43, 62)
performance.set_ylabel("Contact P@L (%) on our 20,775-chain split ↑")
performance.set_xlabel("Estimated training FLOPs · nominal token budget")
style(performance)
rows = [data["models"][-1], *data["models"][:-1]]
x = np.arange(len(rows))
s1 = np.array([r["stage_nominal_tokens"][0] for r in rows]) / 1e12
s2 = np.array([r["stage_nominal_tokens"][1] for r in rows]) / 1e12
tokens.bar(x, s1, width=0.5, label="Stage 1 · context 512", color=BLUE)
tokens.bar(x, s2, width=0.5, bottom=s1, label="Stage 2 · context 2,048", color="#98c8e6")
for i, total in enumerate(s1 + s2):
    tokens.text(i, total + 0.16, f"{total:.2f}T", ha="center", fontweight="bold")
tokens.set_xticks(x, ["Auto Research\n171M", "ESMC\n300M", "ESMC\n600M"])
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
    0.04,
    "All P@L values use our split. Full-split CIs were not recovered for the released "
    "models; the 171M error bar is a 95% chain-bootstrap CI.",
    color="#64748b",
    fontsize=8.7,
)
fig.text(
    0.075,
    0.008,
    "Token budgets use batch × maximum context × steps; FLOPs follow the paper's "
    "formula on that nominal basis, not measured GPU work.",
    color="#64748b",
    fontsize=8.7,
)
save(fig, "released-model-comparison")

paired = ROOT / ".dev/reports/nibi-paired-unique-b2048-100k-20260909"
curve = json.loads((paired / "learning-curve.json").read_text())
assert curve["global_batch"] == 2048 and curve["target_steps"] == 100000
fig, (contact, training) = plt.subplots(
    2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [1.15, 1]}
)
fig.subplots_adjust(left=0.075, right=0.92, top=0.77, bottom=0.13, hspace=0.38)
validation = contact.twinx()
validation.spines["right"].set_visible(True)
validation.tick_params(length=0, pad=8)
fig.suptitle(
    "171M models: Auto Research improves the training recipe",
    x=0.075,
    ha="left",
    y=0.975,
    fontsize=18,
    fontweight="bold",
)
fig.text(
    0.075,
    0.928,
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
    bbox_to_anchor=(0.067, 0.892),
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
    bbox_to_anchor=(0.067, 0.851),
    ncol=2,
    fontsize=10,
)
training.set_title(
    "Training loss · raw 10-step logs + 1,000-step trailing mean",
    loc="left",
    fontsize=11,
    pad=12,
)
warmup = training.inset_axes([0.59, 0.52, 0.36, 0.4])
warmup.set_facecolor("#f8fafc")
warmup.set_title("First 1,000 steps · full loss range", fontsize=9, pad=6)
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
    0.075,
    0.06,
    "One seed per recipe; all ten evaluations are shown. P@L bands: 95% chain-bootstrap "
    "CI (20,775 chains). Validation: 4,096 sequences.",
    fontsize=9,
    color="#64748b",
)
fig.text(
    0.075,
    0.032,
    "Training traces: 10,001 logged rank-0 sequence-mean losses per recipe; warmup is "
    "shown in the inset. ESMC 171M is our AdamW reproduction.",
    fontsize=9,
    color="#64748b",
)
save(fig, "matched-100k-curves")
(OUT / "training-curve-summary.json").write_text(json.dumps(training_stats, indent=2) + "\n")
print(f"Wrote four figures and training summary to {OUT}")
