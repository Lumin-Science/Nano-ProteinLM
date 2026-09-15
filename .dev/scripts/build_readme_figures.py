"""Build static README figures with Matplotlib from the saved, verified measurements."""

import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from plot_autoresearch_history import plot_history

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".dev/reports/readme-overview-20260913"
OUT = ROOT / ".dev/reports/readme-figures-20260914"
BLUE, ORANGE = "tab:blue", "tab:orange"


def configure():
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "nanoprotein-readme-figures",
        }
    )


def style(ax):
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3, width=0.8)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / f"{stem}.png", dpi=200, bbox_inches="tight", metadata={"Software": "Matplotlib"}
    )
    svg_path = OUT / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def trailing_mean(values, window=100):
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def released_comparison(data):
    rows = sorted(data["models"], key=lambda row: row["p_at_l"], reverse=True)
    fig, ax = plt.subplots(figsize=(8.3, 3.5), layout="constrained")
    for index, row in enumerate(rows):
        ours = row["model"].startswith("Nano")
        value = row["p_at_l"] * 100
        interval = row["p_at_l_95_ci"]
        error = (
            None
            if interval is None
            else [[value - interval[0] * 100], [interval[1] * 100 - value]]
        )
        ax.barh(index, value, height=0.56, color=ORANGE if ours else BLUE)
        if error is not None:
            ax.errorbar(value, index, xerr=error, fmt="none", ecolor="black", capsize=3)
    ax.set_yticks(
        range(len(rows)),
        [
            "AutoResearch 171M" if row["model"].startswith("Nano") else row["model"]
            for row in rows
        ],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 65)
    ax.set_xlabel("Contact P@L (%)")
    ax.set_title("Protein-model comparison")
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    save(fig, "released-model-comparison")


def matched_curves():
    paired = ROOT / ".dev/reports/nibi-paired-unique-b2048-100k-20260909"
    curve = json.loads((paired / "learning-curve.json").read_text())
    assert curve["global_batch"] == 2048 and curve["target_steps"] == 100000
    fig, (training, contact) = plt.subplots(1, 2, figsize=(12.8, 4.3))
    fig.subplots_adjust(left=0.065, right=0.935, bottom=0.16, top=0.79, wspace=0.29)
    validation = contact.twinx()
    validation.spines["right"].set_visible(True)
    warmup = training.inset_axes([0.45, 0.57, 0.5, 0.37])
    warmup.set_title("Warmup", fontsize=8, pad=3)
    statistics = {}
    recipes = [
        ("baseline", "ESMC-like 171M (AdamW)", BLUE),
        ("setting3", "AutoResearch 171M", ORANGE),
    ]
    for key, label, color in recipes:
        rows = curve["runs"][key]
        assert [row["optimizer_step"] for row in rows] == list(range(10000, 100001, 10000))
        assert all(row["independently_audited"] for row in rows)
        steps = np.array([row["optimizer_step"] for row in rows]) / 1000
        pal = np.array([row["p_at_l"] for row in rows]) * 100
        ci = np.array([row["confidence_interval_95"] for row in rows]) * 100
        loss = np.array([row["validation_loss"] for row in rows])
        contact.fill_between(steps, ci[:, 0], ci[:, 1], color=color, alpha=0.18, linewidth=0)
        contact.plot(steps, pal, "o-", color=color, lw=1.8, markersize=4, label=label)
        validation.plot(steps, loss, "s--", color=color, lw=1.5, markersize=3.5)
        with gzip.open(paired / "full" / key / "metrics.jsonl.gz", "rt") as handle:
            logged = [
                record
                for line in handle
                if (record := json.loads(line)).get("event") == "train"
            ]
        expected_steps = [1, *range(10, 100001, 10)]
        assert [record["optimizer_step"] for record in logged] == expected_steps
        train_steps = np.array(expected_steps)
        raw = np.array([record["loss"] for record in logged])
        assert np.isfinite(raw).all()
        smoothed = trailing_mean(raw)
        visible = train_steps >= 1000
        training.plot(
            train_steps[visible] / 1000,
            raw[visible],
            color=color,
            alpha=0.16,
            lw=0.45,
            rasterized=True,
        )
        training.plot(train_steps[visible] / 1000, smoothed[visible], color=color, lw=1.7)
        early = train_steps <= 1000
        warmup.plot(train_steps[early], raw[early], color=color, lw=1)
        statistics[key] = {
            "records": len(logged),
            "first_step": int(train_steps[0]),
            "last_step": int(train_steps[-1]),
            "final_raw_sequence_mean_loss": float(raw[-1]),
            "final_100_record_mean": float(smoothed[-1]),
            "minimum_after_warmup": float(raw[visible].min()),
            "maximum_after_warmup": float(raw[visible].max()),
        }
    handles, labels = contact.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.99), ncol=2)
    for ax, title, ylabel in (
        (training, "MLM training", "Training loss"),
        (contact, "P@L prediction / validation loss", "P@L (%)"),
    ):
        ax.set_title(title)
        ax.set_xlabel("Training steps (×1,000)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 103)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        style(ax)
    contact.set_ylim(10, 40)
    validation.set_ylim(2.35, 2.61)
    validation.set_ylabel("Validation loss")
    validation.tick_params(direction="out", length=3, width=0.8)
    contact.legend(
        [
            Line2D([], [], color="0.35", marker="o", lw=1.8, markersize=4),
            Line2D([], [], color="0.35", marker="s", ls="--", lw=1.5, markersize=3.5),
        ],
        ["P@L (left axis)", "Validation loss (right axis)"],
        loc="lower center",
        fontsize=8.5,
    )
    training.set_ylim(2.28, 2.85)
    warmup.set_xlim(0, 1000)
    warmup.set_ylim(2.6, 4.5)
    warmup.set_xticks([0, 500, 1000])
    warmup.set_yticks([3, 4])
    warmup.set_xlabel("Steps", fontsize=7, labelpad=1)
    warmup.tick_params(labelsize=7, length=2, pad=2)
    save(fig, "matched-100k-curves")
    (OUT / "training-curve-summary.json").write_text(json.dumps(statistics, indent=2) + "\n")


def box(ax, x, y, text, *, color=BLUE, width=1.7, height=0.7):
    ax.add_patch(
        Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            facecolor="white",
            edgecolor=color,
            linewidth=1.1,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=10.5, linespacing=1.3)


def arrow(ax, start, end, *, color="0.3"):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=12, color=color, linewidth=1.0
        )
    )


def pipelines():
    centers = np.arange(6) * 2.05 + 1
    fig, ax = plt.subplots(figsize=(12.6, 2.35))
    ax.set_xlim(-0.05, 12.25)
    ax.set_ylim(-0.55, 1.55)
    ax.axis("off")
    labels = [
        "UniRef90\nMGnify\nOMG/IMG",
        "Quality filter\n& deduplicate",
        "Cluster at\n70% identity",
        "Evaluation\ndecontamination",
        "Split & verify",
        "666.0M\ntraining proteins",
    ]
    for index, (x, label) in enumerate(zip(centers, labels, strict=True)):
        box(ax, x, 0, label, color=ORANGE if index == 5 else BLUE)
        if index:
            arrow(ax, (centers[index - 1] + 0.85, 0), (x - 0.85, 0))
    box(ax, centers[3], 1.1, "Protected\nevaluation sets")
    arrow(ax, (centers[3], 0.75), (centers[3], 0.35))
    save(fig, "data-preparation")

    fig, ax = plt.subplots(figsize=(12.6, 2.25))
    ax.set_xlim(-0.05, 12.25)
    ax.set_ylim(-1.25, 0.65)
    ax.axis("off")
    labels = [
        "Evaluate\nbaseline",
        "Propose\none change",
        "Train & evaluate\ntwo seeds",
        "Keep / discard",
        "Record\nresult",
        "Scale-up\ntest",
    ]
    for index, (x, label) in enumerate(zip(centers, labels, strict=True)):
        box(ax, x, 0, label, color=ORANGE if index == 5 else BLUE)
        if index:
            arrow(ax, (centers[index - 1] + 0.85, 0), (x - 0.85, 0))
    ax.plot([centers[4], centers[4], centers[1]], [-0.35, -0.85, -0.85], color="0.3", lw=1)
    arrow(ax, (centers[1], -0.85), (centers[1], -0.35))
    ax.text(
        (centers[1] + centers[4]) / 2, -1.0, "Next trial", ha="center", va="top", fontsize=10
    )
    ax.text((centers[4] + centers[5]) / 2, 0.47, "After search", ha="center", fontsize=9)
    save(fig, "autoresearch-loop")


def main():
    data = json.loads((SOURCE / "comparison-data.json").read_text())
    for relative, digest in data["sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    configure()
    matched_curves()
    released_comparison(data)
    pipelines()
    plot_history(ROOT / ".dev/reports/program2/runs-through-r38.tsv", OUT / "validation-loss")
    print(f"Wrote five PNG/SVG figure pairs to {OUT}")


if __name__ == "__main__":
    main()
