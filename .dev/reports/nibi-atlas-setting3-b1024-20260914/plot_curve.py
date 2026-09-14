"""Plot verified Atlas evaluations and a moving mean of the saved training log."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metrics", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
paths = sorted((root / "evaluations").glob("step-*/RESULT_VERIFIED.json"))
paths.append(root / "final/RESULT_VERIFIED.json")
results = [json.loads(path.read_text()) for path in paths]
results.sort(key=lambda row: row["optimizer_steps"])
records = []
for result in results:
    lo, hi = result["uncertainty"]["confidence_interval_95"]
    records.append(
        dict(
            optimizer_step=result["optimizer_steps"],
            validation_nll=result["validation"]["sequence_mean_nll"],
            p_at_l=result["p_at_l"],
            p_at_l_ci_low=lo,
            p_at_l_ci_high=hi,
            model_tokens=result["model_tokens"],
            sequences_seen=result["sequences_seen"],
            checkpoint_sha256=result["checkpoint_sha256"],
        )
    )
with (root / "PERFORMANCE_CURVE.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

metrics = [json.loads(line) for line in args.metrics.read_text().splitlines()]
metrics = [
    row for row in metrics if row.get("event") == "train" and row["optimizer_step"] % 10 == 0
]
steps = np.array([row["optimizer_step"] for row in metrics])
losses = np.array([row["loss"] for row in metrics])
assert np.isfinite(losses).all() and np.all(np.diff(steps) == 10)
window = 100  # 100 logged measurements, one every 10 optimizer updates.
smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
smooth_steps = steps[window - 1 :]
with (root / "TRAINING_CURVE.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["optimizer_step", "training_sequence_nll_1k_step_moving_mean"])
    writer.writerows(zip(smooth_steps[::10], smooth[::10], strict=True))

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#334155",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "svg.fonttype": "none",
    }
)
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.19, top=0.77, wspace=0.27)
fig.suptitle("Setting 3 (171M) on ESM Atlas", fontsize=18, weight="bold", y=0.96)
fig.text(
    0.5,
    0.875,
    "Fresh training · 8 H100s · Batch 1,024 · Unscreened Atlas subset",
    ha="center",
    color="#64748B",
)
x = np.array([row["optimizer_step"] for row in records]) / 1000
p = np.array([row["p_at_l"] for row in records]) * 100
lo = np.array([row["p_at_l_ci_low"] for row in records]) * 100
hi = np.array([row["p_at_l_ci_high"] for row in records]) * 100
v = np.array([row["validation_nll"] for row in records])
left.fill_between(x, lo, hi, color="#16866B", alpha=0.18)
left.plot(x, p, "o-", color="#16866B", linewidth=2.2, markersize=4)
left.set(title="Contact precision ↑", ylabel="P@L (%)")
left.annotate(
    f"{p[-1]:.2f}%",
    (x[-1], p[-1]),
    xytext=(-5, 10),
    textcoords="offset points",
    ha="right",
    color="#126B55",
    weight="bold",
)
right.plot(
    smooth_steps / 1000,
    smooth,
    color="#DB8B2B",
    linewidth=1.2,
    label="Atlas train, 1k-step mean",
)
right.plot(x, v, "o-", color="#4164B4", linewidth=2.2, markersize=4, label="Fixed validation")
right.set(title="Masked language modeling loss ↓", ylabel="Sequence-mean NLL")
right.legend(frameon=False, fontsize=9, loc="upper right")
for ax in (left, right):
    ax.set_xlim(0, 92)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Optimizer steps (thousands)")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    ax.set_axisbelow(True)
fig.text(
    0.07,
    0.045,
    f"Stopped at {records[-1]['optimizer_step']:,} updates by allocation deadline. "
    "P@L band: 95% chain-bootstrap CI, 20,775 chains.",
    fontsize=9,
    color="#64748B",
)
fig.savefig(root / "performance_curve.png", dpi=200, facecolor="white")
fig.savefig(root / "performance_curve.svg", facecolor="white")
plt.close(fig)
print(f"Wrote {len(records)} evaluation points and {len(smooth[::10])} training curve samples.")
