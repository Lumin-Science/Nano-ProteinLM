"""Compare recorded training dynamics and held-out performance around the resume."""

import ast
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
REPORT = OUT.parent
REPO = OUT.parents[3]
WORKSPACE = Path("/Users/jojo/workspace/AutoResearch_ESMC/nano-protein-embedding")
OLD = REPO / ".dev/reports/nibi-paired-unique-b2048-100k-20260909"
RAW_OLD = WORKSPACE / ".exps/nibi-paired-unique-b2048-100k-20260909/full/setting3/metrics.jsonl"
RAW_NEW = REPORT / "full/metrics-through-123640.jsonl.gz"
before = [
    r
    for line in RAW_OLD.read_text().splitlines()
    if (r := json.loads(line)).get("event") == "train"
]
after = [
    r
    for line in gzip.decompress(RAW_NEW.read_bytes()).decode().splitlines()
    if (r := json.loads(line)).get("event") == "train"
]
rows = before + after
keys = ("loss", "objective_loss", "gradient_norm", "step_model_tokens", "learning_rate")
windows = {}
for lo, hi in (
    (95000, 100000),
    (99000, 100000),
    (99900, 100000),
    (100000, 100100),
    (100000, 101000),
    (100000, 105000),
    (105000, 110000),
    (110000, 120000),
):
    selected = [r for r in rows if lo < r["optimizer_step"] <= hi]
    windows[f"{lo}-{hi}"] = {
        "logged_updates": len(selected),
        **{
            key: {
                "mean": float(np.mean([r[key] for r in selected])),
                "std": float(np.std([r[key] for r in selected])),
                "max": max(r[key] for r in selected),
            }
            for key in keys
        },
    }
evals = []
for step in (80000, 90000, 100000, 110000, 120000):
    p = (
        (OLD / "full/setting3" if step <= 100000 else REPORT / "full")
        / "evaluations"
        / f"step-{step:06d}"
    )
    a = json.loads((p / "LOCAL_AUDIT.json").read_text())
    probe = json.loads((p / "eval-p-at-l/CONTACT_PROBE.json").read_text())
    evals.append(
        {
            "step": step,
            "validation_loss": a["validation"]["sequence_mean_nll"],
            "p_at_l": a["p_at_l"],
            "ci": a["uncertainty"]["confidence_interval_95"],
            "probe_C": probe["selected_C"],
        }
    )
changes = [
    {
        "interval": f"{a['step']}-{b['step']}",
        "validation_loss_reduction": a["validation_loss"] - b["validation_loss"],
        "p_at_l_gain_pp": 100 * (b["p_at_l"] - a["p_at_l"]),
    }
    for a, b in zip(evals, evals[1:], strict=False)
]
commits = [
    "bc54124abe193623abf64b44e65b881cded65f48",
    "bd0b754e817363bf5f5f9d48b2c482818a79559f",
]


def source(commit, name):
    return subprocess.check_output(
        ["git", "show", f"{commit}:src/nanoprotein/{name}.py"], cwd=REPO
    )


unchanged = {
    name: source(commits[0], name) == source(commits[1], name)
    for name in ("model", "tokenizer", "batch_balance", "flash_attention", "schedule")
}
trees = [ast.parse(source(c, "train")) for c in commits]
for name in (
    "training_losses",
    "build_optimizer",
    "muon_adamw_parameter_groups",
    "_OptimizerBundle",
):
    nodes = [
        next(
            n
            for n in t.body
            if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name == name
        )
        for t in trees
    ]
    unchanged[name] = ast.dump(nodes[0]) == ast.dump(nodes[1])
assert all(unchanged.values())
result = {
    "scope": "observational diagnosis, no production changes",
    "logged_every_steps": 10,
    "unobserved_first_resume_steps": list(range(100001, 100010)),
    "source_implementation_unchanged": unchanged,
    "windows": windows,
    "evaluations": evals,
    "evaluation_changes": changes,
    "old_metrics_sha256": hashlib.sha256(RAW_OLD.read_bytes()).hexdigest(),
    "new_metrics_gzip_sha256": hashlib.sha256(RAW_NEW.read_bytes()).hexdigest(),
    "rank0_logged_sequences_per_update_before": 512,
    "rank0_logged_sequences_per_update_after": 256,
    "global_sequences_per_update": 2048,
    "global_microbatch_sequences_before_and_after": 512,
    "gradient_accumulation_before_and_after": 4,
}
(OUT / "ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
with gzip.open(OUT / "boundary-metrics-95k-105k.jsonl.gz", "wb") as handle:
    handle.write(
        "".join(
            json.dumps(r) + "\n" for r in rows if 95000 < r["optimizer_step"] <= 105000
        ).encode()
    )
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.18,
        "figure.facecolor": "white",
    }
)
fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.1), layout="constrained")
colors = ["#3459a0", "#008775"]
for ax, key, title in [
    (axes[0, 0], "objective_loss", "A  Global training objective stays continuous"),
    (axes[0, 1], "loss", "B  Rank-0 loss becomes noisier"),
]:
    for group, color, label in zip(
        (before, after), colors, ("Original, 4 GPUs", "Resumed, 8 GPUs"), strict=True
    ):
        selected = [r for r in group if 95000 < r["optimizer_step"] <= 105000]
        x = np.array([r["optimizer_step"] for r in selected]) / 1000
        y = np.array([r[key] for r in selected])
        ax.scatter(x, y, s=5, color=color, alpha=0.19, rasterized=True)
        bx = []
        by = []
        for lo in np.arange(95, 105, 0.5):
            sel = (x > lo) & (x <= lo + 0.5)
            if sel.any():
                bx.append(lo + 0.25)
                by.append(y[sel].mean())
        ax.plot(bx, by, color=color, lw=2, label=label)
    ax.set(
        title=title, xlabel="Total optimizer steps (thousands)", ylabel="Loss", xlim=(95, 105)
    )
    ax.axvline(100, color="#aa4455", ls="--", lw=1)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
axes[0, 0].text(
    0.02,
    0.98,
    "Mean over 1k updates: 2.31644 → 2.31690",
    transform=axes[0, 0].transAxes,
    va="top",
    fontsize=10,
)
std0 = windows["95000-100000"]["loss"]["std"]
std1 = windows["100000-105000"]["loss"]["std"]
axes[0, 1].text(
    0.02,
    0.98,
    f"5k-update SD: {std0:.4f} → {std1:.4f}\nOnly rank 0 is logged: 512 → 256 sequences/update",
    transform=axes[0, 1].transAxes,
    va="top",
    fontsize=9,
)
for ax, key, title, label in [
    (
        axes[1, 0],
        "validation_loss",
        "C  Held-out validation keeps improving",
        "Validation loss",
    ),
    (axes[1, 1], "p_at_l", "D  P@L gains vary across 10k intervals", "P@L (%)"),
]:
    x = np.array([v["step"] for v in evals]) / 1000
    y = np.array([v[key] for v in evals])
    factor = 100 if key == "p_at_l" else 1
    ax.plot(x[:3], y[:3] * factor, "o-", color=colors[0], lw=2)
    ax.plot(x[2:], y[2:] * factor, "o-", color=colors[1], lw=2)
    if key == "p_at_l":
        low = y - np.array([v["ci"][0] for v in evals])
        high = np.array([v["ci"][1] for v in evals]) - y
        ax.errorbar(
            x,
            y * 100,
            yerr=np.array([low, high]) * 100,
            fmt="none",
            ecolor="#555555",
            capsize=4,
        )
    ax.axvline(100, color="#aa4455", ls="--", lw=1)
    ax.set(
        title=title,
        xlabel="Total optimizer steps (thousands)",
        ylabel=label,
        xticks=[80, 90, 100, 110, 120],
        xlim=(78, 125),
    )
    ax.annotate(
        f"{y[-1] * factor:.3f}",
        (x[-1], y[-1] * factor),
        xytext=(8, 4),
        textcoords="offset points",
        color=colors[1],
        fontweight="bold",
    )
fig.suptitle("Setting 3: examining the 100k resume", fontsize=17, fontweight="bold")
fig.supxlabel(
    (
        "Top panels: every 10th update; lines average 500 updates, separately on each side. "
        "Dashed line: resume.\nP@L error bars are 95% chain-bootstrap intervals; "
        "all evaluation protocols are unchanged."
    ),
    fontsize=9,
)
fig.savefig(OUT / "resume-dynamics.png", dpi=180)
fig.savefig(OUT / "resume-dynamics.pdf")
print(
    json.dumps(
        {"windows": windows, "evaluation_changes": changes, "unchanged": unchanged}, indent=2
    )
)
