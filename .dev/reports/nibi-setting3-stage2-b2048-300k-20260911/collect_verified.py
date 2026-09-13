"""Curate an audited periodic Stage 2 endpoint; finalize 700k separately."""

import argparse
import gzip
import hashlib
import json
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPORT = Path(__file__).resolve().parent
REPO = REPORT.parents[2]
RAW = REPO / ".exps" / REPORT.name
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("step", type=int)
parser.add_argument("--status", type=Path, required=True)
parser.add_argument(
    "--expected-program-sha256",
    default="fbfb6078304127b3d5dd13cd1d6021b6ed28eca2b6032bab5d25551b8b941c0b",
    help="Observed hash of the user's unrelated program file; collection must preserve it.",
)
args = parser.parse_args()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curve_row(audit, stage):
    return dict(
        global_step=audit["optimizer_steps"],
        stage2_updates=audit["optimizer_steps"] - 400000,
        stage=stage,
        validation_loss=audit["validation"]["sequence_mean_nll"],
        p_at_l=audit["p_at_l"],
        p_at_l_95_ci=audit["uncertainty"]["confidence_interval_95"],
        checkpoint_sha256=audit["checkpoint_sha256"],
    )


source = RAW / "full/evaluations" / f"step-{args.step}"
audit = read(source / "LOCAL_AUDIT.json")
status = read(args.status)
current = status["full"]["latest_train"]
assert 410000 <= args.step < 700000 and args.step % 10000 == 0
assert audit["status"] == "passed" and audit["optimizer_steps"] == args.step
assert args.step in status["full"]["evaluated_steps"]
assert current["optimizer_step"] >= args.step and not status["full"]["failure"]
assert sha(REPO / "autoresearch/program.md") == args.expected_program_sha256
destination = REPORT / "full/evaluations" / source.name
for path in source.rglob("*"):
    relative = path.relative_to(source)
    if (
        path.is_file()
        and "components" not in relative.parts
        and path.suffix in (".json", ".gz")
    ):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

audits = [
    read(p) for p in sorted((REPORT / "full/evaluations").glob("step-*/LOCAL_AUDIT.json"))
]
assert [a["optimizer_steps"] for a in audits] == list(range(410000, args.step + 1, 10000))
parent_path = (
    REPO / ".dev/reports/nibi-setting3-b2048-400k-20260910/full/evaluations/step-400000"
)
parent = read(parent_path / "RESULT_VERIFIED.json")
rows = [curve_row(parent, "stage1_parent")] + [curve_row(a, "stage2") for a in audits]
write(
    REPORT / "learning-curve.json",
    dict(
        note=(
            "400k is the Stage 1 parent reference; later rows are verified production "
            "Stage 2 endpoints. Qualification trials are excluded."
        ),
        rows=rows,
        comparisons_to_400k={
            str(a["optimizer_steps"]): a["comparison_to_400k"] for a in audits
        },
    ),
)
latest = audit["latest_verified_training_step"]
metrics = RAW / "full/metrics.jsonl"
metric_rows = [json.loads(line) for line in metrics.read_text().splitlines()]
metric_rows = [r for r in metric_rows if r.get("event") == "train"]
assert metric_rows[-1]["optimizer_step"] == latest
snapshot = REPORT / "full" / f"metrics-through-{latest}.jsonl.gz"
snapshot.write_bytes(gzip.compress(metrics.read_bytes(), mtime=0))
write(
    REPORT / "full" / f"METRICS_SNAPSHOT-{latest}.json",
    dict(
        snapshot_file=snapshot.name,
        last_logged_step=latest,
        sha256=sha(snapshot),
        description=(
            f"Metric snapshot used in the one-time independent {args.step // 1000}k "
            "endpoint audit."
        ),
    ),
)

now = datetime.fromisoformat(status["checked_utc"])
tz = ZoneInfo("America/Toronto")
previous = audits[-2] if len(audits) > 1 else None
reference_step = previous["optimizer_steps"] if previous else 400000
reference_seconds = previous["training_seconds_at_checkpoint"] if previous else 0
rate = (current["training_seconds"] - reference_seconds) / (
    current["optimizer_step"] - reference_step
)
remaining = 700000 - current["optimizer_step"]
remaining_evaluations = 30 - len(audits)
evaluation_seconds = audit["evaluation_run"]["wall_seconds"]
duration = remaining * rate + remaining_evaluations * evaluation_seconds
finish = now + timedelta(seconds=duration)
slow_finish = finish + timedelta(seconds=0.1 * remaining * rate)
allocation_end = datetime(2026, 9, 14, 7, 24, 3, tzinfo=tz)
mg = current["source_exposure_global"]["mgnify"]
unused = mg["available_records"] - mg["unique_records_seen"]
expected = remaining * 2048 * 0.06
assert mg["repeated_draws"] == 0
assessment = dict(
    checked_utc=status["checked_utc"],
    global_step=current["optimizer_step"],
    stage2_updates=current["optimizer_step"] - 400000,
    stage2_progress_percent=(current["optimizer_step"] - 400000) / 3000,
    recent_training_seconds_per_update=rate,
    completed_evaluations=status["full"]["evaluated_steps"],
    remaining_evaluations=remaining_evaluations,
    evaluation_wall_seconds_estimate=evaluation_seconds,
    finish_toronto=finish.astimezone(tz).isoformat(),
    finish_10pct_slower_toronto=slow_finish.astimezone(tz).isoformat(),
    allocation_hours_remaining=(allocation_end - now).total_seconds() / 3600,
    finish_headroom_hours=(allocation_end - finish).total_seconds() / 3600,
    mgnify_unused=unused,
    mgnify_expected_remaining_draws=expected,
    mgnify_headroom_fraction=unused / expected - 1 if expected else None,
    latest_verified_training_step=latest,
    training_continued_after_evaluation=audit["training_continued_after_evaluation"],
    evaluation_audit_passed=True,
)
stamp = now.strftime("%Y%m%dT%H%MZ")
write(
    REPORT / "monitoring" / f"{stamp}.json",
    dict(
        assessment=assessment,
        allocation=status["allocation"],
        latest_train=current,
        completed_evaluation=audit["evaluation_run"],
        production_failure=status["full"]["failure"],
    ),
)

lines = ["## Production evaluation results", ""]
lines.append(
    f"**{len(audits)} of 30 production Stage 2 evaluations are verified**, through global "
    f"**{args.step // 1000}k**. Each "
    f"[independent audit](full/evaluations/step-{args.step}/LOCAL_AUDIT.json) checks "
    "checkpoint bindings, all 16 contact shards, all 20,775 historical chain identities, "
    "the unchanged MLM/probe protocols, and an independently recomputed 5,000-replicate "
    "bootstrap. The [machine-readable curve](learning-curve.json) excludes qualification "
    "trials."
)
lines += [
    "",
    "| Checkpoint | Stage 2 updates | Validation loss ↓ | P@L ↑ | 95% chain-bootstrap CI |",
    "|---|---:|---:|---:|---:|",
]
for row in rows:
    label = "Stage 1 parent" if row["global_step"] == 400000 else "Stage 2"
    low, high = row["p_at_l_95_ci"]
    lines.append(
        f"| {label}, {row['global_step'] // 1000}k | {row['stage2_updates']:,} | "
        f"{row['validation_loss']:.6f} | {row['p_at_l'] * 100:.3f}% | "
        f"{low * 100:.3f}–{high * 100:.3f}% |"
    )
comparison = audit["comparison_to_400k"]
low, high = comparison["paired_chain_bootstrap_95_ci_percentage_points"]
lines += [
    "",
    "Relative to 400k, validation loss changed by "
    f"**{comparison['validation_loss_difference']:+.6f}**, and P@L changed by "
    f"**{comparison['p_at_l_difference_percentage_points']:+.3f} percentage points**. "
    "The paired chain-bootstrap 95% interval for this P@L change is "
    f"**{low:+.3f} to {high:+.3f} points**, using the same chains and 5,000 replicates. "
    "These intervals quantify uncertainty across evaluation chains, not variation "
    "across training seeds.",
]
if previous:
    change = (audit["p_at_l"] - previous["p_at_l"]) * 100
    loss_change = (
        audit["validation"]["sequence_mean_nll"] - previous["validation"]["sequence_mean_nll"]
    )
    lines += [
        "",
        f"From {previous['optimizer_steps'] // 1000}k to {args.step // 1000}k, "
        f"validation loss changed by **{loss_change:+.6f}** and P@L by "
        f"**{change:+.3f} points**.",
    ]
lines += [
    "",
    f"The latest independently checked training metrics reached global **{latest:,}**. "
    "MGnify had no repeats, and the fixed Stage 2 decay schedule remained correct. "
    f"The latest evaluation took **{evaluation_seconds:.1f} seconds**.",
    "",
    "## Latest monitoring check",
    "",
]
local_time = now.astimezone(tz).strftime("%B %d, %H:%M")
headroom = (
    f"{assessment['mgnify_headroom_fraction'] * 100:.2f}%" if expected else "no remaining draws"
)
lines += [
    f"At **{local_time} Toronto**, the [monitoring receipt](monitoring/{stamp}.json) "
    f"recorded global **{current['optimizer_step']:,}**, or "
    f"**{assessment['stage2_updates']:,} / 300,000 Stage 2 updates "
    f"({assessment['stage2_progress_percent']:.2f}%)**, with no production failure. "
    f"Throughput was **{rate:.3f} seconds per update** since {reference_step // 1000}k. "
    f"The allocation had **{assessment['allocation_hours_remaining']:.2f} hours remaining**; "
    f"estimated completion was **{finish.astimezone(tz).strftime('%B %d at %H:%M')} Toronto**, "
    f"or **{slow_finish.astimezone(tz).strftime('%B %d at %H:%M')}** with a 10% training "
    f"slowdown. MGnify had **{unused:,} unused records** versus "
    f"**{expected:,.0f} expected remaining draws**, with headroom of **{headroom}**.",
    "",
    "",
]
readme = REPORT / "README.md"
text = readme.read_text()
start = text.index("## Production evaluation results\n")
end = text.index("## Earlier monitoring check\n", start)
readme.write_text(text[:start] + "\n".join(lines) + text[end:])
main = REPO / "README.md"
low, high = audit["uncertainty"]["confidence_interval_95"]
replacement = (
    f"The latest **{args.step // 1000}k** production evaluation reached validation loss "
    f"**{audit['validation']['sequence_mean_nll']:.6f}** and P@L "
    f"**{audit['p_at_l'] * 100:.3f}%** (95% CI **{low * 100:.3f}–{high * 100:.3f}%**), "
    f"a **{comparison['p_at_l_difference_percentage_points']:.3f}-point** P@L gain over "
    "its 400k parent."
)
text, count = re.subn(
    r"The latest \*\*\d+k\*\* production evaluation reached[^\n]+?over its 400k parent\.",
    replacement,
    main.read_text(),
)
assert count == 1
main.write_text(text)
for target in re.findall(r"\]\(([^)]+)\)", readme.read_text()):
    if not target.startswith(("http", "#")):
        assert (REPORT / target).exists(), target
files = [p for p in sorted(REPORT.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
(REPORT / "SHA256SUMS").write_text(
    "".join(f"{sha(p)}  {p.relative_to(REPORT)}\n" for p in files)
)
assert sha(REPO / "autoresearch/program.md") == args.expected_program_sha256
print(json.dumps(dict(assessment=assessment, comparison_to_400k=comparison), indent=2))
