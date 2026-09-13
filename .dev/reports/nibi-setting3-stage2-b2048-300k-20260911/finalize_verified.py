"""Publish the verified 700k endpoint after durable full-state restoration passes."""

import argparse
import gzip
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPORT = Path(__file__).resolve().parent
REPO = REPORT.parents[2]
RAW = REPO / ".exps" / REPORT.name
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--status", type=Path, required=True)
parser.add_argument("--expected-program-sha256", required=True)
args = parser.parse_args()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ai(block):
    return '<div class="ai">\n\n' + block + "\n\n</div>\n\n"


assert sha(REPO / "autoresearch/program.md") == args.expected_program_sha256
status = read(args.status)
full = status["full"]
complete = read(RAW / "full/TRAINING_COMPLETE.json")
storage = read(RAW / "final-storage-check.json")
preservation = read(RAW / "checkpoint-700000.json")
restoration = read(RAW / "checkpoint-700000-restore-verified.json")
source = RAW / "full/evaluations/step-700000"
audit = read(source / "LOCAL_AUDIT.json")
checkpoint = audit["checkpoint_sha256"]
assert status["workflow"]["phase"] == "complete"
assert status["workflow"]["status"] == "passed" and not full["failure"]
assert full["evaluated_steps"] == list(range(410000, 700001, 10000))
assert full["latest_train"]["optimizer_step"] == 700000
assert complete == full["complete"] and complete["stop_reason"] == "max_steps"
assert complete["optimizer_steps"] == complete["target_optimizer_steps"] == 700000
assert complete["final_checkpoint"]["sha256"] == checkpoint
assert audit["status"] == preservation["status"] == restoration["status"] == "passed"
assert preservation["checkpoint_sha256"] == restoration["checkpoint_sha256"] == checkpoint
assert restoration["optimizer_steps"] == 700000 and restoration["full_training_complete"]
assert restoration["model_and_optimizer_roundtrip_exact"]
assert (
    restoration["finite_model_and_optimizer"] and restoration["global_sampler_state_verified"]
)
assert restoration["source_exposure_global"] == audit["source_exposure_at_checkpoint"]
assert restoration["source_exposure_global"] == full["latest_train"]["source_exposure_global"]
assert restoration["data_manifest_sha256"] == audit["data_manifest_sha256"]
assert storage["source_commit"] == audit["source_commit"] and not storage["source_dirty"]
assert "12162637.44|" not in storage["slurm_steps"]
assert "12162637|RUNNING|" in status["allocation"]
assert storage["durable_files"]["checkpoint-700000.pt"]["size_bytes"] == preservation["bytes"]

destination = REPORT / "full/evaluations/step-700000"
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
for name in (
    "TRAINING_COMPLETE.json",
    "TRAINING_VERIFIED.json",
    "COMPLETE.txt",
    "launch-started-utc.txt",
):
    shutil.copy2(RAW / "full" / name, REPORT / "full" / name)
for name in ("checkpoint-700000.json", "checkpoint-700000-restore-verified.json"):
    shutil.copy2(RAW / name, REPORT / "milestones" / name)
shutil.copy2(RAW / "final-storage-check.json", REPORT / "monitoring/final-storage-check.json")
audits = [
    read(p) for p in sorted((REPORT / "full/evaluations").glob("step-*/LOCAL_AUDIT.json"))
]
assert [a["optimizer_steps"] for a in audits] == list(range(410000, 700001, 10000))
assert all(a["status"] == "passed" for a in audits)
curve = read(REPORT / "learning-curve.json")
curve["rows"] = [row for row in curve["rows"] if row["global_step"] < 700000]
assert [row["global_step"] for row in curve["rows"]] == list(range(400000, 700000, 10000))
curve["rows"].append(
    dict(
        global_step=700000,
        stage2_updates=300000,
        stage="stage2",
        validation_loss=audit["validation"]["sequence_mean_nll"],
        p_at_l=audit["p_at_l"],
        p_at_l_95_ci=audit["uncertainty"]["confidence_interval_95"],
        checkpoint_sha256=checkpoint,
    )
)
curve["comparisons_to_400k"]["700000"] = audit["comparison_to_400k"]
curve["status"] = "complete"
write(REPORT / "learning-curve.json", curve)
assert audit["p_at_l"] == max(row["p_at_l"] for row in curve["rows"])
assert audit["validation"]["sequence_mean_nll"] == min(
    row["validation_loss"] for row in curve["rows"]
)
snapshot = REPORT / "full/metrics-through-700000.jsonl.gz"
snapshot.write_bytes(gzip.compress((RAW / "full/metrics.jsonl").read_bytes(), mtime=0))
write(
    REPORT / "full/METRICS_SNAPSHOT-700000.json",
    dict(
        snapshot_file=snapshot.name,
        last_logged_step=700000,
        sha256=sha(snapshot),
        description="Final metric snapshot used in the independent 700k endpoint audit.",
    ),
)
tz = ZoneInfo("America/Toronto")
started = datetime.fromisoformat((RAW / "full/launch-started-utc.txt").read_text().strip())
finished = datetime.fromisoformat((RAW / "full/COMPLETE.txt").read_text().strip())
workflow_finished = datetime.fromisoformat(status["workflow"]["utc"])
checked = datetime.fromisoformat(status["checked_utc"])
allocation_end = datetime(2026, 9, 14, 7, 24, 3, tzinfo=tz)
mg = restoration["source_exposure_global"]["mgnify"]
assert mg["repeated_draws"] == 0
comparison = audit["comparison_to_400k"]
final = dict(
    status="passed",
    checked_utc=status["checked_utc"],
    global_step=700000,
    stage2_updates=300000,
    completed_evaluation_count=30,
    launch_started_utc=started.isoformat(),
    production_completed_utc=finished.isoformat(),
    workflow_completed_utc=workflow_finished.isoformat(),
    production_wall_seconds=(finished - started).total_seconds(),
    workflow_wall_seconds=(workflow_finished - started).total_seconds(),
    training_loop_seconds=complete["training_seconds"],
    periodic_evaluation_seconds=complete["periodic_evaluation_seconds"],
    validation_loss=audit["validation"]["sequence_mean_nll"],
    p_at_l=audit["p_at_l"],
    p_at_l_95_ci=audit["uncertainty"]["confidence_interval_95"],
    comparison_to_400k=comparison,
    final_checkpoint=preservation,
    exact_model_optimizer_restoration_passed=True,
    source_commit=audit["source_commit"],
    data_manifest_sha256=audit["data_manifest_sha256"],
    source_exposure_global=restoration["source_exposure_global"],
    mgnify_unused_records=mg["available_records"] - mg["unique_records_seen"],
    allocation_retained=True,
    allocation=status["allocation"],
    allocation_hours_remaining_at_check=(allocation_end - checked).total_seconds() / 3600,
    user_program_sha256_preserved=args.expected_program_sha256,
)
write(REPORT / "FINAL_VERIFIED.json", final)

low, high = final["p_at_l_95_ci"]
blocks = [
    ai(
        "Production completed **300,000 Stage 2 updates, global 400k → 700k**, on eight Nibi "
        "H100 GPUs. The [final verification](FINAL_VERIFIED.json) passed: all **30 production "
        "evaluations** are verified and the durable 700k checkpoint passed exact model and "
        "optimizer restoration. The production launcher finished September 13, 2026 at "
        f"**{finished.astimezone(tz):%H:%M:%S} Toronto**; the complete workflow finished at "
        f"**{workflow_finished.astimezone(tz):%H:%M:%S}**. Production took **48h 00m 39s** "
        "including evaluations and checkpoint handling, with **46h 13m 23s** recorded in "
        "the training loop. The user-owned allocation remains retained."
    ),
    ai("## Production evaluation results"),
    ai(
        f"The final **700k** checkpoint has validation loss **{final['validation_loss']:.6f}** "
        f"and P@L **{final['p_at_l'] * 100:.3f}%** "
        f"(95% CI **{low * 100:.3f}–{high * 100:.3f}%**), "
        "the best measured values on both metrics in this Stage 2 run. Each "
        "[independent endpoint audit](full/evaluations/step-700000/LOCAL_AUDIT.json) checks "
        "checkpoint bindings, all 16 contact shards, all 20,775 historical chain identities, "
        "the unchanged MLM/probe protocols and an independently recomputed 5,000-replicate "
        "bootstrap. The [complete machine-readable curve](learning-curve.json) "
        "excludes qualification trials."
    ),
]
table = [
    "| Checkpoint | Stage 2 updates | Validation loss ↓ | P@L ↑ | 95% chain-bootstrap CI |",
    "|---|---:|---:|---:|---:|",
]
for row in curve["rows"]:
    label = "Stage 1 parent" if row["global_step"] == 400000 else "Stage 2"
    lo, hi = row["p_at_l_95_ci"]
    table.append(
        f"| {label}, {row['global_step'] // 1000}k | {row['stage2_updates']:,} | "
        f"{row['validation_loss']:.6f} | {row['p_at_l'] * 100:.3f}% | "
        f"{lo * 100:.3f}–{hi * 100:.3f}% |"
    )
blocks.append(ai("\n".join(table)))
lo, hi = comparison["paired_chain_bootstrap_95_ci_percentage_points"]
blocks.append(
    ai(
        "Relative to 400k, validation loss changed by "
        f"**{comparison['validation_loss_difference']:+.6f}**, and P@L improved by "
        f"**{comparison['p_at_l_difference_percentage_points']:.3f} percentage points**. "
        "The paired chain-bootstrap 95% interval for this gain is "
        f"**{lo:.3f}–{hi:.3f} points**. These intervals quantify uncertainty across "
        "evaluation chains, not variation across training seeds."
    )
)
blocks += [
    ai("## Final training and storage checks"),
    ai(
        "Training stopped at the requested **700,000** global updates with no production "
        "failure. The source checkout remained clean at the frozen commit, and the final "
        "metrics followed the original Stage 2 decay clock. The completed run retains "
        "**zero MGnify repeats** and "
        f"**{final['mgnify_unused_records']:,} unused MGnify records**. "
        "UniRef90 and OMG continued their permitted complete global passes."
    ),
    ai(
        "The [durable final checkpoint](milestones/checkpoint-700000.json) and "
        "[independent restoration audit](milestones/checkpoint-700000-restore-verified.json) "
        "bind the evaluated 700k weights to the saved full model, optimizer and global "
        f"sampler state. Checkpoint SHA-256: `{checkpoint}`. The "
        "[host receipt](monitoring/final-storage-check.json) confirms production step "
        "12162637.44 ended while allocation 12162637 remains running on g27, with "
        f"**{final['allocation_hours_remaining_at_check']:.2f} hours remaining** at the "
        f"**{checked.astimezone(tz):%B %d, %H:%M} Toronto** check."
    ),
]
readme = REPORT / "README.md"
text = readme.read_text()
title = text.split("\n", 1)[0]
earlier = text[text.index("## Earlier monitoring check\n") :]
text = title + "\n\n" + "".join(blocks) + earlier
storage_paragraph = next(
    p for p in text.split("\n\n") if p.startswith("The durable destination is ")
)
replacement = (
    "The durable destination is `/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/"
    "nibi-setting3-stage2-b2048-300k-20260911`. It contains the recipe, source bundle, "
    "transition and qualification records, full model/optimizer checkpoints at 500k, "
    "600k and 700k, and the final report under `run-record/`. The final checkpoint is "
    "`checkpoint-700000.pt`; scratch `full/checkpoint-final.pt` contains the same evaluated "
    "state. Later four-GPU continuation uses microbatch 128 per GPU with accumulation 4 "
    "to retain batch 2,048; memory and throughput on that layout still require qualification."
)
text = text.replace(storage_paragraph + "\n\n", ai(replacement))
monitor_paragraph = next(
    p for p in text.split("\n\n") if p.startswith("The existing two-hour thread monitor")
)
replacement = (
    "The requested 700k training, evaluation and checkpoint-verification scope is complete. "
    "The two-hour monitor is to be paused after final publication; the allocation remains "
    "retained. [status.py](status.py) reads state, [launch.py](launch.py) records the gated "
    "workflow, and [verify_launch.py](verify_launch.py) checks the launch. Checkpoints and "
    "large raw contact components remain remote; this report retains small receipts, "
    "per-chain summaries and compressed metrics."
)
text = text.replace(monitor_paragraph, ai(replacement).rstrip())
readme.write_text(text)
main = REPO / "README.md"
text = main.read_text()
old = next(p for p in text.split("\n\n") if p.startswith("Setting 3 [launched Stage 2"))
replacement = (
    "Setting 3 [completed Stage 2 on eight Nibi H100s](.dev/reports/"
    f"{REPORT.name}/README.md) on September 13: **300k additional updates, global "
    "400k → 700k**, batch **2,048**, context **2,048**, mixture **63% UniRef90 / 6% "
    "MGnify / 31% OMG**, and evaluation every **10k**. The "
    "[frozen recipe](.dev/configs/nibi/setting3-nibi-stage2-b2048-300k.yaml) "
    "decayed base LR from "
    "**5e-4 to 5e-5** while preserving optimizer and sampler history. The final "
    f"**700k** checkpoint reached validation loss **{final['validation_loss']:.6f}** "
    f"and P@L **{final['p_at_l'] * 100:.3f}%** (95% CI **{low * 100:.3f}–{high * 100:.3f}%**), "
    f"a **{comparison['p_at_l_difference_percentage_points']:.3f}-point** gain over its "
    "400k parent. All **30 evaluations** passed independent verification. The final "
    "full model/optimizer checkpoint is preserved in project storage and passed exact "
    "restoration; MGnify had zero repeats. Production took **48h 00m 39s** including "
    "evaluations and checkpoint handling."
)
main.write_text(text.replace(old + "\n\n", ai(replacement)))
for target in re.findall(r"\]\(([^)]+)\)", readme.read_text()):
    if not target.startswith(("http", "#")):
        assert (REPORT / target).exists(), target
assert sha(REPO / "autoresearch/program.md") == args.expected_program_sha256
files = [p for p in sorted(REPORT.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
(REPORT / "SHA256SUMS").write_text(
    "".join(f"{sha(p)}  {p.relative_to(REPORT)}\n" for p in files)
)
print(json.dumps(final, indent=2))
