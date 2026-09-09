"""Read-only status of training and completed checkpoint evaluations."""

import datetime as dt
import json
import time
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path("/scratch/muchenli/Nano-Protein-LM-nibi-b2048-100k-eval10k-20260908")
now = time.time()
gate_path = root / "PERIODIC_EVALUATION_TRIAL_PASSED.json"
gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
report = {
    "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "qualification_passed": gate.get("status") == "passed",
    "runs": {},
}
for name in ("trial", "full"):
    path = root / name
    if not path.exists():
        continue
    run = {"state": "starting", "output_root": str(path), "evaluations": []}
    metrics = path / "metrics.jsonl"
    if metrics.exists():
        rows = []
        with metrics.open("rb") as handle:
            handle.seek(max(0, metrics.stat().st_size - 150000))
            lines = handle.read().splitlines()
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "train":
                rows.append(row)
        if rows:
            run.update(
                state="training",
                latest=rows[-1],
                metrics_age_seconds=now - metrics.stat().st_mtime,
            )
        if len(rows) > 1:
            a, b = rows[0], rows[-1]
            run["seconds_per_step"] = (b["training_seconds"] - a["training_seconds"]) / (
                b["optimizer_step"] - a["optimizer_step"]
            )
    for filename, key in [
        ("TRAINING_COMPLETE.json", "training_completion"),
        ("PERIODIC_EVALUATION.json", "periodic_evaluation"),
    ]:
        if (path / filename).exists():
            run[key] = json.loads((path / filename).read_text())
    for result in sorted(path.glob("evaluations/step-*/RESULT_VERIFIED.json")):
        value = json.loads(result.read_text())
        run["evaluations"].append(
            {
                "step": value["optimizer_steps"],
                "validation_loss": value["validation"]["sequence_mean_nll"],
                "p_at_l": value["p_at_l"],
                "uncertainty": value["uncertainty"],
                "checkpoint_sha256": value["checkpoint_sha256"],
            }
        )
    if "training_completion" in run:
        run["state"] = "final_evaluation"
    if run.get("periodic_evaluation", {}).get("status") == "running":
        run["state"] = "evaluating"
    if (path / "COMPLETE.txt").exists():
        run["state"] = "complete"
    if (path / "FAILED.txt").exists():
        run.update(state="failed", failure=(path / "FAILED.txt").read_text())
    if (path / "CHECKPOINT_PRESERVED.txt").exists():
        run["checkpoint_preserved"] = True
    if (
        name == "full"
        and "seconds_per_step" in run
        and run["state"] not in ("complete", "failed")
    ):
        step = run["latest"]["optimizer_step"]
        training_left = max(0, 100000 - step) * run["seconds_per_step"]
        evaluation_time = gate.get("execution", {}).get("wall_seconds", 0)
        evaluations_left = 10 - len(run["evaluations"])
        evaluation_left = evaluation_time * evaluations_left
        if run["state"] == "evaluating":
            elapsed = (
                now
                - dt.datetime.fromisoformat(
                    run["periodic_evaluation"]["started_utc"]
                ).timestamp()
            )
            evaluation_left = max(0, evaluation_left - min(elapsed, evaluation_time))
        run["remaining_hours_estimate"] = (training_left + evaluation_left) / 3600
        run["estimated_completion_toronto"] = dt.datetime.fromtimestamp(
            now + training_left + evaluation_left, ZoneInfo("America/Toronto")
        ).isoformat()
    report["runs"][name] = run
print(json.dumps(report, indent=2))
