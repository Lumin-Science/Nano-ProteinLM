"""Small, read-only status probe suitable for a Nibi login node."""

import datetime
import json
import math
import sys
from pathlib import Path

root = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909"
)
now = datetime.datetime.now(datetime.timezone.utc)


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


result = {
    "observed_utc": now.isoformat(),
    "root": str(root),
    "data_ready": (root / "DATA_READY.json").exists(),
    "qualification_passed": (root / "QUALIFICATION_PASSED.json").exists(),
    "preparation": read(root / "PREPARATION.json"),
    "runs": {},
}
for name in ("baseline", "setting3"):
    out = root / "full" / name
    state = {
        "physical_gpus": [0, 1, 2, 3] if name == "baseline" else [4, 5, 6, 7],
        "status": "preparing",
        "output_root": str(out),
    }
    metrics = out / "metrics.jsonl"
    if metrics.exists():
        with metrics.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 65536))
            lines = handle.read().splitlines()
        rows = []
        for line in lines:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") == "train":
                rows.append(row)
        if rows:
            latest = rows[-1]
            state.update(
                status="training",
                latest=latest,
                metrics_age_seconds=now.timestamp() - metrics.stat().st_mtime,
            )
            durations = [r["step_compute_seconds"] for r in rows[-20:]]
            seconds_per_step = sum(durations) / len(durations)
            evals = [read(p) for p in out.glob("evaluations/step-*/EVALUATION_RUN.json")]
            evals = [e["wall_seconds"] for e in evals if e and e.get("status") == "passed"]
            evaluation_seconds = sum(evals) / len(evals) if evals else 300
            remaining = max(0, 100000 - latest["optimizer_step"])
            eta = now + datetime.timedelta(
                seconds=remaining * seconds_per_step
                + math.ceil(remaining / 10000) * evaluation_seconds
            )
            state.update(
                estimated_seconds_per_step=seconds_per_step,
                estimated_finish_utc=eta.isoformat(),
            )
    evaluation = read(out / "PERIODIC_EVALUATION.json")
    state["evaluation"] = evaluation
    if evaluation and evaluation["status"] == "running":
        state["status"] = "evaluating"
    if (out / "FAILED.txt").exists():
        state["status"] = "failed"
    if (out / "COMPLETE.txt").exists():
        state["status"] = "complete"
        state["completion"] = read(out / "TRAINING_VERIFIED.json")
    state["checkpoint_preserved"] = (out / "CHECKPOINT_PRESERVED.txt").exists()
    result["runs"][name] = state
print(json.dumps(result, indent=2))
