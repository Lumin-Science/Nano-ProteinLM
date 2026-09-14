"""Read small Atlas run receipts without loading checkpoints or training data."""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-atlas-setting3-b1024-20260914")


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


result = dict(checked_utc=datetime.now(timezone.utc).isoformat())
for mode in ("pilot", "full"):
    out = ROOT / f"{mode}-training"
    metrics = out / "metrics.jsonl"
    latest = []
    if metrics.is_file():
        with metrics.open("rb") as handle:
            handle.seek(max(0, metrics.stat().st_size - 20000))
            for line in handle.read().splitlines()[-10:]:
                try:
                    latest.append(json.loads(line))
                except (ValueError, UnicodeDecodeError):
                    pass
    evaluations = sorted((out / "evaluations").glob("step-*/RESULT_VERIFIED.json"))
    result[mode] = dict(
        launch=read(out / "LAUNCH.json"),
        latest_metrics=latest[-2:],
        evaluation=read(out / "PERIODIC_EVALUATION.json"),
        latest_result=read(evaluations[-1]) if evaluations else None,
        final_result=read(out / "final-evaluation/RESULT_VERIFIED.json"),
        complete=read(out / "TRAINING_COMPLETE.json"),
        qualified=read(out / "PILOT_PASSED.json"),
    )
result["data_preserved"] = read(ROOT / "DATA_PRESERVED.json")
print(json.dumps(result, indent=2))
