"""Read-only launch/status summary; safe to run on the Nibi login node."""

import datetime as dt
import json
import time
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path("/scratch/muchenli/Nano-Protein-LM-nibi-b2048-100k-20260908")
now = time.time()
result = {
    "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "transfers_complete": [p.name for p in root.glob("TRANSFER_*_COMPLETE")],
    "qualification_passed": (root / "QUALIFICATION_PASSED.json").exists(),
    "failures": {p.name: p.read_text() for p in root.glob("*FAILED.txt")},
    "runs": {},
}
for name in ("trial-8gpu", "resume-8gpu", "resume-4gpu", "full"):
    path = root / name
    if not path.exists():
        continue
    r = {"state": "starting", "path": str(path)}
    for filename, key in [
        ("RESUME.json", "resume"),
        ("TRAINING_COMPLETE.json", "completion"),
        ("TRAINING_VERIFIED.json", "verification"),
        ("LATEST_CHECKPOINT.json", "latest_checkpoint"),
    ]:
        if (path / filename).exists():
            r[key] = json.loads((path / filename).read_text())
    if (path / "metrics.jsonl").exists():
        metrics = path / "metrics.jsonl"
        with metrics.open("rb") as handle:
            handle.seek(max(0, metrics.stat().st_size - 150000))
            lines = handle.read().splitlines()
        rows = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "train":
                rows.append(row)
        if rows:
            r.update(
                state="training",
                latest=rows[-1],
                metrics_age_seconds=now - metrics.stat().st_mtime,
            )
            if len(rows) > 1:
                a, b = rows[0], rows[-1]
                rate = (b["training_seconds"] - a["training_seconds"]) / (
                    b["optimizer_step"] - a["optimizer_step"]
                )
                r["seconds_per_step"] = rate
                if name == "full" and "completion" not in r:
                    remaining = (100000 - b["optimizer_step"]) * rate
                    r["remaining_hours"] = remaining / 3600
                    r["training_eta_toronto"] = dt.datetime.fromtimestamp(
                        now + remaining, ZoneInfo("America/Toronto")
                    ).isoformat()
    if "completion" in r:
        r["state"] = "training_finished"
    if (path / "COMPLETE.txt").exists():
        r["state"] = "complete"
    if (path / "CHECKPOINT_PRESERVED.txt").exists():
        r["checkpoint_preserved"] = True
    if (path / "eval-validation/VALIDATION_MLM.json").exists():
        r["validation"] = json.loads((path / "eval-validation/VALIDATION_MLM.json").read_text())
    if (path / "eval-p-at-l/P_AT_L_UNCERTAINTY.json").exists():
        r["p_at_l"] = json.loads((path / "eval-p-at-l/P_AT_L_UNCERTAINTY.json").read_text())
    result["runs"][name] = r
print(json.dumps(result, indent=2))
