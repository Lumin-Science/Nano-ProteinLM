"""Write a small queue state receipt; safe for a lightweight login-node driver."""

import datetime as dt
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = dict(
    observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    state=sys.argv[2],
    detail=sys.argv[3],
    allocation="12162637",
    node="g27",
    predecessor_step="12162637.11",
    source_commit=(root / "SOURCE_COMMIT.txt").read_text().strip(),
)
if (root / "QUEUE_DRIVER_PID").exists():
    payload["driver_pid"] = int((root / "QUEUE_DRIVER_PID").read_text())
target = root / "QUEUE_STATUS.json"
temporary = target.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(target)
