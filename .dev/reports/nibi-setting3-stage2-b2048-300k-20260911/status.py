"""Lightweight live Stage 2 status; large model/checkpoint audits run in Slurm."""

import datetime
import json
import subprocess
from pathlib import Path

ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-setting3-stage2-b2048-300k-20260911")


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def tail(path, size=64000):
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode(errors="replace")


result = dict(
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    allocation=subprocess.check_output(
        [
            "/opt/software/slurm/25.11.7p/bin/squeue",
            "-j",
            "12162637,12162638",
            "-h",
            "-o",
            "%i|%T|%L|%e|%N",
        ],
        text=True,
    ).strip(),
    workflow=read(ROOT / "WORKFLOW.json"),
    data_ready=(ROOT / "DATA_READY.json").is_file(),
    transition=read(ROOT / "transition/TRAINING_VERIFIED.json"),
    qualification=read(ROOT / "QUALIFICATION_PASSED.json"),
)
for name in ("qualification", "qualification-resume", "full"):
    out = ROOT / name
    rows = []
    for line in tail(out / "metrics.jsonl").splitlines():
        try:
            row = json.loads(line)
            if row.get("event") == "train":
                rows.append(row)
        except json.JSONDecodeError:
            pass
    if out.exists():
        result[name] = dict(
            latest_train=rows[-1] if rows else None,
            recent_train=rows[-6:],
            complete=read(out / "TRAINING_COMPLETE.json"),
            periodic_evaluation=read(out / "PERIODIC_EVALUATION.json"),
            failure=tail(out / "FAILED.txt"),
            launch_identity=tail(out / "launch-identity.txt"),
            log_tail=tail(out / "train.log", 3000) if not rows else None,
        )
        evaluations = sorted((out / "evaluations").glob("step-*/RESULT_VERIFIED.json"))
        result[name]["evaluated_steps"] = [
            int(p.parent.name.split("-")[1]) for p in evaluations
        ]
        result[name]["latest_evaluation"] = read(evaluations[-1]) if evaluations else None
print(json.dumps(result, indent=2))
