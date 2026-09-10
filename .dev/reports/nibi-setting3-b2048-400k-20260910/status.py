import datetime
import json
import subprocess
from pathlib import Path

ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-400k-20260910")


def read(p):
    return json.loads(p.read_text()) if p.exists() else None


def tail(p, n=10000):
    if not p.exists():
        return ""
    with p.open("rb") as f:
        f.seek(max(0, p.stat().st_size - n))
        return f.read().decode(errors="replace")


a = {
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "root": str(ROOT),
    "source_commit": (ROOT / "SOURCE_COMMIT.txt").read_text().strip(),
    "production_driver": read(ROOT / "PRODUCTION_DRIVER.json"),
    "preparation": read(ROOT / "PREPARATION.json"),
    "data_ready": (ROOT / "DATA_READY.json").exists(),
    "migration_verified": (ROOT / "DATA_MIGRATION.json").exists(),
    "qualification": read(ROOT / "QUALIFICATION_PASSED.json"),
    "runs": {},
}
for name in ["qualification", "qualification-resume", "full"]:
    p = ROOT / name
    rows = []
    for line in tail(p / "metrics.jsonl", 50000).splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("event") == "train":
            rows.append(r)
    a["runs"][name] = {
        "exists": p.exists(),
        "latest": rows[-1] if rows else None,
        "resume": read(p / "RESUME.json"),
        "evaluation": read(p / "PERIODIC_EVALUATION.json"),
        "complete": (p / "COMPLETE.txt").exists(),
        "failed": (p / "FAILED.txt").exists(),
        "training_complete": read(p / "TRAINING_COMPLETE.json"),
        "verification": read(p / "TRAINING_VERIFIED.json"),
    }
    if p.exists() and not rows:
        a["runs"][name]["log_tail"] = tail(p / "train.log", 1800)
a["driver_tail"] = tail(ROOT / "qualification-driver.log", 1600)
a["production_driver_tail"] = tail(ROOT / "production-driver.log", 1600)
a["latest_production_evaluation"] = None
for evaluation in sorted(
    (ROOT / "full/evaluations").glob("step-*/RESULT_VERIFIED.json"), reverse=True
):
    a["latest_production_evaluation"] = read(evaluation)
    break
a["allocation"] = subprocess.check_output(
    ["/opt/software/slurm/25.11.7p/bin/squeue", "-h", "-j", "12162637", "-o", "%i|%T|%L|%N"],
    text=True,
).strip()
a["steps"] = subprocess.check_output(
    [
        "/opt/software/slurm/25.11.7p/bin/squeue",
        "--steps",
        "-h",
        "-j",
        "12162637",
        "-o",
        "%i|%N",
    ],
    text=True,
).splitlines()
print(json.dumps(a, indent=2))
