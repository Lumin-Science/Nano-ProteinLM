"""Drive the user-authorized transition, qualification and production on Nibi."""

import datetime
import json
import math
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-setting3-stage2-b2048-300k-20260911")
SLURM = "/opt/software/slurm/25.11.7p/bin"


def stamp(phase, **kw):
    value = dict(
        phase=phase, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kw
    )
    (ROOT / "WORKFLOW.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value), flush=True)


def allocation():
    row = subprocess.check_output(
        [SLURM + "/squeue", "-h", "-j", "12162637", "-o", "%i|%T|%L|%N"], text=True
    ).strip()
    job, state, left, node = row.split("|")
    if job != "12162637" or state != "RUNNING" or node != "g27":
        raise RuntimeError("Expected Nibi allocation is unavailable: " + row)
    days, clock = left.split("-", 1) if "-" in left else ("0", left)
    parts = [int(v) for v in clock.split(":")]
    seconds = int(days) * 86400 + sum(v * 60**i for i, v in enumerate(reversed(parts)))
    return row, seconds


def run(mode, gpus, minutes):
    row, remaining = allocation()
    command = [
        SLURM + "/srun",
        "--jobid=12162637",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=32",
        "--mem=256G",
        "--gres=" + ("gpu:8" if gpus else "none"),
        "--time=" + str(minutes),
        "bash",
        str(ROOT) + "-run/runs/nibi_setting3_stage2.sh",
        mode,
    ]
    stamp(mode, allocation=row, command=command)
    with (ROOT / (mode + "-driver.log")).open("xb") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
        )
        receipt = dict(
            controller_pid=process.pid,
            controller_host=socket.gethostname(),
            dispatched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            command=command,
            allocation_at_dispatch=row,
        )
        (ROOT / (mode.upper() + "_DRIVER.json")).write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        code = process.wait()
    if code:
        raise RuntimeError(f"{mode} exited with code {code}; see {mode}-driver.log")


def main():
    assert not (ROOT / "WORKFLOW.json").exists()
    stamp("waiting_for_verified_data")
    started = time.monotonic()
    while not (ROOT / "DATA_READY.json").exists():
        if time.monotonic() - started > 3600:
            raise TimeoutError("Data preparation did not finish within one hour")
        time.sleep(15)
    assert json.loads((ROOT / "DATA_READY.json").read_text())["status"] == "passed"
    run("transition", False, 45)
    run("qualification", True, 60)
    qualification = json.loads((ROOT / "QUALIFICATION_PASSED.json").read_text())
    assert qualification["status"] == "passed"
    _, remaining = allocation()
    estimate = (
        qualification["median_compute_seconds_per_update"] * 1.1 * 300000 + 30 * 260 + 900
    )
    if remaining < estimate:
        raise RuntimeError(
            f"Remaining allocation {remaining}s is below the estimated budget {estimate}s"
        )
    run("production", True, min(3600, math.floor((remaining - 600) / 60)))
    stamp("complete", status="passed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        stamp("failed", status="failed", error=repr(exc))
        raise
