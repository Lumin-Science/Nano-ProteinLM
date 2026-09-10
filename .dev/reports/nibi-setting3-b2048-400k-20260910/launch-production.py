"""Dispatch the qualified, user-authorized continuation in the existing allocation."""

import datetime
import json
import socket
import subprocess
from pathlib import Path

ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-400k-20260910")
assert json.loads((ROOT / "QUALIFICATION_PASSED.json").read_text())["status"] == "passed"
assert (ROOT / "qualification/COMPLETE.txt").exists()
assert not (ROOT / "full").exists()
assert not (ROOT / "PRODUCTION_DRIVER.json").exists()
allocation = subprocess.check_output(
    ["/opt/software/slurm/25.11.7p/bin/squeue", "-h", "-j", "12162637", "-o", "%i|%T|%L|%N"],
    text=True,
).strip()
assert allocation.startswith("12162637|RUNNING|") and allocation.endswith("|g27"), allocation
command = [
    "/opt/software/slurm/25.11.7p/bin/srun",
    "--jobid=12162637",
    "--overlap",
    "--nodes=1",
    "--ntasks=1",
    "--cpus-per-task=32",
    "--mem=256G",
    "--gres=gpu:8",
    "--time=72:00:00",
    "bash",
    str(ROOT) + "-run/runs/nibi_setting3_400k.sh",
    "production",
]
with (ROOT / "production-driver.log").open("xb") as log:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=ROOT,
    )
receipt = {
    "controller_pid": process.pid,
    "controller_host": socket.gethostname(),
    "dispatched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": command,
    "allocation_at_dispatch": allocation,
}
(ROOT / "PRODUCTION_DRIVER.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
