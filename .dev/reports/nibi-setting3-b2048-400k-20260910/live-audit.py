import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-400k-20260910")
DEST = Path(
    "/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-setting3-b2048-400k-20260910"
)
REPO = Path(str(ROOT) + "-run")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def query(kind):
    lines = subprocess.check_output(
        ["nvidia-smi", kind, "--format=csv,noheader,nounits"], text=True
    )
    return [[item.strip() for item in row] for row in csv.reader(lines.splitlines())]


assert os.environ["SLURM_JOB_ID"] == "12162637"
assert subprocess.check_output(["hostname", "-s"], text=True).strip() == "g27"
gpus = query("--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total")
processes = query("--query-compute-apps=pid,gpu_uuid,used_memory")
assert len(gpus) == 8 and len(processes) == 8
assert {g[1] for g in gpus} == {p[1] for p in processes}
assert all("H100" in g[2] for g in gpus)
for p in processes:
    command = (Path("/proc") / p[0] / "cmdline").read_bytes().split(b"\0")
    assert str(ROOT / "full").encode() in command
    assert b"nanoprotein.train" in command
proof = {}
for source, target in [
    (REPO / "configs/setting3-nibi-b2048-400k.yaml", DEST / "recipe.yaml")
] + [
    (ROOT / name, DEST / name)
    for name in ["DATA_PLAN.json", "DATA_MIGRATION.json", "SOURCE_COMMIT.txt", "source.bundle"]
]:
    expected = digest(source)
    assert digest(target) == expected
    proof[target.name] = {
        "status": "passed",
        "source": str(source),
        "preserved": str(target),
        "sha256": expected,
        "bytes": target.stat().st_size,
    }
contract = json.loads((ROOT / "full/run_contract.json").read_text())
assert contract["world_size"] == 8 and contract["git_dirty"] is False
assert contract["git_commit"] == (ROOT / "SOURCE_COMMIT.txt").read_text().strip()
assert contract["config_overrides"] == {}
assert contract["resume_optimizer_step"] == 100000
assert (
    contract["resume_checkpoint_sha256"]
    == "d0f891cfd46e5517e1ce90a29f02bc88646a80023b7cf8e86fc3008311b66412"
)
assert contract["attention_kernel"]["implementation"] == "FlashAttention-3"
assert contract["config_sha256"] == proof["recipe.yaml"]["sha256"]
rows = [json.loads(line) for line in (ROOT / "full/metrics.jsonl").read_text().splitlines()]
rows = [r for r in rows if r.get("event") == "train"]
assert len(rows) >= 10
for row in rows:
    assert all(
        math.isfinite(row[k])
        for k in ["loss", "objective_loss", "gradient_norm", "step_compute_seconds"]
    )
    assert row["sequences_seen"] == row["optimizer_step"] * 2048
    assert sum(row["source_counts_global"].values()) == row["sequences_seen"]
    assert row["source_exposure_global"]["mgnify"]["repeated_draws"] == 0
    assert len(row["batch_balance"]["rank_tokens_after"]) == 8
receipt = {
    "status": "passed",
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "allocation": "12162637",
    "node": "g27",
    "source_commit": contract["git_commit"],
    "production_from_original_100k_checkpoint": True,
    "all_eight_h100s_owned_by_this_training_run": True,
    "gpu_columns": [
        "index",
        "uuid",
        "name",
        "utilization_percent",
        "memory_used_mib",
        "memory_total_mib",
    ],
    "gpus": gpus,
    "process_columns": ["pid", "gpu_uuid", "used_memory_mib"],
    "processes": processes,
    "preserved_files": proof,
    "finite_production_metric_rows": len(rows),
    "latest": rows[-1],
}
(ROOT / "LAUNCH_GPU_AND_PRESERVATION.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
