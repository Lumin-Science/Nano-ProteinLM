"""Verify the live production launch and preserved recipe on allocated hardware."""

import csv
import datetime
import hashlib
import io
import json
import math
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-setting3-stage2-b2048-300k-20260911")
REPO = Path(str(ROOT) + "-run")
DURABLE = Path(
    "/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-setting3-stage2-b2048-300k-20260911"
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


out = ROOT / "full"
c = read(out / "run_contract.json")
resume = read(out / "RESUME.json")
coverage = read(out / "DATA_COVERAGE.json")
config = yaml.safe_load((out / "config.yaml").read_text())
source = (ROOT / "SOURCE_COMMIT.txt").read_text().strip()
assert c["git_commit"] == source and not c["git_dirty"]
assert (
    subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == source
)
assert not subprocess.check_output(
    ["git", "status", "--porcelain"], cwd=REPO, text=True
).strip()
assert c["world_size"] == 8 and c["attention_kernel"]["implementation"] == "FlashAttention-3"
assert (
    config["max_steps"] == config["schedule_steps"] == 700000
    and config["schedule_start_step"] == 400000
)
stage = config["stages"][0]
assert stage["name"] == "stage2" and stage["context_length"] == 2048
assert stage["micro_batch_size"] == 64 and stage["gradient_accumulation"] == 4
assert stage["mixture"] == dict(uniref90=0.63, mgnify=0.06, omg_img=0.31)
assert config["checkpoint_interval"] == config["periodic_evaluation_interval"] == 10000
assert config["periodic_evaluation_command"] == ["bash", "runs/evaluate_global_checkpoint.sh"]
assert config["learning_rate"] == 0.0005 and config["weight_decay"] == 0.01
assert (
    resume["optimizer_step"] == 400000
    and resume["optimizer_restored"]
    and resume["global_batch_preserved"]
)
assert resume["data_mode"] == "restore_rank_rng_and_global_sampler"
assert (
    c["resume_checkpoint_sha256"]
    == read(ROOT / "transition/checkpoint-stage2-start.json")["transition_checkpoint_sha256"]
)
assert coverage["status"] == "passed" and coverage["stages"][0]["steps"] == 300000
assert coverage["sources"]["mgnify"]["unused_records"] == 41496700
assert coverage["sources"]["mgnify"]["sufficient"]
assert config == yaml.safe_load((DURABLE / "recipe.yaml").read_text())
assert sha(ROOT / "source.bundle") == sha(DURABLE / "source.bundle")
metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
rows = [r for r in metrics if r.get("event") == "train"]
assert rows[-1]["optimizer_step"] >= 400100
for r in rows:
    assert r["stage"] == "stage2" and r["sequences_seen"] == r["optimizer_step"] * 2048
    assert sum(r["source_counts_global"].values()) == r["sequences_seen"]
    assert r["source_exposure_global"]["mgnify"]["repeated_draws"] == 0
    assert (
        r["attention_backend"] == "flash3" and r["training_loss_reduction"] == "sqrt_mask_count"
    )
    assert len(r["batch_balance"]["rank_tokens_after"]) == 8
    assert all(
        math.isfinite(r[k])
        for k in ("loss", "objective_loss", "gradient_norm", "step_compute_seconds")
    )
    progress = (r["optimizer_step"] - 1 - 400000) / 300000
    assert abs(r["stage_progress"] - progress) < 1e-12
    assert abs(r["learning_rate"] - 0.00045 * (1 - 0.9 * progress)) < 1e-12
first = next(r for r in rows if r["optimizer_step"] >= 400020)
last = rows[-1]
seconds = (last["training_seconds"] - first["training_seconds"]) / (
    last["optimizer_step"] - first["optimizer_step"]
)
gpu_csv = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used", "--format=csv"],
    text=True,
)
gpus = [
    {k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(io.StringIO(gpu_csv))
]
assert len(gpus) == 8 and all("H100" in g["name"] for g in gpus)
assert all(int(g["memory.used [MiB]"].split()[0]) > 5000 for g in gpus)
(out / "launch-gpu-status.csv").write_text(gpu_csv)
now = datetime.datetime.now(datetime.timezone.utc)
evaluation_seconds = read(ROOT / "qualification/evaluations/step-400100/EVALUATION_RUN.json")[
    "wall_seconds"
]
remaining = 700000 - last["optimizer_step"]
eta = now + datetime.timedelta(seconds=remaining * seconds + 30 * evaluation_seconds)
slow_eta = now + datetime.timedelta(seconds=remaining * seconds * 1.1 + 30 * evaluation_seconds)
first_eval = now + datetime.timedelta(
    seconds=(410000 - last["optimizer_step"]) * seconds + evaluation_seconds
)
receipt = dict(
    status="passed",
    checked_utc=now.isoformat(),
    source_commit=source,
    slurm_identity=(out / "launch-identity.txt").read_text(),
    global_batch=2048,
    world_size=8,
    context_length=2048,
    additional_steps=300000,
    target_global_step=700000,
    checkpoint_interval=10000,
    periodic_evaluation_interval=10000,
    observed_step=last["optimizer_step"],
    last_training_record=last,
    steady_training_seconds_per_update=seconds,
    evaluation_seconds=evaluation_seconds,
    estimated_finish_toronto=eta.astimezone(ZoneInfo("America/Toronto")).isoformat(),
    estimated_finish_10pct_slower_toronto=slow_eta.astimezone(
        ZoneInfo("America/Toronto")
    ).isoformat(),
    estimated_first_410k_evaluation_toronto=first_eval.astimezone(
        ZoneInfo("America/Toronto")
    ).isoformat(),
    finite_training_metrics=True,
    full_optimizer_and_source_history_restored=True,
    stage2_decay_clock_verified=True,
    recipe_and_source_bundle_preserved=True,
    gpus=gpus,
)
(ROOT / "LAUNCH_VERIFIED.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
