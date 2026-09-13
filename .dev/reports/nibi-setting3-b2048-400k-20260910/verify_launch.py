"""Independently bind collected launch receipts to the committed production recipe."""

import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
LAUNCH = ROOT / "launch"


def read(path):
    return json.loads(path.read_text())


def metrics(name):
    return [
        r
        for line in (LAUNCH / name / "metrics.jsonl").read_text().splitlines()
        if (r := json.loads(line)).get("event") == "train"
    ]


recipe_path = REPO / ".dev/configs/nibi/setting3-nibi-b2048-400k.yaml"
recipe = yaml.safe_load(recipe_path.read_text())
actual = yaml.safe_load((LAUNCH / "full/config.yaml").read_text())
assert actual == recipe
assert recipe["max_steps"] == recipe["schedule_steps"] == 400000
assert recipe["checkpoint_interval"] == recipe["periodic_evaluation_interval"] == 10000
assert recipe["expected_world_size"] == 8
assert recipe["learning_rate"] == 5e-4 and recipe["weight_decay"] == 0.01
assert recipe["warmup_steps"] == 1000 and len(recipe["stages"]) == 1
stage = recipe["stages"][0]
assert stage["micro_batch_size"] == 64 and stage["gradient_accumulation"] == 4
assert stage["context_length"] == 512
assert recipe["training_loss_reduction"] == "sqrt_mask_count"
assert recipe["balance_batches_across_ranks"] is True
contract = read(LAUNCH / "full/run_contract.json")
gpu = read(LAUNCH / "LAUNCH_GPU_AND_PRESERVATION.json")
qualification = read(LAUNCH / "QUALIFICATION_PASSED.json")
migration = read(LAUNCH / "DATA_MIGRATION.json")
assert contract["config_sha256"] == hashlib.sha256(recipe_path.read_bytes()).hexdigest()
assert contract["config_overrides"] == {} and contract["git_dirty"] is False
assert contract["git_commit"] == gpu["source_commit"] == qualification["source_commit"]
assert contract["world_size"] == 8
assert contract["attention_kernel"]["implementation"] == "FlashAttention-3"
assert contract["resume_checkpoint_sha256"] == migration["parent_checkpoint_sha256"]
assert contract["resume_optimizer_step"] == 100000
assert contract["data_manifest_sha256"] == migration["new_manifest_sha256"]
assert gpu["status"] == qualification["status"] == "passed"
assert gpu["all_eight_h100s_owned_by_this_training_run"]
assert gpu["preserved_files"]["recipe.yaml"]["sha256"] == contract["config_sha256"]
assert read(LAUNCH / "full/DATA_COVERAGE.json") == read(
    ROOT / "PRODUCTION_CAPACITY_VERIFIED.json"
)
for name, endpoint in (("qualification", 100200), ("qualification-resume", 100210)):
    audit = read(LAUNCH / name / "TRAINING_VERIFIED.json")
    assert audit["status"] == "passed" and audit["optimizer_steps"] == endpoint
    assert audit["model_and_optimizer_roundtrip_exact"]
    assert audit["global_sampler_state_verified"]
rows = metrics("full")
trial = {r["optimizer_step"]: r for r in metrics("qualification")}
assert rows[-1]["optimizer_step"] >= 100300
shared = 0
for row in rows:
    assert row["sequences_seen"] == row["optimizer_step"] * 2048
    assert sum(row["source_counts_global"].values()) == row["sequences_seen"]
    assert all(
        math.isfinite(row[k])
        for k in ("loss", "objective_loss", "gradient_norm", "step_compute_seconds")
    )
    assert row["source_exposure_global"]["mgnify"]["repeated_draws"] == 0
    assert math.isclose(row["learning_rate"], 4.5e-4, rel_tol=1e-12)
    if row["optimizer_step"] in trial:
        shared += 1
        for key in (
            "source_counts_global",
            "source_exposure_global",
            "model_tokens",
            "sequences_seen",
        ):
            assert row[key] == trial[row["optimizer_step"]][key]
speed = statistics.median(r["step_compute_seconds"] for r in rows[-20:])
evaluation_seconds = read(LAUNCH / "qualification/PERIODIC_EVALUATION.json")["wall_seconds"]
started = datetime.datetime.fromisoformat(
    (LAUNCH / "full/launch-started-utc.txt").read_text().strip().replace("Z", "+00:00")
)
estimated_seconds = 300000 * speed + 30 * evaluation_seconds + 900
finish = started + datetime.timedelta(seconds=estimated_seconds)
result = {
    "status": "passed",
    "scope": "healthy production launch; no production evaluation endpoint completed yet",
    "source_commit": contract["git_commit"],
    "production_step": "12162637.32",
    "started_utc": started.isoformat(),
    "parent_optimizer_step": 100000,
    "target_total_optimizer_steps": 400000,
    "global_batch": 2048,
    "world_size": 8,
    "latest_collected_optimizer_step": rows[-1]["optimizer_step"],
    "finite_metric_rows": len(rows),
    "shared_trial_production_steps_with_identical_data_accounting": shared,
    "recipe_matches_local_source_actual_run_and_durable_copy": True,
    "qualification_optimizer_restores_exact": True,
    "production_median_seconds_per_step_last_200_updates": speed,
    "qualification_full_evaluation_seconds": evaluation_seconds,
    "estimate_overhead_allowance_seconds": 900,
    "estimated_launch_to_finish_hours": estimated_seconds / 3600,
    "estimated_finish_toronto": finish.astimezone(ZoneInfo("America/Toronto")).isoformat(),
    "estimate_note": (
        "Early throughput projection including 30 evaluations and 15 minutes overhead; "
        "allow roughly two hours uncertainty."
    ),
}
(ROOT / "LAUNCH_VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
