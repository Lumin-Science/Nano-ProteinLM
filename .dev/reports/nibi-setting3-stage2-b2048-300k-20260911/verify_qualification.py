"""Require real Stage 2 continuation, full evaluation, and a full-state restart."""

import json
import math
import statistics
from pathlib import Path

ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-setting3-stage2-b2048-300k-20260911")


def read(p):
    return json.loads(p.read_text())


commit = (ROOT / "SOURCE_COMMIT.txt").read_text().strip()
transition = read(ROOT / "transition/checkpoint-stage2-start.json")
assert transition["status"] == "passed" and transition["model_optimizer_and_counters_exact"]
metrics = []
for name, endpoint in (("qualification", 400200), ("qualification-resume", 400210)):
    out = ROOT / name
    c = read(out / "run_contract.json")
    done = read(out / "TRAINING_COMPLETE.json")
    audit = read(out / "TRAINING_VERIFIED.json")
    assert c["git_commit"] == commit and not c["git_dirty"]
    assert (
        c["world_size"] == 8 and c["attention_kernel"]["implementation"] == "FlashAttention-3"
    )
    assert done["optimizer_steps"] == endpoint and done["stop_reason"] == "max_steps"
    assert done["sequences_seen"] == endpoint * 2048
    assert audit["status"] == "passed" and audit["model_and_optimizer_roundtrip_exact"]
    assert audit["global_sampler_state_verified"]
    assert audit["source_exposure_global"]["mgnify"]["repeated_draws"] == 0
    assert done["resume_data_mode"] == "restore_rank_rng_and_global_sampler"
    config = __import__("yaml").safe_load((out / "config.yaml").read_text())
    assert config["schedule_start_step"] == 400000 and config["schedule_steps"] == 700000
    assert (
        config["stages"][0]["name"] == "stage2"
        and config["stages"][0]["context_length"] == 2048
    )
    for source, x in audit["source_exposure_global"].items():
        assert x["draws"] >= transition["source_exposure_preserved"][source]["draws"]
    for line in (out / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("event") != "train":
            continue
        assert row["stage"] == "stage2" and row["optimizer_step"] > 400000
        assert all(
            math.isfinite(row[k])
            for k in ("loss", "objective_loss", "gradient_norm", "step_compute_seconds")
        )
        assert row["data_resampling"] == "per_source" and row["attention_backend"] == "flash3"
        assert row["training_loss_reduction"] == "sqrt_mask_count"
        assert len(row["batch_balance"]["rank_tokens_after"]) == 8
        assert sum(row["source_counts_global"].values()) == row["sequences_seen"]
        assert row["source_exposure_global"]["mgnify"]["repeated_draws"] == 0
        expected = (row["optimizer_step"] - 1 - 400000) / 300000
        assert abs(row["stage_progress"] - expected) < 1e-12
        assert abs(row["learning_rate"] - 0.00045 * (1 - 0.9 * expected)) < 1e-12
        if name == "qualification":
            metrics.append(row)
evaluation = read(ROOT / "qualification/evaluations/step-400100/RESULT_VERIFIED.json")
assert evaluation["status"] == "passed" and evaluation["optimizer_steps"] == 400100
assert (
    evaluation["uncertainty"]["unit_count"] == 20775
    and evaluation["uncertainty"]["replicates"] == 5000
)
assert 1.5 < evaluation["validation"]["sequence_mean_nll"] < 3.0
assert 0.30 < evaluation["p_at_l"] < 0.55
result = dict(
    status="passed",
    source_commit=commit,
    world_size=8,
    parent_optimizer_step=400000,
    qualification_endpoint=400200,
    resumed_endpoint=400210,
    global_batch=2048,
    evaluation_at_400100_and_training_continued=True,
    full_optimizer_restored_twice=True,
    mgnify_no_repeat=True,
    source_history_preserved=True,
    stage2_decay_clock_verified=True,
    median_compute_seconds_per_update=statistics.median(
        r["step_compute_seconds"] for r in metrics[-10:]
    ),
    qualification_evaluation=evaluation,
)
(ROOT / "QUALIFICATION_PASSED.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
