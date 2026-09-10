"""Gate production on a full-data eight-GPU continuation and checkpoint restore."""

import json
import math
import statistics
from pathlib import Path

ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-400k-20260910")


def read(path):
    return json.loads(path.read_text())


commit = (ROOT / "SOURCE_COMMIT.txt").read_text().strip()
metrics = []
for name, endpoint in (("qualification", 100200), ("qualification-resume", 100210)):
    out = ROOT / name
    contract = read(out / "run_contract.json")
    complete = read(out / "TRAINING_COMPLETE.json")
    audit = read(out / "TRAINING_VERIFIED.json")
    assert contract["git_commit"] == commit and not contract["git_dirty"]
    assert (
        contract["world_size"] == 8
        and contract["attention_kernel"]["implementation"] == "FlashAttention-3"
    )
    assert complete["optimizer_steps"] == endpoint and complete["stop_reason"] == "max_steps"
    assert complete["sequences_seen"] == endpoint * 2048
    assert audit["status"] == "passed" and audit["model_and_optimizer_roundtrip_exact"]
    assert audit["global_sampler_state_verified"]
    assert audit["source_exposure_global"]["mgnify"]["repeated_draws"] == 0
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    for row in rows:
        if row.get("event") != "train":
            continue
        assert row["optimizer_step"] > 100000
        assert all(
            math.isfinite(row[k])
            for k in ("loss", "objective_loss", "gradient_norm", "step_compute_seconds")
        )
        assert row["data_resampling"] == "per_source" and row["attention_backend"] == "flash3"
        assert row["training_loss_reduction"] == "sqrt_mask_count"
        assert len(row["batch_balance"]["rank_tokens_after"]) == 8
        assert sum(row["source_counts_global"].values()) == row["sequences_seen"]
        assert all(v["repeated_draws"] == 0 for v in row["source_exposure_global"].values())
        assert row["source_epoch_maxima"]["mgnify"] == 0
        if name == "qualification":
            metrics.append(row)
    if name == "qualification":
        assert complete["resume_data_mode"] == "verified_append_only_global_sampler_migration"
    else:
        assert complete["resume_data_mode"] == "restore_rank_rng_and_global_sampler"
evaluation = read(ROOT / "qualification/evaluations/step-100100/RESULT_VERIFIED.json")
assert evaluation["status"] == "passed" and evaluation["optimizer_steps"] == 100100
assert (
    evaluation["uncertainty"]["unit_count"] == 20775
    and evaluation["uncertainty"]["replicates"] == 5000
)
assert 2.0 < evaluation["validation"]["sequence_mean_nll"] < 3.0
assert 0.30 < evaluation["p_at_l"] < 0.50
speed = statistics.median(row["step_compute_seconds"] for row in metrics[-10:])
result = {
    "status": "passed",
    "source_commit": commit,
    "world_size": 8,
    "parent_optimizer_step": 100000,
    "qualification_endpoint": 100200,
    "resumed_endpoint": 100210,
    "global_batch": 2048,
    "full_evaluation_at_100100_and_training_continued": True,
    "full_optimizer_restored_twice": True,
    "no_premature_repeats": True,
    "median_seconds_per_step_last_100_updates": speed,
    "projected_additional_300k_training_seconds": speed * 300000,
    "qualification_evaluation": evaluation,
}
(ROOT / "QUALIFICATION_PASSED.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
