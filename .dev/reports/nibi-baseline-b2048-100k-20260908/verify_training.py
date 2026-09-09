"""Run on the allocated compute node after each launch/continuation."""

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import yaml

root = Path(sys.argv[1])
expected_commit = sys.argv[2]
config = yaml.safe_load((root / "config.yaml").read_text())
complete = json.loads((root / "TRAINING_COMPLETE.json").read_text())
contract = json.loads((root / "run_contract.json").read_text())
assert contract["git_commit"] == expected_commit and not contract["git_dirty"]
assert contract["world_size"] == config["expected_world_size"]
assert contract["attention_kernel"]["implementation"] == "FlashAttention-3"
assert contract["attention_kernel"]["revision"] == "e29f138fc363b396e5d2706c8a5f6fa7d36f41e0"
assert (
    contract["data_manifest_sha256"]
    == "43675d51421066ce8c5f68427886d57980e808c53c5bb1641de90cb74dda39ab"
)
assert complete["stop_reason"] == "max_steps"
assert complete["optimizer_steps"] == config["max_steps"]
assert complete["sequences_seen"] == config["max_steps"] * 2048
assert complete["parameter_count"] == 170671168
rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
rows = [row for row in rows if row.get("event") == "train"]
assert rows[-1]["optimizer_step"] == config["max_steps"]
for row in rows:
    assert all(
        math.isfinite(row[k])
        for k in ("loss", "gradient_norm", "step_compute_seconds", "learning_rate")
    )
    assert row["sequences_seen"] == row["optimizer_step"] * 2048
    assert row["attention_backend"] == "flash3"
assert abs(rows[-1]["learning_rate"] - 5e-4) < 1e-12
checkpoint = root / "checkpoint-final.pt"
with checkpoint.open("rb") as handle:
    digest = hashlib.file_digest(handle, "sha256").hexdigest()
assert digest == complete["final_checkpoint"]["sha256"]
packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert packet["train_config"] == config
assert packet["world_size"] == config["expected_world_size"]
assert len(packet["runtime_states"]) == packet["world_size"]
assert packet["optimizer_layout"] == "replicated_ddp_full_state"
assert packet["data_manifest_sha256"] == contract["data_manifest_sha256"]
assert packet["optimizer_step"] == config["max_steps"]
assert all(torch.isfinite(t).all().item() for t in packet["model"].values())
optimizer = packet["optimizer"]
parameter_ids = [p for group in optimizer["param_groups"] for p in group["params"]]
assert len(parameter_ids) == len(set(parameter_ids)) == len(optimizer["state"])
for state in optimizer["state"].values():
    assert float(state["step"]) == config["max_steps"]
    assert torch.isfinite(state["exp_avg"]).all().item()
    assert torch.isfinite(state["exp_avg_sq"]).all().item()
    assert (state["exp_avg_sq"] >= 0).all().item()
recent = rows[-10:]
seconds_per_step = (recent[-1]["training_seconds"] - recent[0]["training_seconds"]) / (
    recent[-1]["optimizer_step"] - recent[0]["optimizer_step"]
)
receipt = dict(
    status="passed",
    checkpoint_sha256=digest,
    config_sha256=contract["config_sha256"],
    optimizer_steps=config["max_steps"],
    sequences_seen=complete["sequences_seen"],
    model_tokens=complete["model_tokens"],
    world_size=packet["world_size"],
    optimizer_state_tensors=len(parameter_ids),
    optimizer_layout=packet["optimizer_layout"],
    finite_model_and_optimizer=True,
    flash3_verified=True,
    training_seconds=complete["training_seconds"],
    measured_seconds_per_step=seconds_per_step,
    peak_cuda_memory_bytes=complete["peak_cuda_memory_bytes"],
)
if (root / "RESUME.json").exists():
    receipt["resume"] = json.loads((root / "RESUME.json").read_text())
    assert (
        receipt["resume"]["optimizer_restored"] and receipt["resume"]["global_batch_preserved"]
    )
(root / "TRAINING_VERIFIED.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
