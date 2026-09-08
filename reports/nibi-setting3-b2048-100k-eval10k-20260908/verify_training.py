"""Verify a completed Setting-3 run and full hybrid-optimizer checkpoint on CPU.

Run only within the allocated compute step; this loads the full model and state.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import yaml

from nano_protein.model import build_model
from nano_protein.train import _OptimizerBundle, build_optimizer

root = Path(sys.argv[1])
expected_commit = sys.argv[2]
config = yaml.safe_load((root / "config.yaml").read_text())
complete = json.loads((root / "TRAINING_COMPLETE.json").read_text())
contract = json.loads((root / "run_contract.json").read_text())
assert contract["git_commit"] == expected_commit and not contract["git_dirty"]
assert (
    contract["config_sha256"] == hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest()
)
assert contract["world_size"] == config["expected_world_size"]
assert contract["attention_kernel"]["revision"] == "e29f138fc363b396e5d2706c8a5f6fa7d36f41e0"
assert contract["attention_kernel"]["implementation"] == "FlashAttention-3"
assert (
    contract["data_manifest_sha256"]
    == "43675d51421066ce8c5f68427886d57980e808c53c5bb1641de90cb74dda39ab"
)
assert complete["stop_reason"] == "max_steps"
assert complete["optimizer_steps"] == config["max_steps"]
assert complete["sequences_seen"] == config["max_steps"] * 2048
assert complete["parameter_count"] == 170559856
rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
rows = [row for row in rows if row.get("event") == "train"]
assert rows[-1]["optimizer_step"] == config["max_steps"]
for row in rows:
    assert all(
        math.isfinite(row[k])
        for k in (
            "loss",
            "objective_loss",
            "gradient_norm",
            "step_compute_seconds",
            "learning_rate",
        )
    )
    assert row["sequences_seen"] == row["optimizer_step"] * 2048
    assert row["attention_backend"] == "flash3" and row["optimizer"] == "muon"
    assert row["training_loss_reduction"] == "sqrt_mask_count"
    balance = row["batch_balance"]
    assert len(balance["rank_tokens_before"]) == contract["world_size"]
    assert sum(balance["rank_tokens_before"]) == sum(balance["rank_tokens_after"])
assert abs(rows[-1]["learning_rate"] - 4.5e-4) < 1e-12
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
optimizer_packet = packet["optimizer"]
assert optimizer_packet["class"] == "_OptimizerBundle"
children = optimizer_packet["optimizers"]
assert [c["name"] for c in children] == ["muon", "adamw"]
counts = {}
for child in children:
    state = child["state_dict"]
    ids = [p for group in state["param_groups"] for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(state["state"])
    counts[child["name"]] = len(ids)
    for values in state["state"].values():
        for value in values.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all().item()
        if child["name"] == "adamw":
            assert float(values["step"]) == config["max_steps"]
            assert (values["exp_avg_sq"] >= 0).all().item()
        else:
            assert "momentum_buffer" in values
assert counts == {"muon": 96, "adamw": 9}

# Restore the full model and both optimizers, and compare their serialized state
# exactly. This also exercises ownership and the bundle's live param-group links.
options = {
    k: config[k]
    for k in (
        "learned_residual_routing",
        "transformer_norm",
        "depth_scaled_residual_init",
        "rotary_base",
        "ffn_hidden_dim",
        "tie_word_embeddings",
    )
}
model = build_model(config["model"], attention_backend="math", **options)
model.load_state_dict(packet["model"])
optimizer = build_optimizer(model, config)
assert isinstance(optimizer, _OptimizerBundle)
optimizer.load_state_dict(optimizer_packet)


def exact(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, list | tuple):
        assert len(a) == len(b)
        for first, second in zip(a, b, strict=True):
            exact(first, second)
    else:
        assert a == b


exact(packet["model"], model.state_dict())
exact(optimizer_packet, optimizer.state_dict())
owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
assert len(owned) == len(set(owned)) == len(list(model.parameters()))
assert {id(p) for p in model.parameters()} == set(owned)
expected_groups = {
    "muon_attention": (4.5e-4, 0.0075),
    "muon_ffn": (3.75e-4, 0.0075),
    "adamw_decay": (5e-4, 0.01),
    "adamw_no_decay": (5e-4, 0.0),
}
for group in optimizer.param_groups:
    lr, wd = expected_groups[group["name"]]
    assert math.isclose(group["lr"], lr, rel_tol=0, abs_tol=1e-12)
    assert group["weight_decay"] == wd
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
    parameter_count=170559856,
    optimizer_parameters_with_state=sum(counts.values()),
    optimizer_children=counts,
    optimizer_layout=packet["optimizer_layout"],
    finite_model_and_optimizer=True,
    model_and_optimizer_roundtrip_exact=True,
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
