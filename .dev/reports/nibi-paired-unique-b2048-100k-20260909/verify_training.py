"""Audit either paired run and restore its full optimizer on allocated CPUs."""

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import yaml

from nanoprotein.model import build_model
from nanoprotein.train import build_optimizer

root = Path(sys.argv[1])
commit = sys.argv[2]
config = yaml.safe_load((root / "config.yaml").read_text())
complete = json.loads((root / "TRAINING_COMPLETE.json").read_text())
contract = json.loads((root / "run_contract.json").read_text())
assert contract["git_commit"] == commit and not contract["git_dirty"]
assert contract["world_size"] == config["expected_world_size"] == 4
assert (
    contract["config_sha256"] == hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest()
)
assert contract["attention_kernel"]["implementation"] == "FlashAttention-3"
assert contract["attention_kernel"]["revision"] == "e29f138fc363b396e5d2706c8a5f6fa7d36f41e0"
assert contract["data_coverage"]["status"] == "passed"
assert config["data_resampling"] == "error"
assert complete["stop_reason"] == "max_steps"
assert complete["optimizer_steps"] == config["max_steps"]
assert complete["sequences_seen"] == config["max_steps"] * 2048
assert complete["parameter_count"] == config["expected_parameter_count"]
assert all(v == 0 for s in complete["source_epoch_maxima"].values() for v in s.values())
rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
rows = [r for r in rows if r.get("event") == "train"]
assert rows[-1]["optimizer_step"] == config["max_steps"]
for row in rows:
    assert all(
        math.isfinite(row[k])
        for k in ("loss", "gradient_norm", "learning_rate", "step_compute_seconds")
    )
    assert row["sequences_seen"] == row["optimizer_step"] * 2048
    assert sum(row["source_counts_global"].values()) == row["sequences_seen"]
    assert all(v == 0 for v in row["source_epoch_maxima"].values())
    if config["optimizer"] == "muon":
        assert math.isfinite(row["objective_loss"])
        assert row["training_loss_reduction"] == "sqrt_mask_count"
        balance = row["batch_balance"]
        assert len(balance["rank_tokens_before"]) == 4
        assert sum(balance["rank_tokens_before"]) == sum(balance["rank_tokens_after"])
checkpoint = root / "checkpoint-final.pt"
with checkpoint.open("rb") as f:
    digest = hashlib.file_digest(f, "sha256").hexdigest()
assert digest == complete["final_checkpoint"]["sha256"]
packet = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert packet["train_config"] == config
assert packet["optimizer_step"] == config["max_steps"]
assert packet["optimizer_layout"] == "replicated_ddp_full_state"
assert packet["data_manifest_sha256"] == contract["data_manifest_sha256"]
assert len(packet["runtime_states"]) == 4
for rank, runtime in enumerate(packet["runtime_states"]):
    for batcher in runtime["batchers"].values():
        for name, sampler in batcher["samplers"].items():
            assert (
                sampler["epoch"] == 0 and sampler["rank"] == rank and sampler["world_size"] == 4
            )
            assert sampler["cursor"] == batcher["source_counts"][name]
            assert sampler["cursor"] <= len(range(rank, sampler["size"], 4))


def finite(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all().item()
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, list | tuple):
        for v in value:
            finite(v)


def exact(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, list | tuple):
        assert len(a) == len(b)
        for first, second in zip(a, b, strict=True):
            exact(first, second)
    else:
        assert a == b


finite(packet["model"])
finite(packet["optimizer"])
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
    if k in config
}
model = build_model(config["model"], attention_backend="math", **options)
model.load_state_dict(packet["model"])
optimizer = build_optimizer(model, config)
optimizer.load_state_dict(packet["optimizer"])
exact(packet["model"], model.state_dict())
exact(packet["optimizer"], optimizer.state_dict())
owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
assert len(owned) == len(set(owned)) == len(list(model.parameters()))
assert set(owned) == {id(p) for p in model.parameters()}
if config["optimizer"] == "muon":
    children = packet["optimizer"]["optimizers"]
    assert [c["name"] for c in children] == ["muon", "adamw"]
    assert {c["name"]: len(c["state_dict"]["state"]) for c in children} == {
        "muon": 96,
        "adamw": 9,
    }
else:
    assert len(packet["optimizer"]["state"]) == len(owned)
receipt = dict(
    status="passed",
    source_commit=commit,
    checkpoint_sha256=digest,
    optimizer_steps=config["max_steps"],
    sequences_seen=complete["sequences_seen"],
    parameter_count=packet["parameter_count"],
    world_size=4,
    finite_model_and_optimizer=True,
    model_and_optimizer_roundtrip_exact=True,
    optimizer_layout=packet["optimizer_layout"],
    flash3_verified=True,
    source_epoch_maxima=complete["source_epoch_maxima"],
    no_resampling_verified=True,
    data_manifest_sha256=contract["data_manifest_sha256"],
    training_seconds=complete["training_seconds"],
    peak_cuda_memory_bytes=complete["peak_cuda_memory_bytes"],
)
(root / "TRAINING_VERIFIED.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
