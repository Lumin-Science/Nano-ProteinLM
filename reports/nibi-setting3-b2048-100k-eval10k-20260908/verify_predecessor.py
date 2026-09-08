"""Check the completed baseline; --deep hashes checkpoints/shards on the compute node."""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import socket
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--deep", action="store_true")
args = parser.parse_args()
root = Path("/scratch/muchenli/Nano-Protein-LM-nibi-b2048-100k-eval10k-20260908/full")
durable = Path(
    "/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-baseline-b2048-100k-eval10k-20260908"
)
assert (root / "COMPLETE.txt").exists() and not (root / "FAILED.txt").exists()
assert (root / "CHECKPOINT_PRESERVED.txt").exists()


def read(path):
    return json.loads(path.read_text())


complete = read(root / "TRAINING_COMPLETE.json")
verified = read(root / "TRAINING_VERIFIED.json")
contract = read(root / "run_contract.json")
assert contract["git_commit"] == "caa95a15b55ff2ca2395687f71e1c4b3a294b3d4"
assert not contract["git_dirty"] and contract["world_size"] == 8
assert verified["status"] == "passed" and complete["stop_reason"] == "max_steps"
assert complete["optimizer_steps"] == verified["optimizer_steps"] == 100000
assert complete["sequences_seen"] == verified["sequences_seen"] == 204800000
checkpoint_sha = complete["final_checkpoint"]["sha256"]
assert verified["checkpoint_sha256"] == checkpoint_sha
assert read(durable / "TRAINING_VERIFIED.json")["checkpoint_sha256"] == checkpoint_sha
assert (durable / "CHECKPOINT_SHA256.txt").read_text().split()[0] == checkpoint_sha
assert (durable / "checkpoint-final.pt").is_file()
if args.deep:
    assert socket.gethostname().split(".")[0] == "g27"
    assert os.environ["SLURM_JOB_ID"] == "12162637"
    with (durable / "checkpoint-final.pt").open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == checkpoint_sha
evaluations = []
for step in range(10000, 100001, 10000):
    p = root / "evaluations" / f"step-{step:06d}"
    result = read(p / "RESULT_VERIFIED.json")
    training = read(p / "TRAINING_VERIFIED.json")
    contact = read(p / "eval-p-at-l/P_AT_L.json")
    uncertainty = read(p / "eval-p-at-l/P_AT_L_UNCERTAINTY.json")
    evaluation = read(p / "eval-validation/EVALUATION.json")
    probe = read(p / "eval-p-at-l/CONTACT_PROBE.json")
    assert result["status"] == training["status"] == "passed"
    assert result["optimizer_steps"] == training["optimizer_steps"] == step
    assert result["sequences_seen"] == step * 2048
    digest = result["checkpoint_sha256"]
    for r in (training, contact, uncertainty, evaluation, probe):
        assert r["checkpoint_sha256"] == digest
    assert result["validation"]["sequences"] == 4096
    assert result["validation"]["masked_residues"] == 139963
    assert math.isfinite(result["validation"]["sequence_mean_nll"])
    assert result["uncertainty"]["unit_count"] == contact["evaluation_chains"] == 20775
    assert result["uncertainty"]["replicates"] == 5000
    assert result["uncertainty"] == uncertainty["uncertainty"]
    assert result["p_at_l"] == uncertainty["p_at_l"] == contact["p_at_l"]
    assert len(contact["components"]) == 16
    assert {c["shard_index"] for c in contact["components"]} == set(range(16))
    assert sum(c["chains"] for c in contact["components"]) == 20775
    if step == 100000:
        assert digest == checkpoint_sha
    if args.deep:
        rows = []
        probe_sha = hashlib.sha256(
            (p / "eval-p-at-l/CONTACT_PROBE.json").read_bytes()
        ).hexdigest()
        for component in contact["components"]:
            file = Path(component["path"])
            file.relative_to(p)
            assert hashlib.sha256(file.read_bytes()).hexdigest() == component["sha256"]
            shard = read(file)
            assert shard["checkpoint_sha256"] == digest
            assert shard["contact"]["probe_receipt_sha256"] == probe_sha
            rows.extend(shard["contact"]["rows"])
        assert len(rows) == len({r["chain_id"] for r in rows}) == 20775
        assert all(
            math.isfinite(r["precision_at_l"]) and 0 <= r["precision_at_l"] <= 1 for r in rows
        )
        assert (
            abs(math.fsum(r["precision_at_l"] for r in rows) / 20775 - result["p_at_l"]) < 1e-12
        )
    evaluations.append(dict(optimizer_step=step, checkpoint_sha256=digest))
print(
    json.dumps(
        dict(
            status="passed",
            observed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            deep_hash_check=args.deep,
            checkpoint_sha256=checkpoint_sha,
            durable_checkpoint=str(durable / "checkpoint-final.pt"),
            evaluations=evaluations,
        ),
        indent=2,
    )
)
