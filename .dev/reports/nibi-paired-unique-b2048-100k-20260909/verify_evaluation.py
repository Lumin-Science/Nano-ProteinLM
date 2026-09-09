"""Audit a downloaded production evaluation against its frozen protocol and shards."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

NAME = "nibi-paired-unique-b2048-100k-20260909"
REMOTE = Path("/scratch/muchenli/Nano-Protein-LM-nibi-paired-unique-b2048-100k-20260909")
REPO = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("recipe", choices=("baseline", "setting3"))
parser.add_argument("step", type=int)
parser.add_argument("--artifact-root", type=Path, default=REPO / ".exps" / NAME)
args = parser.parse_args()
SOURCE = (args.artifact_root / "SOURCE_COMMIT.txt").read_text().strip()
assert 10000 <= args.step <= 100000 and args.step % 10000 == 0
root = args.artifact_root / "full" / args.recipe
p = root / "evaluations" / f"step-{args.step:06d}"
remote = REMOTE / "full" / args.recipe / "evaluations" / p.name


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


contract = read(root / "run_contract.json")
assert contract["git_commit"] == SOURCE and not contract["git_dirty"]
assert contract["world_size"] == 4
assert contract["attention_kernel"]["revision"] == "e29f138fc363b396e5d2706c8a5f6fa7d36f41e0"
assert (
    contract["data_manifest_sha256"]
    == "c30da4a52028cb63bf4f764ff039009d965144db5a6fbf1069f4eebf97c7d3fa"
)
assert contract["config_sha256"] == sha(root / "config.yaml")
frozen_config = subprocess.check_output(
    ["git", "show", f"{SOURCE}:configs/esmc-171m-{args.recipe}-nibi-unique-b2048-100k.yaml"],
    cwd=REPO,
)
assert yaml.safe_load((root / "config.yaml").read_text()) == yaml.safe_load(frozen_config)
coverage = read(root / "DATA_COVERAGE.json")
ready = read(args.artifact_root / "DATA_READY.json")
assert coverage == contract["data_coverage"]
assert coverage["status"] == ready["status"] == "passed"
assert coverage["policy"] == "error" and coverage["runtime_exhaustion_guard"]
assert ready["verification"]["manifest_sha256"] == contract["data_manifest_sha256"]
assert ready["validation_identical_to_previous_runs"]
assert {k: v["train"]["records"] for k, v in ready["manifest"]["sources"].items()} == {
    "uniref90": 74175974,
    "mgnify": 22984503,
    "omg_img": 111296892,
}
receipt = read(p / "RESULT_VERIFIED.json")
training = read(p / "TRAINING_VERIFIED.json")
mlm = read(p / "eval-validation/VALIDATION_MLM.json")
evaluation = read(p / "eval-validation/EVALUATION.json")
contact = read(p / "eval-p-at-l/P_AT_L.json")
probe = read(p / "eval-p-at-l/CONTACT_PROBE.json")
unc = read(p / "eval-p-at-l/P_AT_L_UNCERTAINTY.json")
checkpoint = training["checkpoint_sha256"]
for item in (receipt, training):
    assert item["status"] == "passed"
    assert item["optimizer_steps"] == args.step
    assert item["sequences_seen"] == args.step * 2048
    assert item["full_training_complete"] == (args.step == 100000)
for item in (receipt, evaluation, contact, probe, unc):
    assert item["checkpoint_sha256"] == checkpoint
assert receipt["validation"] == evaluation["validation_mlm"] == mlm
assert mlm["sequences"] == 4096 and mlm["masked_residues"] == 139963
assert all(math.isfinite(mlm[k]) for k in ("sequence_mean_nll", "perplexity"))
historical = REPO / ".dev/reports/fir-171m-100k-20260906"
old_mlm = read(historical / "default/eval-validation/VALIDATION_MLM.json")
old_probe = read(historical / "default/eval-p-at-l/CONTACT_PROBE.json")
assert mlm["source_counts"] == old_mlm["source_counts"]
for key in (
    "dataset_manifest_sha256",
    "probe_train_chain_ids",
    "probe_validation_chain_ids",
    "maximum_pairs_per_class",
    "pair_sampling_seed",
    "channels",
    "protocol",
):
    assert probe[key] == old_probe[key], key
assert contact["selection_seed"] == 20260820
assert [v["C"] for v in probe["validation_trace"]] == [0.01, 0.1, 1.0, 10.0]
assert probe["validation_trace"] == contact["validation_trace"]
assert all(math.isfinite(v) for v in probe["coefficients"]) and math.isfinite(
    probe["intercept"]
)
rows, indices, digests = [], set(), {}
assert len(contact["components"]) == 16
for component in contact["components"]:
    rel = Path(component["path"]).relative_to(remote)
    f = p / rel
    assert sha(f) == component["sha256"]
    digests[str(rel)] = sha(f)
    shard = read(f)
    q = shard["contact"]
    assert shard["checkpoint_sha256"] == checkpoint
    assert q["probe_receipt_sha256"] == sha(p / "eval-p-at-l/CONTACT_PROBE.json")
    assert q["shard_count"] == 16 and q["selection_total_chains"] == 20775
    assert q["shard_index"] == component["shard_index"]
    assert q["selected_C"] == probe["selected_C"] == contact["selected_C"]
    assert q["selection_seed"] == 20260820
    assert q["evaluation_chains"] == component["chains"] == len(q["rows"])
    indices.add(q["shard_index"])
    rows.extend(q["rows"])
assert indices == set(range(16))
assert len(rows) == len({r["chain_id"] for r in rows}) == contact["evaluation_chains"] == 20775
with (historical / "contact-per-chain.tsv").open() as f:
    old_ids = {r["chain_id"] for r in csv.DictReader(f, delimiter="\t")}
assert {r["chain_id"] for r in rows} == old_ids
values = np.array([r["precision_at_l"] for r in rows], dtype=np.float64)
assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
assert abs(values.mean() - contact["p_at_l"]) < 1e-12
assert receipt["p_at_l"] == unc["p_at_l"] == contact["p_at_l"]
rng = np.random.default_rng(20260820)
means = np.empty(5000, dtype=np.float64)
for start in range(0, 5000, 128):
    stop = min(start + 128, 5000)
    samples = rng.integers(0, len(values), size=(stop - start, len(values)))
    means[start:stop] = values[samples].mean(axis=1)
ci = dict(
    replicates=5000,
    unit_count=len(values),
    confidence_interval_95=np.quantile(means, [0.025, 0.975]).tolist(),
)
assert ci == unc["uncertainty"] == receipt["uncertainty"]
metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
metrics = [r for r in metrics if r.get("event") == "train"]
assert any(r["optimizer_step"] == args.step for r in metrics)
assert metrics[-1]["optimizer_step"] >= args.step
for r in metrics:
    assert all(
        math.isfinite(r[k])
        for k in ("loss", "gradient_norm", "learning_rate", "step_compute_seconds")
    )
    assert r["attention_backend"] == "flash3"
    assert r["optimizer"] == ("adamw" if args.recipe == "baseline" else "muon")
    assert r["data_resampling"] == "error"
    assert all(v == 0 for v in r["source_epoch_maxima"].values())
    assert sum(r["source_counts_global"].values()) == r["sequences_seen"]
    if args.recipe == "setting3":
        assert math.isfinite(r["objective_loss"])
        assert r["training_loss_reduction"] == "sqrt_mask_count"
        b = r["batch_balance"]
        assert len(b["rank_tokens_before"]) == 4
        assert sum(b["rank_tokens_before"]) == sum(b["rank_tokens_after"])
    assert r["sequences_seen"] == r["optimizer_step"] * 2048
run = read(p / "EVALUATION_RUN.json") if (p / "EVALUATION_RUN.json").exists() else None
if args.step < 100000:
    assert run and run["status"] == "passed" and run["optimizer_step"] == args.step
    assert math.isfinite(run["wall_seconds"]) and run["wall_seconds"] > 0
table = io.StringIO()
writer = csv.writer(table, delimiter="\t", lineterminator="\n")
writer.writerow(["chain_id", "precision_at_l"])
writer.writerows((r["chain_id"], r["precision_at_l"]) for r in rows)
per_chain = p / "contact-per-chain.tsv.gz"
per_chain.write_bytes(gzip.compress(table.getvalue().encode(), mtime=0))
endpoint_metric = next(r for r in metrics if r["optimizer_step"] == args.step)
audit = dict(
    status="passed",
    recipe=args.recipe,
    world_size=4,
    data_manifest_sha256=contract["data_manifest_sha256"],
    no_sampler_wraps_in_verified_metrics=True,
    data_coverage_passed=True,
    training_seconds_at_checkpoint=endpoint_metric["training_seconds"],
    model_tokens_at_checkpoint=endpoint_metric["model_tokens"],
    source_counts_at_checkpoint=endpoint_metric["source_counts_global"],
    verified_utc=datetime.now(timezone.utc).isoformat(),
    source_commit=SOURCE,
    optimizer_steps=args.step,
    sequences_seen=args.step * 2048,
    checkpoint_sha256=checkpoint,
    validation=mlm,
    p_at_l=contact["p_at_l"],
    uncertainty=ci,
    complete_shards=16,
    component_sha256=digests,
    bootstrap_independently_recomputed=True,
    same_contact_chain_set_as_historical_comparison=True,
    same_probe_split_and_fit_protocol_as_historical_comparison=True,
    checkpoint_receipt_bindings_verified=True,
    latest_verified_training_step=metrics[-1]["optimizer_step"],
    training_continued_after_evaluation=metrics[-1]["optimizer_step"] > args.step,
    per_chain_table_sha256=sha(per_chain),
    evaluation_run=run,
)
(p / "LOCAL_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps({k: v for k, v in audit.items() if k != "component_sha256"}, indent=2))
