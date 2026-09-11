"""Prepare Stage 2 from the pinned release, preserving the Stage 1 corpus."""

import concurrent.futures
import copy
import datetime
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from nanoprotein.data import file_sha256
from nanoprotein.sharded_data import (
    materialize_plan,
    validate_prepared_plan,
    validate_release_manifest,
)

ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-setting3-stage2-b2048-300k-20260911")
OLD = Path("/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-400k-20260910")


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def phase(name, **kw):
    value = dict(phase=name, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kw)
    save("PREPARATION.json", value)
    print(json.dumps(value), flush=True)


def main():
    old_local = Path((OLD / "PRODUCTION_DATA_ROOT.txt").read_text().strip())
    local = Path(os.environ["SLURM_TMPDIR"]) / "nano-nibi-setting3-stage2-20260911"
    assert not local.exists() and not (ROOT / "data").exists()
    oldplan = json.loads((old_local / "download-plan.json").read_text())
    release = json.loads((OLD / "cache/manifest.json").read_text())
    validate_release_manifest(release)
    assert file_sha256(OLD / "cache/manifest.json") == oldplan["release_manifest_sha256"]
    plan = copy.deepcopy(oldplan)
    target = oldplan["sources"]["mgnify"]["selected_unique_records"] + 40_000_000
    selected, records = [], 0
    for shard in release["sources"]["mgnify"]["train"]:
        selected.append(shard)
        records += shard["records"]
        if records >= target:
            break
    assert records >= target
    plan["sources"]["mgnify"].update(train=selected, selected_unique_records=records)
    weights = dict(uniref90=0.63, mgnify=0.06, omg_img=0.31)
    shards = []
    for name, source in plan["sources"].items():
        source["expected_draws"] = 300000 * 2048 * weights[name]
        shards.extend(source["train"] + source["validation"])
    plan.update(
        paths=[s["path"] for s in shards],
        planned_total_steps=700000,
        planned_additional_steps=300000,
        planned_global_batch=2048,
        normalized_weights=weights,
        total_training_shards=sum(len(s["train"]) for s in plan["sources"].values()),
    )
    for key, field in [
        ("selected_records_including_validation", "records"),
        ("selected_residues_including_validation", "residues"),
        ("selected_compressed_bytes_including_validation", "bytes"),
    ]:
        plan[key] = sum(s[field] for s in shards)
    old_paths = set(oldplan["paths"])
    added = [s for s in shards if s["path"] not in old_paths]
    save(
        "DATA_PLAN.json",
        dict(
            release_revision=oldplan["revision"],
            release_manifest_sha256=oldplan["release_manifest_sha256"],
            additional_steps=300000,
            global_batch=2048,
            mixture=weights,
            mgnify_records=records,
            additional_mgnify_records=records
            - oldplan["sources"]["mgnify"]["selected_unique_records"],
            additional_shards=len(added),
            download_bytes=sum(s["bytes"] for s in added),
            plan=plan,
        ),
    )
    save("download-plan.json", plan)
    cache = ROOT / "cache"
    cache.mkdir()
    shutil.copy2(OLD / "cache/manifest.json", cache / "manifest.json")
    phase(
        "downloading",
        added_shards=len(added),
        additional_mgnify_records=records
        - oldplan["sources"]["mgnify"]["selected_unique_records"],
    )

    def fetch(shard):
        dest, parent = cache / shard["path"], OLD / "cache" / shard["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if parent.exists():
            os.link(parent, dest)
        else:
            hf_hub_download(
                repo_id=oldplan["repo_id"],
                repo_type="dataset",
                revision=oldplan["revision"],
                filename=shard["path"],
                local_dir=cache,
            )
        assert file_sha256(dest) == shard["sha256"], shard["path"]
        return shard["path"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, name in enumerate(pool.map(fetch, shards), 1):
            if i % 25 == 0:
                print(
                    json.dumps(dict(verified_downloads=i, total=len(shards), last=name)),
                    flush=True,
                )
    save("DOWNLOAD_VERIFIED.json", dict(status="passed", shards=len(shards), added=len(added)))
    phase("materializing", data_root=str(local))
    proof = materialize_plan(plan, release, cache, local, workers=1, reuse_root=old_local)
    validate_prepared_plan(plan, local)
    manifest = json.loads((local / "manifest.json").read_text())
    previous = json.loads((old_local / "manifest.json").read_text())
    for name in manifest["sources"]:
        assert (
            manifest["sources"][name]["validation"] == previous["sources"][name]["validation"]
        )
    phase("preserving", data_root=str(local))
    durable = ROOT / "data"
    durable.mkdir()
    for name in manifest["sources"]:
        if manifest["sources"][name] == previous["sources"][name]:
            shutil.copytree(OLD / "data" / name, durable / name, copy_function=os.link)
        else:
            shutil.copytree(local / name, durable / name)
    for name in ("manifest.json", "download-plan.json", "CORPUS_VERIFICATION.json"):
        shutil.copy2(local / name, durable / name)
    validate_prepared_plan(plan, durable)
    (ROOT / "PRODUCTION_DATA_ROOT.txt").write_text(str(local) + "\n")
    save(
        "DATA_READY.json",
        dict(
            status="passed",
            node_local_root=str(local),
            scratch_root=str(durable),
            manifest=manifest,
            verification=proof,
            validation_unchanged=True,
        ),
    )
    phase("complete")


if __name__ == "__main__":
    main()
