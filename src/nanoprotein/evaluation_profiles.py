"""Search and final evaluation contracts shared by ordinary research and AutoResearch."""
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

from .data import file_sha256
from .evaluate import write_json
from .prepared_mlm import MASK_SEEDS, verify_cache

PROTOCOL = "nanoprotein-evaluation-v3"
PROBE_SEEDS = (20260819, 20260820, 20260821, 20260822, 20260823)
SEARCH_IDS_SHA256 = "9948c40ab11a18bcdf096b078c13e841a02eb1d2286af638565945ebf71dbb44"


def mean_sd(values: list[float], *, unit: str, variation: str, sequences: int) -> dict:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("metric requires finite attempts")
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else None,
            "std_ddof": 1, "attempts": len(values), "unit": unit,
            "variation": variation, "sequences": sequences}


def merge_mlm(reports: list[dict], cache_name: str, receipt: dict, checkpoint_sha256: str) -> dict:
    if sorted(r["worker"] for r in reports) != list(range(len(reports))):
        raise ValueError("MLM workers are missing or duplicated")
    results = []
    for report in reports:
        if report["checkpoint_sha256"] != checkpoint_sha256 or report["workers"] != len(reports):
            raise ValueError("MLM worker checkpoint/sharding changed")
        result = report["results"][cache_name]
        if result["cache_sha256"] != receipt["cache_sha256"]:
            raise ValueError("MLM worker used different masks")
        results.append(result)
    rows = sorted((row for result in results for row in result["rows"]), key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(receipt["sequences"])):
        raise ValueError("MLM workers do not cover each protein exactly once")
    if sum(result["masked_residues"] for result in results) != receipt["masked_residues"]:
        raise ValueError("MLM masked-target coverage changed")
    values = [row["nll"] for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite MLM loss")
    return {"sequence_mean_nll": sum(values) / len(values), "sequences": len(values),
            "mask_seed": receipt["mask_seed"], "cache_sha256": receipt["cache_sha256"],
            "masked_residues": receipt["masked_residues"],
            "zero_target_unit_ids": receipt["zero_target_unit_ids"],
            "zero_target_convention": receipt["zero_target_convention"]}


def run_profile(args) -> dict:
    from .contact_parallel import contact_devices

    started = time.monotonic()
    profile = args.profile
    search = profile == "search"
    count = 8192 if search else 26062
    seeds = PROBE_SEEDS[:1] if search else PROBE_SEEDS
    masks = MASK_SEEDS[:1] if search else MASK_SEEDS
    populations = ("search8192",) if search else ("contact26062", "original12288")
    prepared = json.loads((args.prepared_root / "PREPARED_INPUTS.json").read_text())
    ready = json.loads((args.contact_root / "PDB_CONTACT_DATASET_READY.json").read_text())
    manifest_sha = file_sha256(args.contact_root / "CONTACT_MANIFEST.jsonl")
    if ready.get("population_protocol") != "nanoprotein-contact-pool-v3" or prepared.get("protocol") != PROTOCOL:
        raise ValueError("run setup for evaluation v3; legacy populations are not profile inputs")
    if prepared["contact_manifest_sha256"] != manifest_sha or ready["manifest_sha256"] != manifest_sha:
        raise ValueError("prepared MLM/contact populations differ")
    overlap_audit = None
    if "training_overlap_audit_sha256" in ready:
        audit_path = args.contact_root / "TRAINING_OVERLAP_AUDIT.json"
        if file_sha256(audit_path) != ready["training_overlap_audit_sha256"]:
            raise ValueError("training-overlap audit changed")
        overlap_audit = {"path": str(audit_path.resolve()), "sha256": ready["training_overlap_audit_sha256"]}
    ids_path = args.contact_root / ("search_8192.ids" if search else "evaluation_26062.ids")
    ids = ids_path.read_text().splitlines()
    expected_ids_sha = SEARCH_IDS_SHA256 if search else ready["evaluation_ids_sha256"]
    if file_sha256(ids_path) != expected_ids_sha or len(ids) != count or len(set(ids)) != count:
        raise ValueError("fixed contact population changed")
    caches = {}
    for population in populations:
        for seed in masks:
            name = f"{population}-mask{seed}.npz"
            receipt = verify_cache(args.prepared_root / name, expected=prepared["caches"][name])
            expected_count = {"search8192": 8192, "contact26062": 26062, "original12288": 12288}[population]
            if receipt["sequences"] != expected_count or receipt["mask_seed"] != seed or receipt["population"] != population:
                raise ValueError("MLM profile contract changed")
            caches[name] = receipt
    if not search:
        for source, files in prepared["original_validation_binding"].items():
            for name, digest in files.items():
                if file_sha256(args.data_root / source / "validation" / name) != digest:
                    raise ValueError("original validation stores differ from prepared masks")
    if search and prepared.get("search_ids_sha256") != SEARCH_IDS_SHA256:
        raise ValueError("prepared search subset changed")
    import numpy as np
    contact_cache_name = f"{populations[0]}-mask{masks[0]}.npz"
    with np.load(args.prepared_root / contact_cache_name, allow_pickle=False) as cache:
        if cache["chain_ids"].tolist() != ids:
            raise ValueError("contact and MLM do not score the exact same ordered chains")
    devices = contact_devices(args.contact_gpus)
    checkpoint_sha = file_sha256(args.checkpoint)
    source_manifest = {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    request = {
        "protocol": PROTOCOL, "profile": profile, "checkpoint_sha256": checkpoint_sha,
        "contact_root": str(args.contact_root.resolve()), "contact_manifest_sha256": manifest_sha,
        "contact_ready_sha256": file_sha256(args.contact_root / "PDB_CONTACT_DATASET_READY.json"),
        "external_src": str(args.external_src.resolve()),
        "external_source_sha256": {str(p.relative_to(args.external_src)): file_sha256(p)
                                   for p in sorted(args.external_src.rglob("*.py"))},
        "runtime_source_sha256": source_manifest, "ids_sha256": expected_ids_sha,
        "prepared_inputs_sha256": file_sha256(args.prepared_root / "PREPARED_INPUTS.json"),
        "probe_seeds": list(seeds), "mask_seeds": list(masks), "devices": devices,
        "mlm_batch_size": args.validation_batch_size, "contact_workers": args.contact_workers,
        "contact_scoring_cache_root": str(args.contact_scoring_cache_root.resolve()) if args.contact_scoring_cache_root else None,
        "contact_scoring_cache_receipt_sha256": file_sha256(args.contact_scoring_cache_root / "CONTACT_SCORING_CACHE.json") if args.contact_scoring_cache_root else None,
    }
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "PROFILE_REQUEST.json"
    if request_path.exists():
        if not args.resume_components or json.loads(request_path.read_text()) != request:
            raise ValueError("use a fresh output directory or resume the identical profile request")
    elif any(root.iterdir()):
        raise ValueError("profile evaluation requires a fresh output directory")
    else:
        write_json(request_path, request)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    contacts, receipts = [], []
    for seed in seeds:
        destination = root / "attempts" / f"probe-{seed}"
        command = [sys.executable, "-m", "nanoprotein.evaluate", "--profile", "component",
                   "--checkpoint", str(args.checkpoint), "--data-root", str(args.data_root),
                   "--output-root", str(destination), "--external-src", str(args.external_src),
                   "--contact-root", str(args.contact_root), "--run-contact", "--skip-validation-mlm",
                   "--contact-chains", str(count), "--contact-chain-ids", str(ids_path),
                   "--probe-seed", str(seed), "--contact-bootstrap", "0",
                   "--contact-gpus", ",".join(devices)]
        if args.contact_workers:
            command += ["--contact-workers", str(args.contact_workers)]
        if args.contact_scoring_cache_root:
            command += ["--contact-scoring-cache-root", str(args.contact_scoring_cache_root)]
        if args.resume_components:
            command.append("--resume-components")
        print(json.dumps({"event": "profile_contact_attempt", "seed": seed, "chains": count}), flush=True)
        with (root / f"probe-{seed}.log").open("a") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        path = destination / "EVALUATION.json"
        report = json.loads(path.read_text())
        if report["checkpoint_sha256"] != checkpoint_sha or [row["chain_id"] for row in report["contact"]["rows"]] != ids:
            raise ValueError("contact attempt checkpoint/population changed")
        probe = json.loads((destination / "CONTACT_PROBE.json").read_text())
        if probe["pair_sampling_seed"] != seed:
            raise ValueError("probe attempt seed changed")
        contacts.append(report["contact"])
        receipts.append({"path": str(path.relative_to(root)), "sha256": file_sha256(path)})
    contact_seconds = time.monotonic() - started
    mlm_started = time.monotonic()
    processes = []
    worker_paths = [root / "attempts" / f"mlm-worker-{i}.json" for i in range(len(devices))]
    with ExitStack() as stack:
        try:
            for index, (device, path) in enumerate(zip(devices, worker_paths, strict=True)):
                if args.resume_components and path.exists():
                    continue
                command = [sys.executable, "-m", "nanoprotein.prepared_mlm", "--checkpoint", str(args.checkpoint),
                           "--output", str(path), "--worker", str(index), "--workers", str(len(devices)),
                           "--batch-size", str(args.validation_batch_size)]
                for name in caches:
                    command += ["--cache", str(args.prepared_root / name)]
                log = stack.enter_context((root / f"mlm-worker-{index}.log").open("w"))
                processes.append(subprocess.Popen(command, env={**env, "CUDA_VISIBLE_DEVICES": device}, stdout=log, stderr=subprocess.STDOUT))
            for process in processes:
                if process.wait() != 0:
                    raise RuntimeError("MLM worker failed; inspect mlm-worker logs")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    workers = [json.loads(path.read_text()) for path in worker_paths]
    mlm_attempts = {name: merge_mlm(workers, name, receipt, checkpoint_sha) for name, receipt in caches.items()}
    write_json(root / "attempts" / "MLM_ATTEMPTS.json", mlm_attempts)
    report = {"schema_version": 3, "protocol": PROTOCOL, "profile": profile,
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
              "checkpoint_training_seconds": workers[0]["training_seconds"],
              "request_sha256": file_sha256(request_path),
              "timing_seconds": {"contact_and_preflight": contact_seconds,
                                 "mlm": time.monotonic() - mlm_started, "total": time.monotonic() - started},
              "probe_seeds": list(seeds), "mask_seeds": list(masks),
              "contact_manifest_sha256": manifest_sha,
              "training_overlap_status": ready["training_overlap_status"],
              "training_overlap_audit": overlap_audit,
              "component_receipts": {"contact": receipts, "mlm": [
                  {"path": str(p.relative_to(root)), "sha256": file_sha256(p)} for p in worker_paths]}}
    if search:
        mlm = next(iter(mlm_attempts.values()))
        report.update(validation_mlm={**mlm, "protocol": PROTOCOL,
                                      "settings": {"population": "search8192", "mask_seed": masks[0],
                                                   "context_length": 512, "ids_sha256": SEARCH_IDS_SHA256}},
                      contact=contacts[0], selection_metric="validation_mlm.sequence_mean_nll",
                      selection_direction="minimize", contact_is_selection_gate=False)
    else:
        metrics = {"p_at_l": mean_sd([r["precision_at_l"] for r in contacts], unit="fraction",
                                     variation="probe_pair_sampling_and_logistic_fit_seed", sequences=count)}
        for population, population_count in (("contact26062", 26062), ("original12288", 12288)):
            metrics[f"mlm_{population}"] = mean_sd(
                [mlm_attempts[f"{population}-mask{seed}.npz"]["sequence_mean_nll"] for seed in masks],
                unit="nats", variation="fixed_masks_same_proteins_and_crops", sequences=population_count)
        report["metrics"] = metrics
    write_json(root / "EVALUATION.json", report)
    print(json.dumps({"event": "evaluation_complete", "profile": profile, "report": str(root / "EVALUATION.json")}), flush=True)
    return report
