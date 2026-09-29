"""Parallel contact execution for the standard checkpoint evaluation command."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone

import torch

from .contact_cache import verify_contact_scoring_cache
from .data import file_sha256
from .evaluate import merge_contact_evaluation, write_json


def contact_devices(value: str | None) -> list[str]:
    visible = value if value is not None else os.environ.get("EVAL_GPUS")
    if visible is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        devices = [str(index) for index in range(torch.cuda.device_count())]
    else:
        devices = [item.strip() for item in visible.split(",")]
    if not devices or any(not item or item == "-1" for item in devices):
        raise ValueError("parallel contact evaluation requires visible GPU identifiers")
    if len(set(devices)) != len(devices):
        raise ValueError("contact GPU identifiers must be distinct")
    return devices


def run_contact_parallel(args: argparse.Namespace, checkpoint_sha256: str) -> dict:
    """Fit once, score deterministic shards, and restore the ordinary contact receipt."""
    devices = contact_devices(args.contact_gpus)
    workers = min(args.contact_workers or 8 * len(devices), args.contact_chains)
    root = args.output_root
    request = {
        key: str(value) if hasattr(value, "__fspath__") else value
        for key, value in vars(args).items()
        if key != "resume_components"
    }
    request.update(checkpoint_sha256=checkpoint_sha256, devices=devices, workers=workers)
    request_path = root / "EVALUATION_REQUEST.json"
    if request_path.exists():
        if not args.resume_components or json.loads(request_path.read_text()) != request:
            raise ValueError("use a fresh evaluation output or resume the identical request")
    elif any(root.iterdir()):
        raise ValueError("parallel evaluation requires a fresh output directory")
    else:
        write_json(request_path, request)

    started = time.monotonic()
    execution = root / "EXECUTION.txt"
    execution.write_text(
        f"started_at_utc={datetime.now(timezone.utc).isoformat()}\n"
        f"checkpoint={args.checkpoint}\nvisible_gpus={','.join(devices)}\n"
        f"contact_chains={args.contact_chains}\ncontact_shards={workers}\n"
    )
    print(
        json.dumps({"event": "contact_parallel_started", "gpus": devices, "workers": workers}),
        flush=True,
    )
    common = [
        "--checkpoint",
        str(args.checkpoint),
        "--external-src",
        str(args.external_src),
        "--contact-root",
        str(args.contact_root),
    ]
    environment = os.environ.copy()
    # Avoid CPU thread oversubscription while GPU workers score their chains.
    environment.setdefault("OMP_NUM_THREADS", "1")
    cache_args = []
    if args.contact_scoring_cache_root is not None:
        preflight = args.contact_scoring_cache_preflight
        if preflight is None:
            preflight = root / "CONTACT_SCORING_CACHE_PREFLIGHT.json"
            verify_contact_scoring_cache(
                cache_root=args.contact_scoring_cache_root, output=preflight
            )
        cache_args = [
            "--contact-scoring-cache-root",
            str(args.contact_scoring_cache_root),
            "--contact-scoring-cache-preflight",
            str(preflight),
        ]
    elif args.contact_scoring_cache_preflight is not None:
        raise ValueError("contact scoring cache preflight requires a cache root")

    probe = root / "CONTACT_PROBE.json"
    if not (args.resume_components and probe.exists()):
        with (
            (root / "probe.stdout").open("w") as stdout,
            (root / "probe.stderr").open("w") as stderr,
        ):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nanoprotein.fit_contact_probe",
                    *common,
                    "--output",
                    str(probe),
                ],
                env={**environment, "CUDA_VISIBLE_DEVICES": devices[0]},
                stdout=stdout,
                stderr=stderr,
                check=True,
            )
    probe_digest = file_sha256(probe)
    paths = []
    processes = []
    with ExitStack() as stack:
        try:
            for shard in range(workers):
                destination = root / "components" / f"contact-shard-{shard}"
                destination.mkdir(parents=True, exist_ok=True)
                path = destination / "EVALUATION.json"
                paths.append(path)
                if args.resume_components and path.exists():
                    continue
                command = [
                    sys.executable,
                    "-m",
                    "nanoprotein.evaluate",
                    *common,
                    "--data-root",
                    str(args.data_root),
                    "--output-root",
                    str(destination),
                    "--run-contact",
                    "--skip-validation-mlm",
                    "--contact-mode",
                    "serial",
                    "--contact-chains",
                    str(args.contact_chains),
                    "--contact-bootstrap",
                    "0",
                    "--contact-shard-index",
                    str(shard),
                    "--contact-shard-count",
                    str(workers),
                    "--contact-probe-receipt",
                    str(probe),
                    *cache_args,
                ]
                stdout = stack.enter_context((destination / "evaluate.stdout").open("w"))
                stderr = stack.enter_context((destination / "evaluate.stderr").open("w"))
                process = subprocess.Popen(
                    command,
                    env={**environment, "CUDA_VISIBLE_DEVICES": devices[shard % len(devices)]},
                    stdout=stdout,
                    stderr=stderr,
                )
                processes.append((process, destination))
            for process, destination in processes:
                if process.wait() != 0:
                    raise RuntimeError(
                        f"contact shard failed; inspect {destination}/evaluate.stderr"
                    )
        finally:
            for process, _destination in processes:
                if process.poll() is None:
                    process.terminate()
            for process, _destination in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    merged = merge_contact_evaluation(
        contact_paths=paths,
        expected_contact_chains=args.contact_chains,
        contact_bootstrap=args.contact_bootstrap,
    )
    contact = merged.pop("contact")
    if (
        merged["checkpoint_sha256"] != checkpoint_sha256
        or contact["probe_receipt_sha256"] != probe_digest
    ):
        raise ValueError(
            "parallel contact receipts do not match the checkpoint and fitted probe"
        )
    write_json(root / "P_AT_L.json", merged)
    write_json(root / "CONTACT.json", contact)
    reports = [json.loads(path.read_text()) for path in paths]
    with execution.open("a") as handle:
        handle.write(
            f"finished_at_utc={datetime.now(timezone.utc).isoformat()}\n"
            f"evaluation_wall_seconds={time.monotonic() - started:.3f}\n"
        )
    return {
        "contact": contact,
        "checkpoint_training_seconds": reports[0]["checkpoint_training_seconds"],
        "peak_cuda_memory_bytes": max(
            report.get("peak_cuda_memory_bytes", 0) for report in reports
        ),
        "contact_execution": {"mode": "parallel", "gpus": devices, "workers": workers},
        "component_receipts": {"contact_shards": merged["components"]},
    }
