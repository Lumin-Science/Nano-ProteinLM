"""Pause DDP for external evaluation without restarting the training processes."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist


def run_periodic_evaluation(
    command: list[str],
    *,
    checkpoint: Path,
    output_root: Path,
    optimizer_step: int,
    device: torch.device,
    data_root: Path | None = None,
    poll_seconds: float = 1.0,
) -> float:
    """All ranks participate in short status broadcasts while rank 0 evaluates."""
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    started = time.perf_counter()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    destination = output_root / "evaluations" / f"step-{optimizer_step:06d}"
    status_path = output_root / "PERIODIC_EVALUATION.json"
    process = None
    log = None
    error = None
    receipt = {
        "status": "running",
        "optimizer_step": optimizer_step,
        "checkpoint": str(checkpoint.resolve()),
        "output_root": str(destination.resolve()),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    def write(path: Path, value: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)

    if rank == 0:
        try:
            destination.mkdir(parents=True, exist_ok=False)
            write(status_path, receipt)
            environment = os.environ.copy()
            if data_root is not None:
                environment["DATA_ROOT"] = str(data_root.resolve())
            for key in (
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "GROUP_RANK",
                "ROLE_RANK",
                "ROLE_WORLD_SIZE",
                "TORCHELASTIC_RESTART_COUNT",
            ):
                environment.pop(key, None)
            log = (destination / "evaluation.log").open("w")
            process = subprocess.Popen(
                [*command, str(checkpoint.resolve()), str(destination.resolve())],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except Exception as exception:
            error = str(exception)
    signal = torch.zeros((), dtype=torch.int32, device=device)
    while True:
        if rank == 0:
            code = process.poll() if process is not None else None
            if error is not None:
                signal.fill_(-1)
            elif code is not None:
                signal.fill_(1 if code == 0 else -1)
                if code != 0:
                    error = (
                        f"evaluator exited with code {code}; see {destination}/evaluation.log"
                    )
        if distributed:
            dist.broadcast(signal, src=0)
        if signal.item() != 0:
            break
        time.sleep(poll_seconds)
    elapsed = time.perf_counter() - started
    if rank == 0:
        if log is not None:
            log.close()
        receipt.update(
            status="passed" if signal.item() == 1 else "failed", wall_seconds=elapsed
        )
        if error is not None:
            receipt["error"] = error
        write(status_path, receipt)
        if destination.is_dir():
            write(destination / "EVALUATION_RUN.json", receipt)
    if signal.item() != 1:
        message = [error]
        if distributed:
            dist.broadcast_object_list(message, src=0)
        raise RuntimeError(f"periodic evaluation failed: {message[0]}")
    return elapsed
