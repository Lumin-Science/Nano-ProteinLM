"""Time-budgeted four-GPU ESMC pretraining loop."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import MixtureBatcher, file_sha256
from .model import ESMCForMaskedLM, build_model, count_parameters, parameter_groups
from .schedule import Stage, stage_for_time, wsd_multiplier
from .tokenizer import ProteinTokenizer, mask_tokens


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("training config must be a mapping")
    return config


def sequence_mean_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none"
    ).view_as(labels)
    selected = labels != -100
    return (per_token.sum(dim=1) / selected.sum(dim=1).clamp_min(1)).mean()


def _distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=5))
    return rank, local_rank, world_size


def _stage(spec: dict[str, Any]) -> Stage:
    return Stage(
        name=str(spec["name"]),
        context_length=int(spec["context_length"]),
        micro_batch_size=int(spec["micro_batch_size"]),
        gradient_accumulation=int(spec["gradient_accumulation"]),
        mixture={name: float(value) for name, value in spec["mixture"].items()},
    )


def _unwrap(model: torch.nn.Module) -> ESMCForMaskedLM:
    current = model.module if isinstance(model, DDP) else model
    original = getattr(current, "_orig_mod", current)
    if not isinstance(original, ESMCForMaskedLM):
        raise TypeError(f"unexpected wrapped model type {type(original)}")
    return original


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    optimizer_step: int,
    training_seconds: float,
    stage: str,
    parameter_count: int,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    raw_model = _unwrap(model)
    torch.save(
        {
            "schema_version": 1,
            "model_name": raw_model.config.name,
            "model_config": raw_model.config.to_dict(),
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_step": optimizer_step,
            "training_seconds": training_seconds,
            "stage": stage,
            "parameter_count": parameter_count,
            "train_config": config,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        temporary,
    )
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_state(root: Path) -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": revision, "git_dirty": dirty}


def train(
    config_path: Path,
    *,
    data_root: Path,
    output_root: Path,
    walltime_override: int | None = None,
) -> None:
    config = load_config(config_path)
    rank, local_rank, world_size = _distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("speedrun training requires CUDA")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    seed = int(config.get("seed", 20260821))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    output_root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        project_root = Path(__file__).resolve().parents[1]
        uv_lock = project_root / "uv.lock"
        _write_json(
            output_root / "run_contract.json",
            {
                "config_path": str(config_path.resolve()),
                "config_sha256": file_sha256(config_path),
                "data_manifest": str((data_root / "manifest.json").resolve()),
                "data_manifest_sha256": file_sha256(data_root / "manifest.json"),
                "world_size": world_size,
                "python": platform.python_version(),
                "cuda": torch.version.cuda,
                "torch": torch.__version__,
                "visible_gpu": torch.cuda.get_device_name(device),
                "attention_implementation": (
                    "whole-transformer-packed-aten::_flash_attention_forward"
                ),
                "uv_lock": str(uv_lock),
                "uv_lock_sha256": file_sha256(uv_lock),
                **_git_state(project_root),
            },
        )
    if world_size > 1:
        dist.barrier()

    model_options = {
        "attention_backend": str(config.get("attention_backend", "flash")),
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", False)),
    }
    model = build_model(str(config["model"]), **model_options).to(device)
    parameter_count = count_parameters(model)
    expected = int(config["expected_parameter_count"])
    if parameter_count != expected:
        raise RuntimeError(
            f"parameter count drift: expected={expected}, observed={parameter_count}"
        )
    if bool(config.get("compile", False)):
        model = torch.compile(
            model,
            mode=str(config.get("compile_mode", "default")),
            dynamic=bool(config.get("compile_dynamic", False)),
        )
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            forward_sync_buffers=False,
            gradient_as_bucket_view=True,
        )

    optimizer = torch.optim.AdamW(
        parameter_groups(model, weight_decay=float(config["weight_decay"])),
        lr=float(config["learning_rate"]),
        betas=tuple(config.get("betas", (0.9, 0.95))),
        eps=1e-8,
        fused=True,
    )
    tokenizer = ProteinTokenizer.esmc()
    stages = (_stage(config["stages"][0]), _stage(config["stages"][1]))
    batchers = {
        stage.name: MixtureBatcher(data_root, "train", stage.mixture, seed=seed, rank=rank)
        for stage in stages
    }
    walltime_seconds = float(
        walltime_override if walltime_override is not None else config["walltime_seconds"]
    )
    stage1_fraction = float(config.get("stage1_fraction", 2.0 / 3.0))
    warmup_steps = int(config.get("warmup_steps", 10))
    log_interval = int(config.get("log_interval", 5))
    clip_norm = float(config.get("gradient_clip_norm", 1.0))
    peak_learning_rate = float(config["learning_rate"])
    metrics_path = output_root / "metrics.jsonl"
    optimizer_step = 0
    training_seconds = 0.0
    model_tokens = 0
    filled_residues = 0
    sequences_seen = 0
    current_stage_name: str | None = None
    stage_checkpoint: dict[str, object] | None = None

    while training_seconds < walltime_seconds:
        stage, stage_progress = stage_for_time(
            training_seconds,
            walltime_seconds=walltime_seconds,
            stage1_fraction=stage1_fraction,
            stages=stages,
        )
        if current_stage_name is not None and stage.name != current_stage_name:
            if rank == 0:
                stage_checkpoint = save_checkpoint(
                    output_root / "checkpoint-stage1.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    optimizer_step=optimizer_step,
                    training_seconds=training_seconds,
                    stage=current_stage_name,
                    parameter_count=parameter_count,
                )
                print(json.dumps({"event": "stage_checkpoint", **stage_checkpoint}), flush=True)
            if world_size > 1:
                dist.barrier()
        current_stage_name = stage.name
        optimizer_step += 1
        multiplier = wsd_multiplier(
            optimizer_step=optimizer_step,
            warmup_steps=warmup_steps,
            stage_name=stage.name,
            stage_progress=stage_progress,
            minimum_ratio=float(config.get("minimum_lr_ratio", 0.1)),
        )
        for group in optimizer.param_groups:
            group["lr"] = peak_learning_rate * multiplier

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0
        step_filled = 0
        step_sequences = 0
        torch.cuda.synchronize(device)
        compute_started = time.perf_counter()
        for micro_step in range(stage.gradient_accumulation):
            input_ids, attention_mask = batchers[stage.name].batch(
                stage.micro_batch_size,
                context_length=stage.context_length,
                tokenizer=tokenizer,
            )
            valid_tokens = int(attention_mask.sum().item())
            step_tokens += valid_tokens
            step_filled += valid_tokens - 2 * stage.micro_batch_size
            step_sequences += stage.micro_batch_size
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            corrupted, labels = mask_tokens(input_ids, attention_mask, tokenizer)
            synchronize = micro_step == stage.gradient_accumulation - 1
            sync_context = nullcontext()
            if isinstance(model, DDP) and not synchronize:
                sync_context = model.no_sync()
            with sync_context, torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids=corrupted, attention_mask=attention_mask)
                if not isinstance(output, dict):
                    raise TypeError("model output contract changed")
                loss = sequence_mean_loss(output["logits"], labels)
                (loss / stage.gradient_accumulation).backward()
            step_loss += float(loss.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        torch.cuda.synchronize(device)
        compute_seconds = time.perf_counter() - compute_started
        if world_size > 1:
            step_seconds = torch.tensor(compute_seconds, dtype=torch.float64, device=device)
            dist.all_reduce(step_seconds, op=dist.ReduceOp.MAX)
            compute_seconds = float(step_seconds.item())
        training_seconds += compute_seconds
        step_counts = torch.tensor(
            [step_tokens, step_filled, step_sequences],
            dtype=torch.int64,
            device=device,
        )
        if world_size > 1:
            dist.all_reduce(step_counts, op=dist.ReduceOp.SUM)
        global_step_tokens, global_step_filled, global_step_sequences = (
            int(value) for value in step_counts.tolist()
        )
        model_tokens += global_step_tokens
        filled_residues += global_step_filled
        sequences_seen += global_step_sequences

        if rank == 0 and (optimizer_step == 1 or optimizer_step % log_interval == 0):
            tokens_per_second = global_step_tokens / compute_seconds
            peak_bf16_tflops = float(config.get("peak_bf16_tflops_per_gpu", 312.0))
            mfu_6n = (
                6 * parameter_count * tokens_per_second / (peak_bf16_tflops * 1e12 * world_size)
            )
            record = {
                "event": "train",
                "optimizer_step": optimizer_step,
                "stage": stage.name,
                "stage_progress": stage_progress,
                "loss": step_loss / stage.gradient_accumulation,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": float(gradient_norm),
                "training_seconds": training_seconds,
                "step_compute_seconds": compute_seconds,
                "step_model_tokens": global_step_tokens,
                "model_tokens": model_tokens,
                "filled_residues": filled_residues,
                "sequences_seen": sequences_seen,
                "tokens_per_second": tokens_per_second,
                "mfu_6n": mfu_6n,
                "mfu_denominator": (
                    f"6 * parameters * model_tokens / ({peak_bf16_tflops}e12 "
                    f"BF16 FLOP/s * {world_size} GPUs)"
                ),
                "estimated_training_flops": 6 * parameter_count * model_tokens,
                "source_counts_rank0": dict(batchers[stage.name].source_counts),
                "attention_backend": model_options["attention_backend"],
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)

    if world_size > 1:
        dist.barrier()
    final_checkpoint: dict[str, object] | None = None
    if rank == 0:
        final_checkpoint = save_checkpoint(
            output_root / "checkpoint-final.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            optimizer_step=optimizer_step,
            training_seconds=training_seconds,
            stage=current_stage_name or "stage1",
            parameter_count=parameter_count,
        )
        completion = {
            "event": "training_complete",
            "optimizer_steps": optimizer_step,
            "training_seconds": training_seconds,
            "walltime_budget_seconds": walltime_seconds,
            "model_tokens": model_tokens,
            "filled_residues": filled_residues,
            "sequences_seen": sequences_seen,
            "parameter_count": parameter_count,
            "stage_checkpoint": stage_checkpoint,
            "final_checkpoint": final_checkpoint,
        }
        _write_json(output_root / "TRAINING_COMPLETE.json", completion)
        print(json.dumps(completion, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--walltime-seconds", type=int)
    args = parser.parse_args()
    train(
        args.config,
        data_root=args.data_root,
        output_root=args.output_root,
        walltime_override=args.walltime_seconds,
    )


if __name__ == "__main__":
    main()
