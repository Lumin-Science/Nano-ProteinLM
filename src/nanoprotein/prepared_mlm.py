"""Fixed-mask, fixed-crop MLM inputs and one GPU worker for profile evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import SOURCES, TokenStore, file_sha256
from .evaluate import load_checkpoint, validation_example, write_json
from .tokenizer import ProteinTokenizer

MASK_SEEDS = (20260821, 20260822, 20260823, 20260824, 20260825)
PROTOCOL = "nanoprotein-fixed-crop-mlm-v3"


def prepare_cache(path: Path, examples: list, *, population: str, seed: int, binding: dict) -> dict:
    """Examples are (unique unit ID, residue tokens, original digest)."""
    tokenizer = ProteinTokenizer.esmc()
    count = len(examples)
    tokens = np.full((count, 512), tokenizer.pad_id, dtype=np.int16)
    labels = np.full((count, 512), -100, dtype=np.int16)
    lengths = np.empty(count, dtype=np.int16)
    manifest = hashlib.sha256()
    ids = []
    for index, (unit_id, residues, digest) in enumerate(examples):
        ids.append(unit_id)
        manifest.update(unit_id.encode("ascii") + b"\0" + digest)
        corrupted, targets = validation_example(
            residues, digest, residue_limit=510, tokenizer=tokenizer, mask_seed=seed
        )
        lengths[index] = len(corrupted)
        tokens[index, :len(corrupted)] = corrupted.numpy()
        labels[index, :len(targets)] = targets.numpy()
    if len(set(ids)) != count:
        raise ValueError("MLM cache requires unique units")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("xb") as handle:
        np.savez(handle, tokens=tokens, labels=labels, lengths=lengths,
                 order=np.argsort(lengths, kind="stable"), chain_ids=np.asarray(ids))
    receipt = {
        "protocol": PROTOCOL, "population": population, "sequences": count,
        "mask_seed": seed, "crop_seed": MASK_SEEDS[0], "context_length": 512,
        "mask_probability": 0.15, "aggregation": "mean_of_per_protein_masked_token_means",
        "masked_residues": int((labels != -100).sum()),
        "zero_target_unit_ids": [ids[i] for i in np.flatnonzero((labels != -100).sum(axis=1) == 0)],
        "zero_target_convention": "zero_loss_in_macro_mean_preserves_original_evaluator",
        "unit_manifest_sha256": manifest.hexdigest(), "binding": binding,
        "cache_sha256": file_sha256(path), "cache_bytes": path.stat().st_size,
    }
    write_json(path.with_suffix(".json"), receipt)
    return receipt


def original_examples(data_root: Path) -> tuple[list, dict]:
    examples, binding = [], {}
    for source in SOURCES:
        root = data_root / source / "validation"
        store = TokenStore.open(root)
        if store.index.size != 4096:
            raise ValueError(f"expected 4096 original validation proteins: {source}")
        binding[source] = {name: file_sha256(root / name) for name in ("index.npy", "tokens.bin")}
        for index in range(store.index.size):
            # Preserve the original evaluator's digest serialization for attempt zero.
            examples.append((f"{source}:{index}", store.sequence(index), bytes(store.index[index]["digest"])))
    return examples, binding


def verify_cache(path: Path, *, expected: dict | None = None) -> dict:
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt.get("protocol") != PROTOCOL or file_sha256(path) != receipt["cache_sha256"]:
        raise ValueError(f"MLM cache changed: {path}")
    if expected is not None and receipt != expected:
        raise ValueError(f"MLM receipt differs from prepared-input contract: {path}")
    return receipt


def score_cache(model, path: Path, *, device, batch_size: int, worker: int, workers: int) -> dict:
    receipt = verify_cache(path)
    with np.load(path, allow_pickle=False) as cache:
        tokens, labels = cache["tokens"], cache["labels"]
        lengths, order = cache["lengths"], cache["order"]
    count = receipt["sequences"]
    if tokens.shape != (count, 512) or labels.shape != tokens.shape:
        raise ValueError("invalid MLM cache shape")
    if sorted(order.tolist()) != list(range(count)) or int((labels != -100).sum()) != receipt["masked_residues"]:
        raise ValueError("invalid MLM cache coverage")
    rows_out, masked = [], 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, count, batch_size)):
            if batch_index % workers != worker:
                continue
            rows = order[start:start + batch_size]
            width = int(lengths[rows].max())
            corrupted = torch.from_numpy(tokens[rows, :width].astype(np.int64)).to(device)
            targets = torch.from_numpy(labels[rows, :width].astype(np.int64)).to(device)
            attention = torch.from_numpy(np.arange(width)[None, :] < lengths[rows, None]).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(corrupted, attention)["logits"]
                per_token = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                            ignore_index=-100, reduction="none").view_as(targets)
            selected = targets != -100
            losses = (per_token.sum(1) / selected.sum(1).clamp_min(1)).float().cpu().numpy()
            masked += int(selected.sum())
            rows_out.extend({"index": int(i), "nll": float(loss)} for i, loss in zip(rows, losses, strict=True))
    return {"cache_sha256": receipt["cache_sha256"], "masked_residues": masked,
            "rows": rows_out, "seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers or args.batch_size <= 0:
        parser.error("invalid worker or batch size")
    device = torch.device("cuda", 0)
    torch.manual_seed(20260821)
    model, packet = load_checkpoint(args.checkpoint, device)
    results = {path.name: score_cache(model, path, device=device, batch_size=args.batch_size,
                                    worker=args.worker, workers=args.workers) for path in args.cache}
    write_json(args.output, {"worker": args.worker, "workers": args.workers,
                            "checkpoint_sha256": file_sha256(args.checkpoint),
                            "training_seconds": packet["training_seconds"], "results": results})


if __name__ == "__main__":
    main()
