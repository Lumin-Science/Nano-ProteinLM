"""Checkpoint evaluation: held-out MLM, current P-CORE, and paper-style P@L."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import MixtureBatcher, file_sha256
from .model import ESMCConfig, ESMCForMaskedLM
from .tokenizer import ProteinTokenizer, mask_tokens

PCORE_DIAGNOSTIC_TASKS = (
    "remote_homology",
    "human_ppi",
    "flip2_hydro_low_to_high",
)


def write_json(path: Path, payload: object) -> str:
    """Atomically persist a component receipt and return its digest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return file_sha256(path)


def embed_sequences_packed(
    model: Any,
    tokenizer: Any,
    sequences: Sequence[str],
    store: Any,
    *,
    device: torch.device,
    batch_residue_budget: int,
    include_residue: bool,
    deterministic_windows: Callable[[str, int], list[str]],
    window_embeddings: Callable[..., tuple[np.ndarray, np.ndarray, list[np.ndarray]]],
) -> dict[str, int]:
    """Embed windows from different proteins in the same GPU batch.

    The frozen evaluator batches windows only within each protein, making almost
    every short protein a batch of one. This keeps its windowing, pooling, cache,
    and residue semantics while filling a residue-budget batch across proteins.
    """

    if batch_residue_budget <= 0:
        raise ValueError("batch_residue_budget must be positive")
    missing: list[tuple[int, str]] = []
    resumed = 0
    for sequence_index, sequence in enumerate(sequences):
        protein_present = store.load(sequence) is not None
        residue_present = not include_residue or store.load_residue(sequence) is not None
        if not protein_present or (include_residue and not residue_present):
            missing.append((sequence_index, sequence))
        else:
            resumed += 1

    records: list[tuple[int, str, str]] = []
    remaining: dict[int, int] = {}
    for sequence_index, sequence in missing:
        windows = deterministic_windows(sequence, store.contract.maximum_residues)
        remaining[sequence_index] = len(windows)
        records.extend((sequence_index, sequence, window) for window in windows)

    weighted_sums: dict[int, np.ndarray] = {}
    residue_counts: dict[int, int] = {}
    residue_chunks: dict[int, list[np.ndarray]] = {}
    written = 0
    residue_written = 0
    cursor = 0
    while cursor < len(records):
        batch: list[tuple[int, str, str]] = []
        batch_residues = 0
        while cursor < len(records):
            record = records[cursor]
            cost = len(record[2]) + 2
            if batch and batch_residues + cost > batch_residue_budget:
                break
            batch.append(record)
            batch_residues += cost
            cursor += 1
        embeddings, counts, batch_residue_embeddings = window_embeddings(
            model,
            tokenizer,
            [window for _, _, window in batch],
            device=device,
        )
        for record, embedding, count, residue_embedding in zip(
            batch, embeddings, counts, batch_residue_embeddings, strict=True
        ):
            sequence_index, sequence, _window = record
            if sequence_index not in weighted_sums:
                weighted_sums[sequence_index] = np.zeros(
                    store.contract.hidden_size, dtype=np.float64
                )
                residue_counts[sequence_index] = 0
                if include_residue:
                    residue_chunks[sequence_index] = []
            weighted_sums[sequence_index] += embedding.astype(np.float64) * int(count)
            residue_counts[sequence_index] += int(count)
            if include_residue:
                residue_chunks[sequence_index].append(residue_embedding)
            remaining[sequence_index] -= 1
            if remaining[sequence_index] != 0:
                continue
            if residue_counts[sequence_index] != len(sequence):
                raise ValueError(
                    f"tokenizer residue count differs for item {sequence_index}: "
                    f"{residue_counts[sequence_index]} != {len(sequence)}"
                )
            store.save(
                sequence,
                weighted_sums[sequence_index] / residue_counts[sequence_index],
            )
            if include_residue:
                store.save_residue(sequence, np.concatenate(residue_chunks[sequence_index]))
                residue_written += 1
                del residue_chunks[sequence_index]
            del weighted_sums[sequence_index]
            del residue_counts[sequence_index]
            del remaining[sequence_index]
            written += 1
    if weighted_sums or residue_counts or remaining:
        raise RuntimeError("incomplete packed embedding state")
    return {
        "total": len(sequences),
        "resumed": resumed,
        "written": written,
        "residue_written": residue_written,
    }


def load_checkpoint(path: Path, device: torch.device) -> tuple[ESMCForMaskedLM, dict[str, Any]]:
    packet = torch.load(path, map_location="cpu", weights_only=False)
    config = ESMCConfig(**packet["model_config"])
    model = ESMCForMaskedLM(config)
    model.load_state_dict(packet["model"], strict=True)
    return model.eval().to(device), packet


def validation_mlm(
    model: ESMCForMaskedLM,
    *,
    data_root: Path,
    device: torch.device,
    context_length: int,
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, object]:
    tokenizer = ProteinTokenizer.esmc()
    batcher = MixtureBatcher(
        data_root,
        "validation",
        {"uniref90": 1.0, "mgnify": 1.0, "omg_img": 1.0},
        seed=seed,
    )
    losses: list[float] = []
    masked = 0
    with torch.inference_mode():
        for _ in range(batches):
            input_ids, attention_mask = batcher.batch(
                batch_size, context_length=context_length, tokenizer=tokenizer
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            corrupted, labels = mask_tokens(input_ids, attention_mask, tokenizer)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(corrupted, attention_mask)
                logits = output["logits"]
                per_token = F.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(labels)
            selected = labels != -100
            per_sequence = per_token.sum(dim=1) / selected.sum(dim=1).clamp_min(1)
            losses.extend(per_sequence.float().cpu().tolist())
            masked += int(selected.sum())
    values = np.asarray(losses, dtype=np.float64)
    return {
        "protocol": "heldout-cluster-representative-mlm-v1",
        "sequences": int(values.size),
        "masked_residues": masked,
        "sequence_mean_nll": float(values.mean()),
        "sequence_median_nll": float(np.median(values)),
        "perplexity": float(np.exp(values.mean())),
        "source_counts": dict(batcher.source_counts),
    }


def run_pcore(
    model: ESMCForMaskedLM,
    *,
    checkpoint: Path,
    output_root: Path,
    external_src: Path,
    pcore_root: Path,
    device: torch.device,
    batch_residues: int,
    bootstrap: int,
) -> dict[str, object]:
    """Run the current P-CORE v0.2 implementation without copying its datasets."""

    sys.path.insert(0, str(external_src))
    try:
        from autoresearch_esm.pcore_embed import (  # type: ignore[import-not-found]
            EmbeddingContract,
            EmbeddingStore,
            _window_embeddings,
            deterministic_windows,
            read_sequence_index,
        )
        from autoresearch_esm.pcore_probe import (
            run as run_probe,  # type: ignore[import-not-found]
        )
    finally:
        sys.path.pop(0)
    digest = file_sha256(checkpoint)
    contract = EmbeddingContract(
        schema_version=1,
        model_id=f"local/{model.config.name}",
        revision=digest[:12],
        weights_sha256=digest,
        hidden_size=model.config.d_model,
        maximum_residues=2046,
    )
    store_root = output_root / "pcore_embeddings" / f"{model.config.name}--{digest[:12]}"
    store = EmbeddingStore(store_root, contract)
    index_path = pcore_root / "index.jsonl"
    sequences = read_sequence_index(index_path)
    protein_embedding_result = embed_sequences_packed(
        model,
        ProteinTokenizer.esmc(),
        sequences,
        store,
        device=device,
        batch_residue_budget=batch_residues,
        include_residue=False,
        deterministic_windows=deterministic_windows,
        window_embeddings=_window_embeddings,
    )
    secondary_sequences = read_sequence_index(index_path, required_task="secondary_structure")
    residue_embedding_result = embed_sequences_packed(
        model,
        ProteinTokenizer.esmc(),
        secondary_sequences,
        store,
        device=device,
        batch_residue_budget=batch_residues,
        include_residue=True,
        deterministic_windows=deterministic_windows,
        window_embeddings=_window_embeddings,
    )
    report_path = output_root / "PCORE_REPORT.json"
    args = SimpleNamespace(
        embedding_store=str(store_root),
        processed_root=str(pcore_root / "processed"),
        raw_root=str(pcore_root / "raw"),
        output=str(report_path),
        seed=20260819,
        bootstrap=bootstrap,
        protocol="pcore-v0.2",
        target_control="none",
    )
    report = run_probe(args)
    return {
        "protocol": "pcore-v0.2",
        "embedding": {
            "protein_all_tasks": protein_embedding_result,
            "residue_secondary_structure_only": residue_embedding_result,
        },
        "pcore": report["pcore"],
        "tasks": report["tasks"],
        "report": str(report_path.resolve()),
        "report_sha256": file_sha256(report_path),
        "benchmark_status": {
            "enzyme_commission": {
                "trusted_for_model_selection": False,
                "reason": (
                    "known unresolved benchmark anomaly: released ESMC-6B collapses "
                    "to 1.607 skill while 300M/600M score about 71 despite a clean "
                    "embedding/split integrity audit"
                ),
            }
        },
    }


def _run_diagnostic_probe(
    *,
    task: str,
    external_src: Path,
    store_root: Path,
    pcore_root: Path,
    output_root: Path,
    timeout_seconds: int,
    threads: int,
) -> tuple[str, dict[str, object]]:
    output_path = output_root / "pcore_tasks" / f"{task}.json"
    log_path = output_root / "pcore_tasks" / f"{task}.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "nano_protein.pcore_task",
        "--task",
        task,
        "--external-src",
        str(external_src),
        "--embedding-store",
        str(store_root),
        "--pcore-root",
        str(pcore_root),
        "--output",
        str(output_path),
    ]
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = str(threads)
    environment["PCORE_PROBE_JOBS"] = str(threads)
    started = time.monotonic()
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                check=False,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired:
        return task, {
            "status": "timed_out",
            "seconds": time.monotonic() - started,
            "timeout_seconds": timeout_seconds,
            "log": str(log_path.resolve()),
        }
    seconds = time.monotonic() - started
    if completed.returncode != 0:
        return task, {
            "status": "failed",
            "seconds": seconds,
            "returncode": completed.returncode,
            "log": str(log_path.resolve()),
        }
    payload = json.loads(output_path.read_text())
    return task, {
        "status": "complete",
        "seconds": seconds,
        **payload["result"],
        "receipt": str(output_path.resolve()),
        "receipt_sha256": file_sha256(output_path),
    }


def run_pcore_diagnostic(
    model: ESMCForMaskedLM,
    *,
    checkpoint: Path,
    output_root: Path,
    external_src: Path,
    pcore_root: Path,
    device: torch.device,
    batch_residues: int,
    task_timeout_seconds: int,
    probe_threads: int,
) -> dict[str, object]:
    """Run a bounded, non-aggregate representation diagnostic.

    This deliberately is not called P-CORE: it evaluates three exact P-CORE
    task metrics without bootstrap, while excluding the expensive or
    quarantined tasks. Each CPU probe runs in its own time-limited process.
    """

    sys.path.insert(0, str(external_src))
    try:
        from autoresearch_esm.pcore_embed import (  # type: ignore[import-not-found]
            EmbeddingContract,
            EmbeddingStore,
            _window_embeddings,
            deterministic_windows,
            read_sequence_index,
        )
    finally:
        sys.path.pop(0)
    digest = file_sha256(checkpoint)
    contract = EmbeddingContract(
        schema_version=1,
        model_id=f"local/{model.config.name}",
        revision=digest[:12],
        weights_sha256=digest,
        hidden_size=model.config.d_model,
        maximum_residues=2046,
    )
    store_root = output_root / "pcore_embeddings" / f"{model.config.name}--{digest[:12]}"
    store = EmbeddingStore(store_root, contract)
    index_path = pcore_root / "index.jsonl"
    selected_sequences: list[str] = []
    seen: set[str] = set()
    task_sequence_counts: dict[str, int] = {}
    for task in PCORE_DIAGNOSTIC_TASKS:
        sequences = read_sequence_index(index_path, required_task=task)
        task_sequence_counts[task] = len(sequences)
        for sequence in sequences:
            if sequence not in seen:
                seen.add(sequence)
                selected_sequences.append(sequence)
    embedding_result = embed_sequences_packed(
        model,
        ProteinTokenizer.esmc(),
        selected_sequences,
        store,
        device=device,
        batch_residue_budget=batch_residues,
        include_residue=False,
        deterministic_windows=deterministic_windows,
        window_embeddings=_window_embeddings,
    )
    embedding_receipt = {
        "protocol": "pcore-diagnostic-v1",
        "task_sequence_counts": task_sequence_counts,
        "unique_sequences": len(selected_sequences),
        "embedding": embedding_result,
        "embedding_store": str(store_root.resolve()),
    }
    embedding_path = output_root / "PCORE_DIAGNOSTIC_EMBEDDING.json"
    write_json(embedding_path, embedding_receipt)

    tasks: dict[str, dict[str, object]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(PCORE_DIAGNOSTIC_TASKS)) as pool:
        futures = [
            pool.submit(
                _run_diagnostic_probe,
                task=task,
                external_src=external_src,
                store_root=store_root,
                pcore_root=pcore_root,
                output_root=output_root,
                timeout_seconds=task_timeout_seconds,
                threads=probe_threads,
            )
            for task in PCORE_DIAGNOSTIC_TASKS
        ]
        for future in concurrent.futures.as_completed(futures):
            task, result = future.result()
            tasks[task] = result
    completed = sorted(task for task, result in tasks.items() if result["status"] == "complete")
    report = {
        "schema_version": 1,
        "protocol": "pcore-diagnostic-v1",
        "claim_level": "non_comparable_partial_diagnostic",
        "aggregate_score": None,
        "bootstrap_replicates": 0,
        "task_timeout_seconds": task_timeout_seconds,
        "probe_threads_per_task": probe_threads,
        "embedding": embedding_receipt,
        "tasks": {task: tasks[task] for task in sorted(tasks)},
        "coverage": {
            "completed_tasks": completed,
            "completed_count": len(completed),
            "declared_count": len(PCORE_DIAGNOSTIC_TASKS),
            "full_pcore_task_count": 6,
        },
        "excluded": {
            "secondary_structure": "full-residue LBFGS exceeds the routine gate budget",
            "enzyme_commission": "quarantined unresolved cross-scale benchmark anomaly",
            "deeploc2": "twenty model selections are reserved for release evaluation",
        },
        "fallback_policy": (
            "timed-out or failed tasks remain explicit; no partial result is promoted "
            "to a P-CORE aggregate"
        ),
    }
    report_path = output_root / "PCORE_DIAGNOSTIC.json"
    write_json(report_path, report)
    return {
        **report,
        "report": str(report_path.resolve()),
        "report_sha256": file_sha256(report_path),
    }


def _attentions(
    model: ESMCForMaskedLM,
    tokenizer: ProteinTokenizer,
    sequence: str,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    input_ids, attention_mask = tokenizer.encode_batch([sequence], max_length=len(sequence) + 2)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids.to(device),
            attention_mask.to(device),
            output_attentions=True,
        )
    attentions = output["attentions"]
    if not isinstance(attentions, tuple) or len(attentions) != model.config.n_layers:
        raise ValueError("attention layer contract changed")
    return attentions


def run_contact_lite(
    model: ESMCForMaskedLM,
    *,
    dataset_root: Path,
    external_src: Path,
    device: torch.device,
    evaluation_chains: int,
) -> dict[str, object]:
    """Fit the frozen 20-chain probe and score a predeclared uniform subset."""

    sys.path.insert(0, str(external_src))
    try:
        from autoresearch_esm.paper_contact import score_chain  # type: ignore[import-not-found]
        from autoresearch_esm.paper_contact_model import (  # type: ignore[import-not-found]
            fit_logistic_probe,
            sampled_pair_feature_matrix,
            score_and_digest_from_attentions,
        )
        from autoresearch_esm.paper_contact_runtime import (
            ContactDataset,  # type: ignore[import-not-found]
        )
    finally:
        sys.path.pop(0)
    dataset = ContactDataset(dataset_root)
    tokenizer = ProteinTokenizer.esmc()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for chain_id in dataset.train_ids:
        _payload, chain = dataset.load_payload(chain_id)
        attention = _attentions(model, tokenizer, chain.sequence, device)
        x, y, _selection = sampled_pair_feature_matrix(
            attention,
            chain.cb_distances,
            chain_id=chain_id,
            seed=20260819,
            maximum_per_class=4096,
        )
        features.append(x)
        labels.append(y)
        del attention
        torch.cuda.empty_cache()
    coefficients, intercept, selected_c, trace = fit_logistic_probe(
        features[:16], labels[:16], features[16:], labels[16:], seed=20260819
    )
    del features, labels
    gc.collect()
    ranked = sorted(
        dataset.eval_ids,
        key=lambda chain_id: hashlib.sha256(f"20260820:{chain_id}".encode()).digest(),
    )[:evaluation_chains]
    rows: list[dict[str, object]] = []
    for chain_id in ranked:
        payload, chain = dataset.load_payload(chain_id)
        sequence = chain.sequence[:510]
        attention = _attentions(model, tokenizer, sequence, device)
        scores, feature_digest = score_and_digest_from_attentions(
            attention,
            coefficients,
            intercept,
            residue_length=len(sequence),
        )
        result = score_chain(
            chain_id,
            scores,
            chain.cb_distances,
            source_length=int(payload["source_length"]),
        )
        if result is None:
            raise ValueError(f"frozen eligible contact chain became ineligible: {chain_id}")
        row = dict(result.__dict__)
        row["attention_feature_sha256"] = feature_digest
        rows.append(row)
        del attention
        torch.cuda.empty_cache()
    precision = np.asarray([float(row["precision_at_l"]) for row in rows])
    random_precision = np.asarray([float(row["random_precision_at_l"]) for row in rows])
    return {
        "protocol": "esmc-paper-contact-lite-v1",
        "claim_level": "paper_aligned_diagnostic_not_paper_identical",
        "selection": "sha256_rank_uniform_without_replacement",
        "selection_seed": 20260820,
        "probe_train_chains": 16,
        "probe_validation_chains": 4,
        "evaluation_chains": len(rows),
        "precision_at_l": float(precision.mean()),
        "random_precision_at_l": float(random_precision.mean()),
        "selected_C": selected_c,
        "validation_trace": trace,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--external-src", type=Path)
    parser.add_argument("--pcore-root", type=Path)
    parser.add_argument("--contact-root", type=Path)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--validation-context", type=int, default=512)
    parser.add_argument("--contact-chains", type=int, default=32)
    parser.add_argument("--pcore-batch-residues", type=int, default=8192)
    parser.add_argument("--pcore-bootstrap", type=int, default=200)
    parser.add_argument("--pcore-diagnostic-timeout", type=int, default=600)
    parser.add_argument("--pcore-probe-threads", type=int, default=4)
    parser.add_argument("--run-pcore", action="store_true")
    parser.add_argument("--run-pcore-diagnostic", action="store_true")
    parser.add_argument("--run-contact", action="store_true")
    parser.add_argument("--resume-components", action="store_true")
    args = parser.parse_args()
    if args.run_pcore and args.run_pcore_diagnostic:
        parser.error("choose either --run-pcore or --run-pcore-diagnostic")
    if args.pcore_diagnostic_timeout <= 0 or args.pcore_probe_threads <= 0:
        parser.error("diagnostic timeout and probe threads must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    evaluation_started = time.monotonic()
    timing_seconds: dict[str, float] = {}
    device = torch.device("cuda", 0)
    model, checkpoint_packet = load_checkpoint(args.checkpoint, device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    resumed_components: list[str] = []
    validation_path = args.output_root / "VALIDATION_MLM.json"
    if args.resume_components and validation_path.exists():
        validation = json.loads(validation_path.read_text())
        timing_seconds["validation_mlm"] = 0.0
        resumed_components.append("validation_mlm")
    else:
        component_started = time.monotonic()
        validation = validation_mlm(
            model,
            data_root=args.data_root,
            device=device,
            context_length=args.validation_context,
            batch_size=args.validation_batch_size,
            batches=args.validation_batches,
            seed=20260821,
        )
        timing_seconds["validation_mlm"] = time.monotonic() - component_started
        write_json(validation_path, validation)
    report: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_training_seconds": checkpoint_packet["training_seconds"],
        "resumed_components": resumed_components,
        "timing_seconds": timing_seconds,
        "validation_mlm": validation,
    }
    if args.run_contact:
        if args.external_src is None or args.contact_root is None:
            parser.error("--run-contact requires --external-src and --contact-root")
        contact_path = args.output_root / "CONTACT.json"
        if args.resume_components and contact_path.exists():
            contact = json.loads(contact_path.read_text())
            timing_seconds["contact"] = 0.0
            resumed_components.append("contact")
        else:
            component_started = time.monotonic()
            contact = run_contact_lite(
                model,
                dataset_root=args.contact_root,
                external_src=args.external_src,
                device=device,
                evaluation_chains=args.contact_chains,
            )
            timing_seconds["contact"] = time.monotonic() - component_started
            write_json(contact_path, contact)
        report["contact"] = contact
    if args.run_pcore_diagnostic:
        if args.external_src is None or args.pcore_root is None:
            parser.error("--run-pcore-diagnostic requires --external-src and --pcore-root")
        component_started = time.monotonic()
        report["pcore_diagnostic"] = run_pcore_diagnostic(
            model,
            checkpoint=args.checkpoint,
            output_root=args.output_root,
            external_src=args.external_src,
            pcore_root=args.pcore_root,
            device=device,
            batch_residues=args.pcore_batch_residues,
            task_timeout_seconds=args.pcore_diagnostic_timeout,
            probe_threads=args.pcore_probe_threads,
        )
        timing_seconds["pcore_diagnostic"] = time.monotonic() - component_started
    if args.run_pcore:
        if args.external_src is None or args.pcore_root is None:
            parser.error("--run-pcore requires --external-src and --pcore-root")
        component_started = time.monotonic()
        report["pcore"] = run_pcore(
            model,
            checkpoint=args.checkpoint,
            output_root=args.output_root,
            external_src=args.external_src,
            pcore_root=args.pcore_root,
            device=device,
            batch_residues=args.pcore_batch_residues,
            bootstrap=args.pcore_bootstrap,
        )
        timing_seconds["pcore"] = time.monotonic() - component_started
    timing_seconds["total"] = time.monotonic() - evaluation_started
    report_path = args.output_root / "EVALUATION.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "evaluation_complete",
                "report": str(report_path.resolve()),
                "report_sha256": file_sha256(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
