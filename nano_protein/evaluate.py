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
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import MixtureBatcher, file_sha256
from .model import ESMCConfig, ESMCForMaskedLM
from .tokenizer import ProteinTokenizer, mask_tokens

PCORE_DIAGNOSTIC_TASKS = (
    "remote_homology",
    "flip2_hydro_low_to_high",
)
PCORE_TASKS = (
    "remote_homology",
    "secondary_structure",
    "enzyme_commission",
    "deeploc2",
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


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap a mean in bounded-memory chunks."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    if replicates < 0:
        raise ValueError("bootstrap replicates cannot be negative")
    if replicates == 0:
        return {"replicates": 0, "confidence_interval_95": None}
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk_size = 128
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "replicates": replicates,
        "unit_count": int(values.size),
        "confidence_interval_95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


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
    # The model preserves sequence independence, so length bucketing is exactly
    # equivalent while avoiding dense tokenizer padding between unlike windows.
    records.sort(key=lambda record: (-len(record[2]), record[0]))

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
    task_parallel: int,
    probe_threads: int,
) -> dict[str, object]:
    """Run exact P-CORE v0.2 as restartable, bounded-parallel task processes."""

    if bootstrap != 10_000:
        raise ValueError("exact taskwise P-CORE v0.2 requires 10,000 bootstrap replicates")
    if task_parallel <= 0 or probe_threads <= 0:
        raise ValueError("task parallelism and probe threads must be positive")

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
    task_root = output_root / "pcore_tasks_exact"
    task_root.mkdir(parents=True, exist_ok=True)
    index_digest = file_sha256(index_path)
    environment = os.environ.copy()
    environment["PCORE_PROBE_JOBS"] = str(probe_threads)
    environment["OMP_NUM_THREADS"] = str(probe_threads)
    environment["MKL_NUM_THREADS"] = str(probe_threads)
    environment["OPENBLAS_NUM_THREADS"] = str(probe_threads)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{external_src}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(external_src)
    )

    def run_task(task: str) -> tuple[str, dict[str, object]]:
        partial_path = task_root / f"{task}.json"
        log_path = task_root / f"{task}.log"
        command = [
            sys.executable,
            "-m",
            "autoresearch_esm.pcore_taskwise",
            "run-task",
            "--task",
            task,
            "--index-jsonl",
            str(index_path),
            "--expected-index-sha256",
            index_digest,
            "--embedding-store",
            str(store_root),
            "--processed-root",
            str(pcore_root / "processed"),
            "--raw-root",
            str(pcore_root / "raw"),
            "--output",
            str(partial_path),
            "--seed",
            "20260819",
        ]
        started = time.monotonic()
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                check=False,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"exact P-CORE task {task} failed; see {log_path}")
        return task, {
            "seconds": time.monotonic() - started,
            "partial": str(partial_path.resolve()),
            "partial_sha256": file_sha256(partial_path),
            "log": str(log_path.resolve()),
        }

    task_runs: dict[str, dict[str, object]] = {}
    workers = min(task_parallel, len(PCORE_TASKS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, task) for task in PCORE_TASKS]
        for future in concurrent.futures.as_completed(futures):
            task, task_run = future.result()
            task_runs[task] = task_run

    report_path = output_root / "PCORE_REPORT.json"
    reduce_log = task_root / "reduce.log"
    reduce_command = [
        sys.executable,
        "-m",
        "autoresearch_esm.pcore_taskwise",
        "reduce",
    ]
    for task in PCORE_TASKS:
        reduce_command.extend(("--partial", str(task_root / f"{task}.json")))
    reduce_command.extend(("--output", str(report_path)))
    with reduce_log.open("w") as log:
        reduced = subprocess.run(
            reduce_command,
            check=False,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if reduced.returncode != 0:
        raise RuntimeError(f"exact P-CORE reduction failed; see {reduce_log}")
    report = json.loads(report_path.read_text())
    trust = report["trust_contract"]
    return {
        "protocol": "pcore-v0.3-q4",
        "probe_protocol": "pcore-v0.2",
        "execution": {
            "mode": "restartable_taskwise_subprocesses",
            "maximum_parallel_tasks": workers,
            "threads_per_task": probe_threads,
            "task_runs": {task: task_runs[task] for task in PCORE_TASKS},
            "reduction_log": str(reduce_log.resolve()),
        },
        "embedding": {
            "protein_all_tasks": protein_embedding_result,
            "residue_secondary_structure_only": residue_embedding_result,
        },
        "pcore": report["pcore"],
        "legacy_pcore_v0_2": report["legacy_pcore_v0_2"],
        "tasks": report["tasks"],
        "report": str(report_path.resolve()),
        "report_sha256": file_sha256(report_path),
        "trust_contract": trust,
        "benchmark_status": trust["tasks"],
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

    This deliberately is not called P-CORE: it evaluates two trusted P-CORE
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
            "human_ppi": "quarantined weak discrimination on the current 237-pair test",
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
    bootstrap: int,
    shard_index: int = 0,
    shard_count: int = 1,
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
    if evaluation_chains <= 0 or shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid contact evaluation/shard contract")
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
    coefficients, intercept, selected_c, trace = fit_logistic_probe(
        features[:16], labels[:16], features[16:], labels[16:], seed=20260819
    )
    del features, labels
    gc.collect()
    ranked_all = sorted(
        dataset.eval_ids,
        key=lambda chain_id: hashlib.sha256(f"20260820:{chain_id}".encode()).digest(),
    )[:evaluation_chains]
    ranked = ranked_all[shard_index::shard_count]
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
    precision = np.asarray([float(row["precision_at_l"]) for row in rows])
    random_precision = np.asarray([float(row["random_precision_at_l"]) for row in rows])
    uncertainty = bootstrap_mean_interval(
        precision,
        replicates=bootstrap,
        seed=20260820,
    )
    return {
        "protocol": "esmc-paper-contact-lite-v1",
        "claim_level": "paper_aligned_diagnostic_not_paper_identical",
        "selection": "sha256_rank_uniform_without_replacement",
        "selection_seed": 20260820,
        "selection_total_chains": len(ranked_all),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "probe_train_chains": 16,
        "probe_validation_chains": 4,
        "evaluation_chains": len(rows),
        "precision_at_l": float(precision.mean()),
        "precision_at_l_uncertainty": uncertainty,
        "random_precision_at_l": float(random_precision.mean()),
        "selected_C": selected_c,
        "validation_trace": trace,
        "rows": rows,
    }


def merge_full_evaluation(
    *,
    contact_paths: list[Path],
    pcore_path: Path,
    expected_contact_chains: int,
    contact_bootstrap: int,
) -> dict[str, object]:
    """Validate and merge parallel exact-evaluation component receipts."""

    if not contact_paths or expected_contact_chains <= 0 or contact_bootstrap <= 0:
        raise ValueError("invalid full-evaluation merge contract")
    pcore_report = json.loads(pcore_path.read_text())
    if not isinstance(pcore_report, dict) or not isinstance(pcore_report.get("pcore"), dict):
        raise ValueError("P-CORE component is absent from its evaluation receipt")
    if not isinstance(pcore_report.get("validation_mlm"), dict):
        raise ValueError("held-out MLM component is absent from the P-CORE receipt")
    checkpoint_sha256 = pcore_report.get("checkpoint_sha256")

    shards: dict[int, tuple[Path, dict[str, object], dict[str, object]]] = {}
    for path in contact_paths:
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or not isinstance(report.get("contact"), dict):
            raise ValueError(f"contact component is missing from {path}")
        contact = report["contact"]
        if report.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("parallel evaluation components use different checkpoints")
        shard_index = int(contact["shard_index"])
        if shard_index in shards:
            raise ValueError(f"duplicate contact shard {shard_index}")
        shards[shard_index] = (path, report, contact)
    shard_count = len(shards)
    if set(shards) != set(range(shard_count)):
        raise ValueError("contact shard indices are incomplete")

    rows: list[dict[str, object]] = []
    shard_chain_ids: dict[int, set[str]] = {}
    selected_c: object | None = None
    validation_trace: object | None = None
    component_receipts: list[dict[str, object]] = []
    component_totals: list[float] = []
    for shard_index in range(shard_count):
        path, report, contact = shards[shard_index]
        if (
            int(contact["shard_count"]) != shard_count
            or int(contact["selection_total_chains"]) != expected_contact_chains
            or int(contact["evaluation_chains"])
            != len(range(shard_index, expected_contact_chains, shard_count))
        ):
            raise ValueError(f"contact shard {shard_index} has the wrong selection contract")
        uncertainty = contact.get("precision_at_l_uncertainty")
        if not isinstance(uncertainty, dict) or int(uncertainty["replicates"]) != 0:
            raise ValueError("contact shards must defer bootstrap to the exact merger")
        if selected_c is None:
            selected_c = contact["selected_C"]
            validation_trace = contact["validation_trace"]
        elif (
            contact["selected_C"] != selected_c
            or contact["validation_trace"] != validation_trace
        ):
            raise ValueError("contact probe fit differs across shards")
        shard_rows = contact.get("rows")
        if not isinstance(shard_rows, list):
            raise ValueError(f"contact rows are missing from shard {shard_index}")
        rows.extend(shard_rows)
        shard_chain_ids[shard_index] = {str(row["chain_id"]) for row in shard_rows}
        component_receipts.append(
            {
                "shard_index": shard_index,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "chains": len(shard_rows),
            }
        )
        timing = report.get("timing_seconds", {})
        component_totals.append(float(timing.get("total", 0.0)))

    chain_ids = [str(row["chain_id"]) for row in rows]
    if len(rows) != expected_contact_chains or len(set(chain_ids)) != len(rows):
        raise ValueError("merged contact rows are incomplete or duplicated")
    rows.sort(key=lambda row: hashlib.sha256(f"20260820:{row['chain_id']}".encode()).digest())
    for position, row in enumerate(rows):
        if str(row["chain_id"]) not in shard_chain_ids[position % shard_count]:
            raise ValueError("contact rows do not follow the frozen deterministic sharding")
    precision = np.asarray([float(row["precision_at_l"]) for row in rows])
    random_precision = np.asarray([float(row["random_precision_at_l"]) for row in rows])
    contact = {
        "protocol": "esmc-paper-contact-full-parallel-v1",
        "claim_level": "paper_aligned_diagnostic_not_paper_identical",
        "selection": "sha256_rank_uniform_without_replacement",
        "selection_seed": 20260820,
        "probe_train_chains": 16,
        "probe_validation_chains": 4,
        "evaluation_chains": len(rows),
        "precision_at_l": float(precision.mean()),
        "precision_at_l_uncertainty": bootstrap_mean_interval(
            precision,
            replicates=contact_bootstrap,
            seed=20260820,
        ),
        "random_precision_at_l": float(random_precision.mean()),
        "selected_C": selected_c,
        "validation_trace": validation_trace,
        "execution": {
            "mode": "deterministic_chain_shards",
            "shards": shard_count,
            "component_receipts": component_receipts,
        },
        "rows": rows,
    }
    pcore_timing = pcore_report.get("timing_seconds", {})
    pcore_seconds = float(pcore_timing.get("total", 0.0))
    return {
        "schema_version": 1,
        "protocol": "full-parallel-evaluation-v1",
        "checkpoint": pcore_report["checkpoint"],
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_training_seconds": pcore_report["checkpoint_training_seconds"],
        "validation_mlm": pcore_report["validation_mlm"],
        "contact": contact,
        "pcore": pcore_report["pcore"],
        "timing_seconds": {
            "pcore_component": pcore_seconds,
            "longest_contact_shard": max(component_totals),
            "parallel_critical_path": max([pcore_seconds, *component_totals]),
        },
        "peak_cuda_memory_bytes": max(
            [
                int(pcore_report.get("peak_cuda_memory_bytes", 0)),
                *[
                    int(report.get("peak_cuda_memory_bytes", 0))
                    for _path, report, _contact in shards.values()
                ],
            ]
        ),
        "component_receipts": {
            "pcore": {
                "path": str(pcore_path.resolve()),
                "sha256": file_sha256(pcore_path),
            },
            "contact_shards": component_receipts,
        },
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
    parser.add_argument("--contact-bootstrap", type=int, default=5000)
    parser.add_argument("--contact-shard-index", type=int, default=0)
    parser.add_argument("--contact-shard-count", type=int, default=1)
    parser.add_argument("--pcore-batch-residues", type=int, default=8192)
    parser.add_argument("--pcore-bootstrap", type=int, default=10000)
    parser.add_argument("--pcore-task-parallel", type=int, default=2)
    parser.add_argument("--pcore-diagnostic-timeout", type=int, default=600)
    parser.add_argument("--pcore-probe-threads", type=int, default=4)
    parser.add_argument("--run-pcore", action="store_true")
    parser.add_argument("--run-pcore-diagnostic", action="store_true")
    parser.add_argument("--run-contact", action="store_true")
    parser.add_argument("--skip-validation-mlm", action="store_true")
    parser.add_argument("--resume-components", action="store_true")
    args = parser.parse_args()
    if args.run_pcore and args.run_pcore_diagnostic:
        parser.error("choose either --run-pcore or --run-pcore-diagnostic")
    if args.pcore_diagnostic_timeout <= 0 or args.pcore_probe_threads <= 0:
        parser.error("diagnostic timeout and probe threads must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    np.random.seed(20260821)
    torch.manual_seed(20260821)
    torch.cuda.manual_seed_all(20260821)
    evaluation_started = time.monotonic()
    timing_seconds: dict[str, float] = {}
    device = torch.device("cuda", 0)
    model, checkpoint_packet = load_checkpoint(args.checkpoint, device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    resumed_components: list[str] = []
    report: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_training_seconds": checkpoint_packet["training_seconds"],
        "resumed_components": resumed_components,
        "timing_seconds": timing_seconds,
    }
    if not args.skip_validation_mlm:
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
        report["validation_mlm"] = validation
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
                bootstrap=args.contact_bootstrap,
                shard_index=args.contact_shard_index,
                shard_count=args.contact_shard_count,
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
            task_parallel=args.pcore_task_parallel,
            probe_threads=args.pcore_probe_threads,
        )
        timing_seconds["pcore"] = time.monotonic() - component_started
    timing_seconds["total"] = time.monotonic() - evaluation_started
    report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
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
