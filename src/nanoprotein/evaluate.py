"""Checkpoint evaluation with paired search and repeated scale-up profiles."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contact_cache import ContactScoringCache
from .data import file_sha256
from .model import ESMCConfig, ESMCForMaskedLM
from .tokenizer import ProteinTokenizer, mask_tokens

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


def load_checkpoint(path: Path, device: torch.device) -> tuple[ESMCForMaskedLM, dict[str, Any]]:
    packet = torch.load(path, map_location="cpu", weights_only=False)
    config = ESMCConfig(**packet["model_config"])
    model = ESMCForMaskedLM(config)
    model.load_state_dict(packet["model"], strict=True)
    return model.eval().to(device), packet


VALIDATION_MLM_SEED = 20260821


def validation_example(
    residues: np.ndarray, digest: bytes, *, residue_limit: int, tokenizer: ProteinTokenizer,
    mask_seed: int = VALIDATION_MLM_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop and mask one protein using randomness derived only from its sequence digest."""
    seed = hashlib.sha256(VALIDATION_MLM_SEED.to_bytes(8, "big") + digest).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(seed[:8], "big"))
    excess = int(residues.size) - residue_limit
    offset = int(torch.randint(excess + 1, (1,), generator=generator)) if excess > 0 else 0
    # Keep the crop fixed across masks. The first seed continues the crop RNG;
    # subsequent seeds initialize independent mask RNG streams.
    if mask_seed != VALIDATION_MLM_SEED:
        mask_digest = hashlib.sha256(mask_seed.to_bytes(8, "big") + digest).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(mask_digest[:8], "big"))
    kept = residues[offset : offset + residue_limit]
    kept = torch.from_numpy(np.asarray(kept, dtype=np.int64))
    tokens = torch.cat(
        (torch.tensor([tokenizer.bos_id]), kept, torch.tensor([tokenizer.eos_id]))
    ).unsqueeze(0)
    corrupted, labels = mask_tokens(
        tokens, torch.ones_like(tokens, dtype=torch.bool), tokenizer, generator=generator
    )
    return corrupted[0], labels[0]


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


def fit_contact_probe_receipt(
    model: ESMCForMaskedLM,
    *,
    checkpoint_sha256: str,
    dataset_root: Path,
    external_src: Path,
    device: torch.device,
    probe_seed: int = 20260819,
) -> dict[str, object]:
    """Fit the frozen probe once so deterministic inference shards can share it."""

    sys.path.insert(0, str(external_src))
    try:
        from autoresearch_esm.paper_contact_model import (  # type: ignore[import-not-found]
            sampled_pair_feature_matrix,
        )
        from .contact_dataset import (
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
            seed=probe_seed,
            maximum_per_class=4096,
        )
        features.append(x)
        labels.append(y)
        del attention
    coefficients, intercept, selected_c, trace = _fit_logistic_probe_concurrent_exact(
        features[:16],
        labels[:16],
        features[16:],
        labels[16:],
        seed=probe_seed,
    )
    if coefficients.size != model.config.n_layers * model.config.n_heads:
        raise ValueError("frozen probe channel count differs from model attention channels")
    return {
        "schema_version": 1,
        "protocol": "autoresearch-frozen-contact-probe-v1",
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_manifest_sha256": dataset.manifest_receipt.manifest_sha256,
        "probe_train_chain_ids": dataset.train_ids[:16],
        "probe_validation_chain_ids": dataset.train_ids[16:],
        "pair_sampling_seed": probe_seed,
        "maximum_pairs_per_class": 4096,
        "channels": int(coefficients.size),
        "coefficients": np.asarray(coefficients, dtype=np.float64).tolist(),
        "intercept": float(intercept),
        "selected_C": selected_c,
        "validation_trace": trace,
    }


def _fit_logistic_probe_concurrent_exact(
    train_features: Sequence[np.ndarray],
    train_labels: Sequence[np.ndarray],
    validation_features: Sequence[np.ndarray],
    validation_labels: Sequence[np.ndarray],
    *,
    candidates_c: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int,
) -> tuple[np.ndarray, float, float, list[dict[str, float]]]:
    """Parallelize frozen C trials and a speculative canonical C=1 refit.

    Each candidate uses the exact estimator arguments from the paper evaluator. The
    all-chain refit also uses the exact canonical arguments. If validation selects a
    different C, its canonical refit runs normally, so this changes scheduling only.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    if not train_features or not validation_features:
        raise ValueError("probe fitting requires non-empty train and validation chains")
    x_train = np.concatenate([np.asarray(value, dtype=np.float32) for value in train_features])
    y_train = np.concatenate([np.asarray(value, dtype=np.int8) for value in train_labels])
    x_valid = np.concatenate(
        [np.asarray(value, dtype=np.float32) for value in validation_features]
    )
    y_valid = np.concatenate([np.asarray(value, dtype=np.int8) for value in validation_labels])
    if x_train.ndim != 2 or x_valid.ndim != 2 or x_train.shape[1] != x_valid.shape[1]:
        raise ValueError("probe feature matrices are not aligned")
    if np.unique(y_train).size != 2 or np.unique(y_valid).size != 2:
        raise ValueError("probe train and validation sets both require two classes")
    candidates = tuple(float(value) for value in candidates_c)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("logistic C values must be positive")

    def fit_candidate(value: float) -> tuple[float, float]:
        model = LogisticRegression(
            penalty="l1",
            C=value,
            solver="saga",
            class_weight=None,
            random_state=seed,
            max_iter=500,
            n_jobs=1,
        ).fit(x_train, y_train)
        score = float(average_precision_score(y_valid, model.decision_function(x_valid)))
        return value, score

    x_all = np.concatenate((x_train, x_valid), axis=0)
    y_all = np.concatenate((y_train, y_valid), axis=0)

    def fit_final(value: float) -> tuple[np.ndarray, float]:
        model = LogisticRegression(
            penalty="l1",
            C=value,
            solver="saga",
            class_weight=None,
            random_state=seed,
            max_iter=1000,
            n_jobs=1,
        ).fit(x_all, y_all)
        return np.asarray(model.coef_[0], dtype=np.float64), float(model.intercept_[0])

    speculative_c = 1.0 if 1.0 in candidates else candidates[0]
    by_c: dict[float, float] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates) + 1) as pool:
        futures = [pool.submit(fit_candidate, value) for value in candidates]
        speculative_final = pool.submit(fit_final, speculative_c)
        for future in concurrent.futures.as_completed(futures):
            value, score = future.result()
            by_c[value] = score
        speculative_result = speculative_final.result()
    trace = [{"C": value, "validation_average_precision": by_c[value]} for value in candidates]
    selected_c = max((by_c[value], -value) for value in candidates)[1] * -1.0
    coefficients, intercept = (
        speculative_result if selected_c == speculative_c else fit_final(selected_c)
    )
    return coefficients, intercept, selected_c, trace


def _score_long_range_pairs_and_digest(
    attentions: Sequence[Any],
    coefficients: np.ndarray,
    intercept: float,
    *,
    residue_length: int,
    sequence_separation: int,
    attention_planes: Callable[..., Any],
) -> tuple[np.ndarray, str]:
    """Apply the exact probe only where the frozen P@L scorer reads scores."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 1:
        raise ValueError("probe coefficients must be one-dimensional")
    i, j = np.triu_indices(residue_length, k=sequence_separation)
    pair_scores = np.full(i.size, float(intercept), dtype=np.float64)
    feature_digest = hashlib.sha256(
        json.dumps(
            {
                "dtype": "float32",
                "residue_length": int(residue_length),
                "sequence_separation": int(sequence_separation),
                "transformation": "long_range_upper_triangle_symmetrized_channels",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    observed = 0
    for index, plane in enumerate(attention_planes(attentions, residue_length=residue_length)):
        if index >= len(coefficients):
            raise ValueError("probe has fewer coefficients than attention channels")
        plane = np.asarray(plane, dtype=np.dtype("<f4"), order="C")
        selected = np.asarray(plane[i, j], dtype=np.dtype("<f4"), order="C")
        feature_digest.update(memoryview(selected).cast("B"))
        pair_scores += float(coefficients[index]) * selected
        observed = index + 1
    if observed != len(coefficients):
        raise ValueError("probe/attention channel count differs")
    scores = np.zeros((residue_length, residue_length), dtype=np.float64)
    scores[i, j] = pair_scores
    return scores, feature_digest.hexdigest()


def _symmetrized_attention_planes_batched(
    attentions: Sequence[Any], *, residue_length: int
) -> Iterator[np.ndarray]:
    """Transfer and symmetrize one complete layer while preserving plane order."""

    stop = residue_length + 1
    for layer_index, layer in enumerate(attentions):
        if hasattr(layer, "detach"):
            layer = layer.detach().float().cpu().numpy()
        array = np.asarray(layer)
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError("contact attention batch size changed")
            array = array[0]
        if array.ndim != 3 or array.shape[1] < stop or array.shape[2] < stop:
            raise ValueError(f"contact attention layer shape changed: {layer_index}")
        trimmed = np.asarray(array[:, 1:stop, 1:stop], dtype=np.float32)
        planes = np.asarray(
            (trimmed + np.swapaxes(trimmed, 1, 2)) * 0.5,
            dtype=np.float32,
            order="C",
        )
        yield from planes


def _score_sparse_long_range_pairs_and_digest(
    attentions: Sequence[Any],
    coefficients: np.ndarray,
    intercept: float,
    *,
    residue_length: int,
    sequence_separation: int,
) -> tuple[np.ndarray, str]:
    """Transfer and score only nonzero L1-probe channels in canonical order."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 1:
        raise ValueError("probe coefficients must be one-dimensional")
    i, j = np.triu_indices(residue_length, k=sequence_separation)
    pair_scores = np.full(i.size, float(intercept), dtype=np.float64)
    stop = residue_length + 1
    channel_offset = 0
    for layer_index, layer in enumerate(attentions):
        if hasattr(layer, "detach"):
            if layer.ndim == 4:
                if int(layer.shape[0]) != 1:
                    raise ValueError("contact attention batch size changed")
                layer = layer[0]
            if layer.ndim != 3 or int(layer.shape[1]) < stop or int(layer.shape[2]) < stop:
                raise ValueError(f"contact attention layer shape changed: {layer_index}")
            heads = int(layer.shape[0])
            layer_coefficients = coefficients[channel_offset : channel_offset + heads]
            selected_heads = np.flatnonzero(layer_coefficients).tolist()
            if selected_heads:
                array = layer[selected_heads, 1:stop, 1:stop].detach().float().cpu().numpy()
            else:
                array = np.empty((0, residue_length, residue_length), dtype=np.float32)
        else:
            array = np.asarray(layer)
            if array.ndim == 4:
                if array.shape[0] != 1:
                    raise ValueError("contact attention batch size changed")
                array = array[0]
            if array.ndim != 3 or array.shape[1] < stop or array.shape[2] < stop:
                raise ValueError(f"contact attention layer shape changed: {layer_index}")
            heads = int(array.shape[0])
            layer_coefficients = coefficients[channel_offset : channel_offset + heads]
            selected_heads = np.flatnonzero(layer_coefficients).tolist()
            array = np.asarray(
                array[selected_heads, 1:stop, 1:stop], dtype=np.float32, order="C"
            )
        if layer_coefficients.size != heads:
            raise ValueError("probe has fewer coefficients than attention channels")
        if selected_heads:
            planes = np.asarray(
                (array + np.swapaxes(array, 1, 2)) * 0.5,
                dtype=np.float32,
                order="C",
            )
            for local_index, head_index in enumerate(selected_heads):
                selected = np.asarray(
                    planes[local_index, i, j], dtype=np.dtype("<f4"), order="C"
                )
                pair_scores += float(layer_coefficients[head_index]) * selected
        channel_offset += heads
    if channel_offset != len(coefficients):
        raise ValueError("probe/attention channel count differs")
    digest = hashlib.sha256(
        json.dumps(
            {
                "dtype": "float64",
                "residue_length": int(residue_length),
                "sequence_separation": int(sequence_separation),
                "transformation": "sparse_probe_long_range_upper_triangle_scores",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    digest.update(memoryview(np.asarray(pair_scores, dtype=np.dtype("<f8"))).cast("B"))
    scores = np.zeros((residue_length, residue_length), dtype=np.float64)
    scores[i, j] = pair_scores
    return scores, digest.hexdigest()


def load_contact_probe_receipt(
    path: Path,
    *,
    checkpoint_sha256: str,
    dataset_manifest_sha256: str,
    channels: int,
    probe_seed: int = 20260819,
) -> tuple[np.ndarray, float, float, object]:
    """Load an exact fitted-probe receipt with checkpoint and dataset binding."""

    receipt = json.loads(path.read_text())
    if not (
        receipt.get("schema_version") == 1
        and receipt.get("protocol") == "autoresearch-frozen-contact-probe-v1"
        and receipt.get("checkpoint_sha256") == checkpoint_sha256
        and receipt.get("dataset_manifest_sha256") == dataset_manifest_sha256
        and receipt.get("pair_sampling_seed") == probe_seed
        and receipt.get("maximum_pairs_per_class") == 4096
        and receipt.get("channels") == channels
    ):
        raise ValueError("fitted contact probe receipt binding changed")
    coefficients = np.asarray(receipt.get("coefficients"), dtype=np.float64)
    if coefficients.shape != (channels,) or not np.isfinite(coefficients).all():
        raise ValueError("fitted contact probe coefficients are invalid")
    intercept = float(receipt.get("intercept"))
    selected_c = float(receipt.get("selected_C"))
    trace = receipt.get("validation_trace")
    if not np.isfinite(intercept) or not np.isfinite(selected_c) or not isinstance(trace, list):
        raise ValueError("fitted contact probe scalar contract is invalid")
    return coefficients, intercept, selected_c, trace


def run_contact_lite(
    model: ESMCForMaskedLM,
    *,
    dataset_root: Path,
    external_src: Path,
    device: torch.device,
    evaluation_chains: int,
    chain_ids_path: Path | None = None,
    probe_seed: int = 20260819,
    bootstrap: int,
    shard_index: int = 0,
    shard_count: int = 1,
    checkpoint_sha256: str | None = None,
    probe_receipt: Path | None = None,
    scoring_cache_root: Path | None = None,
    scoring_cache_preflight: Path | None = None,
) -> dict[str, object]:
    """Fit the frozen 20-chain probe and score a predeclared uniform subset."""

    sys.path.insert(0, str(external_src))
    try:
        from autoresearch_esm.paper_contact import (  # type: ignore[import-not-found]
            SEQUENCE_SEPARATION,
            score_chain,
        )
        from .contact_dataset import (
            ContactDataset,  # type: ignore[import-not-found]
        )
    finally:
        sys.path.pop(0)
    dataset = ContactDataset(dataset_root)
    if evaluation_chains <= 0 or shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid contact evaluation/shard contract")
    tokenizer = ProteinTokenizer.esmc()
    probe_receipt_sha256: str | None = None
    if probe_receipt is not None:
        if checkpoint_sha256 is None:
            raise ValueError("shared contact probe requires a checkpoint digest")
        coefficients, intercept, selected_c, trace = load_contact_probe_receipt(
            probe_receipt,
            checkpoint_sha256=checkpoint_sha256,
            dataset_manifest_sha256=dataset.manifest_receipt.manifest_sha256,
            channels=model.config.n_layers * model.config.n_heads,
            probe_seed=probe_seed,
        )
        probe_receipt_sha256 = file_sha256(probe_receipt)
    else:
        fitted = fit_contact_probe_receipt(
            model,
            checkpoint_sha256=checkpoint_sha256 or "unbound-legacy-call",
            dataset_root=dataset_root,
            external_src=external_src,
            device=device,
            probe_seed=probe_seed,
        )
        coefficients = np.asarray(fitted["coefficients"], dtype=np.float64)
        intercept = float(fitted["intercept"])
        selected_c = float(fitted["selected_C"])
        trace = fitted["validation_trace"]
    gc.collect()
    ranked_all = sorted(
        dataset.eval_ids,
        key=lambda chain_id: hashlib.sha256(f"20260820:{chain_id}".encode()).digest(),
    )[:evaluation_chains]
    if chain_ids_path is not None:
        ranked_all = chain_ids_path.read_text().splitlines()
        if len(ranked_all) != evaluation_chains or len(set(ranked_all)) != evaluation_chains:
            raise ValueError("contact selection must have exact unique chain count")
        if not set(ranked_all).issubset(dataset.eval_ids):
            raise ValueError("contact selection includes non-evaluation chains")
    if len(ranked_all) != evaluation_chains:
        raise ValueError("contact selection exceeds available population")
    ranked = ranked_all[shard_index::shard_count]
    scoring_cache: ContactScoringCache | None = None
    if scoring_cache_root is not None or scoring_cache_preflight is not None:
        if scoring_cache_root is None or scoring_cache_preflight is None:
            raise ValueError("contact scoring cache requires root plus preflight")
        scoring_cache = ContactScoringCache(
            root=scoring_cache_root,
            preflight=scoring_cache_preflight,
            dataset_manifest_sha256=dataset.manifest_receipt.manifest_sha256,
            expected_chain_ids=dataset.eval_ids,
        )
    rows: list[dict[str, object]] = []
    for chain_id in ranked:
        if scoring_cache is None:
            payload, chain = dataset.load_payload(chain_id)
            sequence = chain.sequence[:510]
        else:
            cached = scoring_cache.entries[chain_id]
            sequence = cached.sequence
        attention = _attentions(model, tokenizer, sequence, device)
        scores, score_digest = _score_sparse_long_range_pairs_and_digest(
            attention,
            coefficients,
            intercept,
            residue_length=len(sequence),
            sequence_separation=SEQUENCE_SEPARATION,
        )
        if scoring_cache is None:
            result = score_chain(
                chain_id,
                scores,
                chain.cb_distances,
                source_length=int(payload["source_length"]),
            )
            if result is None:
                raise ValueError(f"frozen eligible contact chain became ineligible: {chain_id}")
            row = dict(result.__dict__)
        else:
            row = scoring_cache.score(chain_id, scores)
        row["long_range_probe_score_sha256"] = score_digest
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
        "selection": "explicit_fixed_ids" if chain_ids_path else "sha256_rank_uniform_without_replacement",
        "chain_ids_sha256": file_sha256(chain_ids_path) if chain_ids_path else None,
        "probe_seed": probe_seed,
        "dataset_manifest_sha256": dataset.manifest_receipt.manifest_sha256,
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
        "probe_receipt_sha256": probe_receipt_sha256,
        "rows": rows,
    }


def merge_contact_evaluation(
    *,
    contact_paths: list[Path],
    expected_contact_chains: int,
    contact_bootstrap: int | None = None,
) -> dict[str, object]:
    """Strictly merge deterministic contact evaluation shards."""

    if not contact_paths or expected_contact_chains <= 0:
        raise ValueError("invalid contact-only merge contract")
    checkpoint_sha256: str | None = None
    selected_c: object | None = None
    validation_trace: object | None = None
    shards: dict[int, tuple[Path, dict[str, object], dict[str, object]]] = {}
    for path in contact_paths:
        report = json.loads(path.read_text())
        contact = report.get("contact") if isinstance(report, dict) else None
        if not isinstance(contact, dict):
            raise ValueError(f"missing contact receipt: {path}")
        if contact.get("protocol") != "esmc-paper-contact-lite-v1":
            raise ValueError(f"unexpected contact protocol: {path}")
        observed_checkpoint = str(report.get("checkpoint_sha256"))
        if checkpoint_sha256 is None:
            checkpoint_sha256 = observed_checkpoint
        elif observed_checkpoint != checkpoint_sha256:
            raise ValueError("contact shards use different checkpoints")
        shard_index = int(contact["shard_index"])
        if shard_index in shards:
            raise ValueError(f"duplicate contact shard {shard_index}")
        shards[shard_index] = (path, report, contact)

    signatures = {(part.get("chain_ids_sha256"), part.get("probe_seed"), part.get("dataset_manifest_sha256"))
                  for _, _, part in shards.values()}
    if len(signatures) != 1:
        raise ValueError("contact shards use different populations or probe seeds")
    shard_count = len(shards)
    if set(shards) != set(range(shard_count)):
        raise ValueError("contact shard indices are incomplete")
    rows: list[dict[str, object]] = []
    shard_ids: dict[int, set[str]] = {}
    components: list[dict[str, object]] = []
    for shard_index in range(shard_count):
        path, _report, contact = shards[shard_index]
        if (
            int(contact["shard_count"]) != shard_count
            or int(contact["selection_total_chains"]) != expected_contact_chains
            or int(contact["evaluation_chains"])
            != len(range(shard_index, expected_contact_chains, shard_count))
        ):
            raise ValueError(f"shard contract mismatch: {path}")
        uncertainty = contact.get("precision_at_l_uncertainty")
        if not isinstance(uncertainty, dict) or int(uncertainty["replicates"]) != 0:
            raise ValueError("contact shards must defer uncertainty aggregation")
        if selected_c is None:
            selected_c = contact["selected_C"]
            validation_trace = contact["validation_trace"]
        elif (
            contact["selected_C"] != selected_c
            or contact["validation_trace"] != validation_trace
        ):
            raise ValueError("frozen probe fit differs between shards")
        shard_rows = contact.get("rows")
        if not isinstance(shard_rows, list):
            raise ValueError(f"missing contact rows: {path}")
        rows.extend(shard_rows)
        shard_ids[shard_index] = {str(row["chain_id"]) for row in shard_rows}
        components.append(
            {
                "shard_index": shard_index,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "chains": len(shard_rows),
            }
        )

    chain_ids = [str(row["chain_id"]) for row in rows]
    if len(rows) != expected_contact_chains or len(set(chain_ids)) != len(rows):
        raise ValueError("merged contact rows are incomplete or duplicated")
    rows.sort(key=lambda row: hashlib.sha256(f"20260820:{row['chain_id']}".encode()).digest())
    for position, row in enumerate(rows):
        if str(row["chain_id"]) not in shard_ids[position % shard_count]:
            raise ValueError("rows violate the frozen deterministic sharding")
    precision = [float(row["precision_at_l"]) for row in rows]
    random_precision = [float(row["random_precision_at_l"]) for row in rows]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in precision):
        raise ValueError("invalid P@L values")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in random_precision):
        raise ValueError("invalid random P@L values")
    receipt = {
        "schema_version": 1,
        "protocol": "autoresearch-frozen-full-contact-merge-v1",
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_chains": len(rows),
        "p_at_l": float(np.asarray(precision, dtype=np.float64).mean()),
        "random_p_at_l": float(np.asarray(random_precision, dtype=np.float64).mean()),
        "selected_C": selected_c,
        "validation_trace": validation_trace,
        "selection_seed": 20260820,
        "components": components,
    }
    if contact_bootstrap is not None:
        contact = dict(shards[0][2])
        probe_digest = contact.get("probe_receipt_sha256")
        if any(
            part.get("probe_receipt_sha256") != probe_digest for _, _, part in shards.values()
        ):
            raise ValueError("contact shards use different fitted probe receipts")
        contact.update(
            evaluation_chains=len(rows),
            selection_total_chains=len(rows),
            shard_index=0,
            shard_count=1,
            precision_at_l=receipt["p_at_l"],
            random_precision_at_l=receipt["random_p_at_l"],
            precision_at_l_uncertainty=bootstrap_mean_interval(
                np.asarray(precision), replicates=contact_bootstrap, seed=20260820
            ),
            rows=rows,
        )
        receipt["contact"] = contact
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("search", "scaleup", "component"), default="scaleup",
                        help="search: paired 8192; scaleup: 5 probe/5 mask final metrics; component: internal contact worker")
    parser.add_argument("--prepared-root", type=Path, help="verified fixed MLM caches")
    parser.add_argument("--contact-chain-ids", type=Path)
    parser.add_argument("--probe-seed", type=int, default=20260819)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--external-src", type=Path)
    parser.add_argument("--contact-root", type=Path)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=32,
        help="Validation proteins per forward pass; does not change the score",
    )
    parser.add_argument("--validation-context", type=int, default=512)
    parser.add_argument(
        "--contact-chains", type=int, help="component population size; profiles freeze their own counts"
    )
    parser.add_argument("--contact-bootstrap", type=int, default=0)
    parser.add_argument(
        "--contact-mode",
        choices=("parallel", "serial"),
        default="parallel",
        help="parallel (default) shares one probe across workers; serial uses one process",
    )
    parser.add_argument(
        "--contact-gpus", help="GPU identifiers; defaults to visible CUDA devices"
    )
    parser.add_argument("--contact-workers", type=int, help="default: eight workers per GPU")
    parser.add_argument("--contact-shard-index", type=int, default=0)
    parser.add_argument("--contact-shard-count", type=int, default=1)
    parser.add_argument("--contact-probe-receipt", type=Path)
    parser.add_argument("--contact-scoring-cache-root", type=Path)
    parser.add_argument("--contact-scoring-cache-preflight", type=Path)
    parser.add_argument("--resume-components", action="store_true")
    args = parser.parse_args(argv)
    if args.profile != "component":
        expected_count = 8192 if args.profile == "search" else 26062
        if args.contact_chains not in (None, expected_count) or args.validation_context != 512:
            parser.error("profile populations and context are fixed")
        if args.contact_mode != "parallel" or args.probe_seed != 20260819:
            parser.error("profile execution uses parallel contact and fixed probe seeds")
        if args.contact_chain_ids is not None or args.contact_probe_receipt is not None or args.contact_shard_count != 1:
            parser.error("profile populations and probes are fixed; custom shards require --profile component")
        evaluation_root = args.data_root.parent / "evaluation"
        args.external_src = args.external_src or evaluation_root / "source"
        args.contact_root = args.contact_root or evaluation_root / "contact-v3"
        args.prepared_root = args.prepared_root or evaluation_root / "prepared-v3"
    if args.contact_chains is None:
        args.contact_chains = 8192 if args.profile == "search" else 26062
    if args.validation_batch_size <= 0:
        parser.error("validation batch size must be positive")
    if args.profile == "component" and (args.external_src is None or args.contact_root is None):
        parser.error("contact workers require --external-src and --contact-root")
    if args.contact_chains <= 0 or args.contact_bootstrap < 0:
        parser.error("contact chains must be positive and bootstrap nonnegative")
    if args.contact_workers is not None and args.contact_workers <= 0:
        parser.error("contact workers must be positive")
    if not 0 <= args.contact_shard_index < args.contact_shard_count:
        parser.error("invalid contact shard index/count")
    return args


def main() -> None:
    args = parse_args()
    if args.profile != "component":
        from .evaluation_profiles import run_profile
        run_profile(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    np.random.seed(20260821)
    torch.manual_seed(20260821)
    torch.cuda.manual_seed_all(20260821)
    evaluation_started = time.monotonic()
    timing_seconds: dict[str, float] = {}
    device = torch.device("cuda", 0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    resumed_components: list[str] = []
    report: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "resumed_components": resumed_components,
        "timing_seconds": timing_seconds,
    }
    parallel_contact = (
        args.contact_mode == "parallel"
        and args.contact_shard_count == 1
        and args.contact_probe_receipt is None
    )
    if parallel_contact:
        from .contact_parallel import run_contact_parallel

        component_started = time.monotonic()
        parallel_report = run_contact_parallel(args, str(report["checkpoint_sha256"]))
        report.update(parallel_report)
        timing_seconds["contact"] = time.monotonic() - component_started
    if not parallel_contact:
        model, checkpoint_packet = load_checkpoint(args.checkpoint, device)
        report["checkpoint_training_seconds"] = checkpoint_packet["training_seconds"]
    if not parallel_contact:
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
                chain_ids_path=args.contact_chain_ids,
                probe_seed=args.probe_seed,
                bootstrap=args.contact_bootstrap,
                shard_index=args.contact_shard_index,
                shard_count=args.contact_shard_count,
                checkpoint_sha256=str(report["checkpoint_sha256"]),
                probe_receipt=args.contact_probe_receipt,
                scoring_cache_root=args.contact_scoring_cache_root,
                scoring_cache_preflight=args.contact_scoring_cache_preflight,
            )
            timing_seconds["contact"] = time.monotonic() - component_started
            write_json(contact_path, contact)
        report["contact"] = contact
    timing_seconds["total"] = time.monotonic() - evaluation_started
    report["peak_cuda_memory_bytes"] = max(
        int(report.get("peak_cuda_memory_bytes", 0)),
        int(torch.cuda.max_memory_allocated(device)),
    )
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
