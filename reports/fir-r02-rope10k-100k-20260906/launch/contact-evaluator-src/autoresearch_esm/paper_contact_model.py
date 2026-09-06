"""Model-side producer for the paper-faithful ESMC attention-contact probe.

The module keeps large attention tensors chain-local. Training features may be
written to NumPy files for the frozen 20-chain probe; evaluation applies the
fitted linear weights one attention plane at a time and never materializes the
full ``pairs × layers × heads`` matrix.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .paper_contact import (
    DISTANCE_THRESHOLD_ANGSTROM,
    SEQUENCE_SEPARATION,
    score_chain,
)

PRODUCER_PROTOCOL_ID = "esmc-paper-contact-producer-v1"
PROBE_PROTOCOL_ID = "esmc-paper-contact-logistic-probe-v1"


def file_sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def attention_planes(
    attentions: Sequence[Any],
    *,
    residue_length: int,
    special_token_offset: int = 1,
) -> Iterable[np.ndarray]:
    """Yield symmetrized residue attention planes in layer-major/head-major order."""

    residue_length = int(residue_length)
    if residue_length <= 0:
        raise ValueError("residue_length must be positive")
    if special_token_offset < 0:
        raise ValueError("special_token_offset cannot be negative")
    stop = special_token_offset + residue_length
    if not attentions:
        raise ValueError("model returned no attention layers")
    expected_heads: int | None = None
    for layer_index, layer in enumerate(attentions):
        array = _numpy(layer)
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError("paper-contact producer requires batch size one")
            array = array[0]
        if array.ndim != 3:
            raise ValueError(f"attention layer {layer_index} must have rank 3 or 4")
        if expected_heads is None:
            expected_heads = int(array.shape[0])
        elif array.shape[0] != expected_heads:
            raise ValueError("attention head count changed across layers")
        if array.shape[1] < stop or array.shape[2] < stop:
            raise ValueError("attention map is shorter than the residue span")
        trimmed = np.asarray(
            array[:, special_token_offset:stop, special_token_offset:stop],
            dtype=np.float32,
        )
        for head in trimmed:
            # An average instead of a sum keeps features on the probability
            # scale; an unconstrained intercept makes the two conventions
            # equivalent absent regularization.
            yield np.asarray((head + head.T) * 0.5, dtype=np.float32)


def long_range_pairs(
    distances: np.ndarray,
    *,
    sequence_separation: int = SEQUENCE_SEPARATION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance matrix must be square")
    i, j = np.triu_indices(distances.shape[0], k=int(sequence_separation))
    valid = np.isfinite(distances[i, j])
    return i[valid], j[valid], distances[i[valid], j[valid]]


def pair_feature_matrix(
    attentions: Sequence[Any],
    distances: np.ndarray,
    *,
    sequence_separation: int = SEQUENCE_SEPARATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize one chain's probe design matrix and binary contact labels."""

    distances = np.asarray(distances, dtype=np.float64)
    i, j, pair_distances = long_range_pairs(distances, sequence_separation=sequence_separation)
    planes = list(attention_planes(attentions, residue_length=distances.shape[0]))
    features = np.empty((i.size, len(planes)), dtype=np.float32)
    for column, plane in enumerate(planes):
        features[:, column] = plane[i, j]
    labels = pair_distances < DISTANCE_THRESHOLD_ANGSTROM
    return features, labels.astype(np.int8)


def sampled_pair_feature_matrix(
    attentions: Sequence[Any],
    distances: np.ndarray,
    *,
    chain_id: str,
    seed: int = 20_260_819,
    maximum_per_class: int = 4_096,
    sequence_separation: int = SEQUENCE_SEPARATION,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build a deterministic class-balanced contact-probe matrix for one chain."""

    chain_id = str(chain_id).strip()
    if not chain_id:
        raise ValueError("chain_id must be non-empty")
    maximum_per_class = int(maximum_per_class)
    if maximum_per_class <= 0:
        raise ValueError("maximum_per_class must be positive")
    distances = np.asarray(distances, dtype=np.float64)
    i, j, pair_distances = long_range_pairs(
        distances, sequence_separation=sequence_separation
    )
    labels = pair_distances < DISTANCE_THRESHOLD_ANGSTROM
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("sampled contact features require both contact classes")
    chain_seed = int.from_bytes(
        hashlib.sha256(f"{int(seed)}:{chain_id}".encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(chain_seed)

    def choose(indices: np.ndarray) -> np.ndarray:
        if indices.size <= maximum_per_class:
            return indices
        return np.sort(rng.choice(indices, size=maximum_per_class, replace=False))

    selected = np.sort(np.concatenate((choose(positive), choose(negative))))
    selected_i = i[selected]
    selected_j = j[selected]
    planes = list(attention_planes(attentions, residue_length=distances.shape[0]))
    features = np.empty((selected.size, len(planes)), dtype=np.float32)
    for column, plane in enumerate(planes):
        features[:, column] = plane[selected_i, selected_j]
    selected_labels = labels[selected].astype(np.int8)
    selection_contract = {
        "chain_id": chain_id,
        "maximum_per_class": maximum_per_class,
        "negative_selected": int(np.sum(selected_labels == 0)),
        "positive_selected": int(np.sum(selected_labels == 1)),
        "seed": int(seed),
        "selected_pair_indices": selected.tolist(),
    }
    selection_sha256 = hashlib.sha256(
        json.dumps(
            selection_contract, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    ).hexdigest()
    return features, selected_labels, selection_sha256


def score_from_attentions(
    attentions: Sequence[Any],
    coefficients: np.ndarray,
    intercept: float,
    *,
    residue_length: int,
) -> np.ndarray:
    """Apply a fitted linear contact probe without a giant pair-feature matrix."""

    output, _feature_sha256 = score_and_digest_from_attentions(
        attentions,
        coefficients,
        intercept,
        residue_length=residue_length,
    )
    return output


def score_and_digest_from_attentions(
    attentions: Sequence[Any],
    coefficients: np.ndarray,
    intercept: float,
    *,
    residue_length: int,
) -> tuple[np.ndarray, str]:
    """Apply the probe and hash the exact streamed symmetrized feature planes."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 1:
        raise ValueError("probe coefficients must be one-dimensional")
    output = np.full((residue_length, residue_length), float(intercept), dtype=np.float64)
    feature_digest = hashlib.sha256(
        json.dumps(
            {
                "dtype": "float32",
                "residue_length": int(residue_length),
                "transformation": "all_layer_all_head_symmetrize_only",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    observed = 0
    for index, plane in enumerate(attention_planes(attentions, residue_length=residue_length)):
        if index >= len(coefficients):
            raise ValueError(
                f"probe/attention channel mismatch: coefficients={len(coefficients)}, "
                f"attention_planes_at_least={index + 1}"
            )
        plane = np.asarray(plane, dtype=np.dtype("<f4"), order="C")
        feature_digest.update(memoryview(plane).cast("B"))
        output += float(coefficients[index]) * plane
        observed = index + 1
    if observed != len(coefficients):
        raise ValueError(
            f"probe/attention channel mismatch: coefficients={len(coefficients)}, "
            f"attention_planes={observed}"
        )
    return output, feature_digest.hexdigest()


@dataclass(frozen=True)
class ProbeReceipt:
    schema_version: int
    protocol_id: str
    model_id: str
    model_revision: str
    weight_manifest_sha256: str
    dataset_manifest_sha256: str
    train_chain_ids: tuple[str, ...]
    validation_chain_ids: tuple[str, ...]
    feature_checkpoint_sha256s: dict[str, str]
    channels: int
    selected_C: float
    seed: int
    penalty: str
    solver: str
    class_weight: str | None
    coefficients_sha256: str


def fit_logistic_probe(
    train_features: Sequence[np.ndarray],
    train_labels: Sequence[np.ndarray],
    validation_features: Sequence[np.ndarray],
    validation_labels: Sequence[np.ndarray],
    *,
    candidates_C: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    seed: int = 20_260_819,
) -> tuple[np.ndarray, float, float, list[dict[str, float]]]:
    """Select L1 logistic regularization on frozen validation chains, then refit."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    if not train_features or not validation_features:
        raise ValueError("probe fitting requires non-empty train and validation chains")
    x_train = np.concatenate([np.asarray(value, dtype=np.float32) for value in train_features])
    y_train = np.concatenate([np.asarray(value, dtype=np.int8) for value in train_labels])
    x_valid = np.concatenate([np.asarray(value, dtype=np.float32) for value in validation_features])
    y_valid = np.concatenate([np.asarray(value, dtype=np.int8) for value in validation_labels])
    if x_train.ndim != 2 or x_valid.ndim != 2 or x_train.shape[1] != x_valid.shape[1]:
        raise ValueError("probe feature matrices are not aligned")
    if np.unique(y_train).size != 2 or np.unique(y_valid).size != 2:
        raise ValueError("probe train and validation sets both require two classes")

    trace: list[dict[str, float]] = []
    candidates: list[tuple[float, float]] = []
    for value in candidates_C:
        value = float(value)
        if value <= 0:
            raise ValueError("logistic C values must be positive")
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
        trace.append({"C": value, "validation_average_precision": score})
        candidates.append((score, -value))
    _score, negative_C = max(candidates)
    selected_C = -negative_C
    x_all = np.concatenate((x_train, x_valid), axis=0)
    y_all = np.concatenate((y_train, y_valid), axis=0)
    final = LogisticRegression(
        penalty="l1",
        C=selected_C,
        solver="saga",
        class_weight=None,
        random_state=seed,
        max_iter=1000,
        n_jobs=1,
    ).fit(x_all, y_all)
    return (
        np.asarray(final.coef_[0], dtype=np.float64),
        float(final.intercept_[0]),
        selected_C,
        trace,
    )


def _sha256_text(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _nonempty(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _frozen_probe_chains(
    train_chain_ids: Sequence[str],
    validation_chain_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    train = tuple(_nonempty(value, field="train_chain_id") for value in train_chain_ids)
    validation = tuple(
        _nonempty(value, field="validation_chain_id") for value in validation_chain_ids
    )
    if not train or not validation:
        raise ValueError("probe train and validation chain sets must both be non-empty")
    combined = train + validation
    if len(combined) != 20:
        raise ValueError(f"paper contact probe requires exactly 20 chains, got {len(combined)}")
    if len(set(combined)) != len(combined):
        raise ValueError("paper contact probe chain IDs must be distinct across train/validation")
    return train, validation


def _publish_immutable_bytes(path: Path, payload: bytes) -> str:
    """Publish exact bytes once; identical retries are idempotent."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"different immutable artifact already exists: {path}")
        return hashlib.sha256(payload).hexdigest()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(f"different immutable artifact already exists: {path}")
        return hashlib.sha256(payload).hexdigest()
    finally:
        temporary.unlink(missing_ok=True)


def save_probe(
    output_root: Path,
    coefficients: np.ndarray,
    intercept: float,
    *,
    model_id: str,
    model_revision: str,
    weight_manifest_sha256: str,
    dataset_manifest_sha256: str,
    train_chain_ids: Sequence[str],
    validation_chain_ids: Sequence[str],
    feature_checkpoint_sha256s: dict[str, str],
    selected_C: float,
    validation_trace: Sequence[dict[str, float]],
    seed: int = 20_260_819,
) -> str:
    """Immutably publish a probe bound to exactly 20 pre-probe feature artifacts."""

    model_id = _nonempty(model_id, field="model_id")
    model_revision = _nonempty(model_revision, field="model_revision")
    weight_manifest_sha256 = _sha256_text(weight_manifest_sha256, field="weight_manifest_sha256")
    dataset_manifest_sha256 = _sha256_text(dataset_manifest_sha256, field="dataset_manifest_sha256")
    train_chain_ids, validation_chain_ids = _frozen_probe_chains(
        train_chain_ids, validation_chain_ids
    )
    expected_chain_ids = set(train_chain_ids + validation_chain_ids)
    if set(feature_checkpoint_sha256s) != expected_chain_ids:
        raise ValueError("feature checkpoint digests must exactly cover the 20 probe chains")
    feature_checkpoint_sha256s = {
        chain_id: _sha256_text(
            feature_checkpoint_sha256s[chain_id],
            field=f"feature_checkpoint_sha256s[{chain_id!r}]",
        )
        for chain_id in sorted(expected_chain_ids)
    }
    coefficients = np.asarray(coefficients, dtype=np.float64)
    intercept = float(intercept)
    selected_C = float(selected_C)
    if coefficients.ndim != 1 or coefficients.size == 0 or not np.isfinite(coefficients).all():
        raise ValueError("probe coefficients must be a non-empty finite vector")
    if not np.isfinite(intercept):
        raise ValueError("probe intercept must be finite")
    if not np.isfinite(selected_C) or selected_C <= 0:
        raise ValueError("selected_C must be finite and positive")

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    coefficient_path = output_root / "probe-coefficients.npz"
    coefficient_buffer = io.BytesIO()
    np.savez(
        coefficient_buffer,
        coefficients=coefficients,
        intercept=np.asarray(intercept, dtype=np.float64),
    )
    coefficient_payload = coefficient_buffer.getvalue()
    coefficient_sha256 = hashlib.sha256(coefficient_payload).hexdigest()
    receipt = {
        "schema_version": 1,
        "protocol_id": PROBE_PROTOCOL_ID,
        "model_id": model_id,
        "model_revision": model_revision,
        "weight_manifest_sha256": weight_manifest_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "train_chain_ids": list(train_chain_ids),
        "validation_chain_ids": list(validation_chain_ids),
        "feature_checkpoint_sha256s": feature_checkpoint_sha256s,
        "channels": int(coefficients.size),
        "selected_C": selected_C,
        "seed": int(seed),
        "penalty": "l1",
        "solver": "saga",
        "class_weight": None,
        "coefficients_sha256": coefficient_sha256,
        "validation_trace": list(validation_trace),
    }
    target = output_root / "PROBE_RECEIPT.json"
    # Fully validate and serialize the receipt before publishing either file,
    # so a malformed trace cannot leave a new coefficient-only probe behind.
    encoded = (json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    published_coefficient_sha256 = _publish_immutable_bytes(coefficient_path, coefficient_payload)
    if published_coefficient_sha256 != coefficient_sha256:  # pragma: no cover
        raise RuntimeError("published probe coefficient digest changed")
    return _publish_immutable_bytes(target, encoded)


def evaluate_chain(
    chain_id: str,
    attentions: Sequence[Any],
    distances: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
    *,
    source_length: int,
) -> dict[str, Any]:
    scores = score_from_attentions(
        attentions,
        coefficients,
        intercept,
        residue_length=min(int(source_length), 510),
    )
    result = score_chain(
        chain_id,
        scores,
        np.asarray(distances)[:510, :510],
        source_length=source_length,
    )
    if result is None:
        raise ValueError(f"manifest chain is ineligible under the frozen contact rule: {chain_id}")
    return {
        "chain_id": result.chain_id,
        "source_length": result.source_length,
        "evaluated_length": result.evaluated_length,
        "eligible_pair_count": result.eligible_pair_count,
        "true_long_range_contacts": result.true_long_range_contacts,
        "precision_at_l": result.precision_at_l,
        "random_precision_at_l": result.random_precision_at_l,
    }
