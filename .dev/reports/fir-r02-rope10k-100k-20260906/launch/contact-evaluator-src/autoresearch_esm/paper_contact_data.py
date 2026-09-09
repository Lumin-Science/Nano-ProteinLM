"""Data and checkpoint contracts for paper-faithful ESMC contact evaluation.

This module does not download PDB data, run a model, or fit a contact probe.  It
provides the strict boundary objects those later stages consume:

* normalized chain records become a sequence and C-beta distance matrix;
* an ordered JSONL manifest binds every chain to source, sequence, and
  coordinate digests plus its frozen train/evaluation role; and
* immutable per-chain attention-feature checkpoints are bound to the exact
  model, weights, and dataset manifest; and
* immutable result checkpoints additionally bind the fitted-probe receipt.

The normalized chain schema is intentionally small.  A record has ``chain_id``
and an ordered ``residues`` list.  Each residue has a strictly increasing
integer ``residue_index``, a one-letter ``amino_acid``, and an ``atoms`` object
whose ``CA`` and ``CB`` values are either three finite coordinates or ``null``.
Glycine always selects CA; every other residue selects CB.  A missing selected
atom leaves the corresponding distance-matrix row and column as NaN.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np

SCHEMA_VERSION = 1
MANIFEST_ROLE = Literal["train", "eval"]
RESULT_STATUS = Literal["eligible", "excluded"]
ARTIFACT_ATTENTION = "esmc-paper-contact-attention-features-v1"
ARTIFACT_RESULT = "esmc-paper-contact-chain-result-v1"
MAXIMUM_RESIDUES = 510


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading the complete payload into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _nonempty(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _coordinate(value: object, *, field: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be three coordinates or null")
    if len(value) != 3:
        raise ValueError(f"{field} must contain exactly three coordinates")
    coordinates = tuple(float(component) for component in value)
    if not all(np.isfinite(coordinates)):
        raise ValueError(f"{field} coordinates must be finite; use null for missing atoms")
    # Canonicalize signed zero before hashing.
    return tuple(0.0 if component == 0.0 else component for component in coordinates)  # type: ignore[return-value]


@dataclass(frozen=True)
class NormalizedContactChain:
    """One ordered protein chain and its paper-style contact ground truth."""

    chain_id: str
    sequence: str
    residue_indices: tuple[int, ...]
    selected_atom_names: tuple[str, ...]
    selected_coordinates: tuple[tuple[float, float, float] | None, ...]
    cb_distances: np.ndarray
    sequence_sha256: str
    coordinate_sha256: str


def parse_normalized_chain(record: Mapping[str, object]) -> NormalizedContactChain:
    """Parse a strict normalized chain and construct its C-beta distance matrix.

    The function does not silently repair malformed or non-finite coordinates.
    Explicit ``null`` selected atoms are the only missing-atom representation.
    Their distance rows and columns remain NaN, allowing downstream filtering to
    distinguish unresolved pairs deterministically.
    """

    chain_id = _nonempty(record.get("chain_id"), field="chain_id")
    residues = record.get("residues")
    if isinstance(residues, (str, bytes)) or not isinstance(residues, Sequence):
        raise TypeError("residues must be a non-empty ordered list")
    if not residues:
        raise ValueError("residues must be non-empty")

    amino_acids: list[str] = []
    residue_indices: list[int] = []
    selected_atom_names: list[str] = []
    selected_coordinates: list[tuple[float, float, float] | None] = []
    previous_index: int | None = None
    for offset, residue in enumerate(residues):
        if not isinstance(residue, Mapping):
            raise TypeError(f"residue {offset} must be an object")
        residue_index = int(residue.get("residue_index", 0))
        if previous_index is not None and residue_index <= previous_index:
            raise ValueError("residue_index values must be strictly increasing")
        previous_index = residue_index
        amino_acid = str(residue.get("amino_acid", ""))
        if len(amino_acid) != 1 or not ("A" <= amino_acid <= "Z"):
            raise ValueError(f"residue {residue_index} amino_acid must be one uppercase letter")
        atoms = residue.get("atoms")
        if not isinstance(atoms, Mapping):
            raise TypeError(f"residue {residue_index} atoms must be an object")
        atom_name = "CA" if amino_acid == "G" else "CB"
        coordinate = _coordinate(
            atoms.get(atom_name),
            field=f"residue {residue_index} atom {atom_name}",
        )
        amino_acids.append(amino_acid)
        residue_indices.append(residue_index)
        selected_atom_names.append(atom_name)
        selected_coordinates.append(coordinate)

    sequence = "".join(amino_acids)
    declared_sequence = record.get("sequence")
    if declared_sequence is not None and str(declared_sequence) != sequence:
        raise ValueError("declared sequence differs from ordered residue amino acids")

    length = len(sequence)
    distances = np.full((length, length), np.nan, dtype=np.float64)
    present = [index for index, coordinate in enumerate(selected_coordinates) if coordinate]
    if present:
        coordinates = np.asarray(
            [selected_coordinates[index] for index in present], dtype=np.float64
        )
        deltas = coordinates[:, None, :] - coordinates[None, :, :]
        resolved_distances = np.sqrt(np.sum(deltas * deltas, axis=-1))
        indices = np.asarray(present, dtype=np.int64)
        distances[np.ix_(indices, indices)] = resolved_distances
    distances.setflags(write=False)

    sequence_digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    coordinate_contract = {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "residues": [
            {
                "residue_index": residue_index,
                "selected_atom": atom_name,
                "coordinate": coordinate,
            }
            for residue_index, atom_name, coordinate in zip(
                residue_indices, selected_atom_names, selected_coordinates
            )
        ],
    }
    coordinate_digest = hashlib.sha256(_canonical_json(coordinate_contract)).hexdigest()
    return NormalizedContactChain(
        chain_id=chain_id,
        sequence=sequence,
        residue_indices=tuple(residue_indices),
        selected_atom_names=tuple(selected_atom_names),
        selected_coordinates=tuple(selected_coordinates),
        cb_distances=distances,
        sequence_sha256=sequence_digest,
        coordinate_sha256=coordinate_digest,
    )


@dataclass(frozen=True)
class ContactManifestEntry:
    schema_version: int
    chain_id: str
    role: MANIFEST_ROLE
    source_payload_sha256: str
    sequence_sha256: str
    coordinate_sha256: str
    sequence_length: int


def manifest_entry(
    chain: NormalizedContactChain,
    *,
    source_payload_sha256: str,
    role: MANIFEST_ROLE,
) -> ContactManifestEntry:
    """Create one manifest row from a parsed chain and frozen source digest."""

    if role not in ("train", "eval"):
        raise ValueError("role must be 'train' or 'eval'")
    return ContactManifestEntry(
        schema_version=SCHEMA_VERSION,
        chain_id=chain.chain_id,
        role=role,
        source_payload_sha256=_validate_sha256(
            source_payload_sha256, field="source_payload_sha256"
        ),
        sequence_sha256=chain.sequence_sha256,
        coordinate_sha256=chain.coordinate_sha256,
        sequence_length=len(chain.sequence),
    )


def _validate_manifest_entry(value: ContactManifestEntry) -> ContactManifestEntry:
    if value.schema_version != SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {SCHEMA_VERSION}")
    _nonempty(value.chain_id, field="chain_id")
    if value.role not in ("train", "eval"):
        raise ValueError("manifest role must be 'train' or 'eval'")
    for field in ("source_payload_sha256", "sequence_sha256", "coordinate_sha256"):
        _validate_sha256(getattr(value, field), field=field)
    if int(value.sequence_length) <= 0:
        raise ValueError("sequence_length must be positive")
    return value


def _publish_immutable(
    path: Path,
    writer: Callable[[BinaryIO], None],
    *,
    verify_existing: Callable[[Path], None] | None = None,
) -> str:
    """Publish a complete file with an atomic hard link and never overwrite it."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if verify_existing is None:
            raise FileExistsError(f"immutable artifact already exists: {path}")
        verify_existing(path)
        return file_sha256(path)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if verify_existing is None:
                raise
            verify_existing(path)
        return file_sha256(path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_bytes(entries: Sequence[ContactManifestEntry]) -> bytes:
    if not entries:
        raise ValueError("contact manifest must contain at least one chain")
    seen: set[str] = set()
    lines: list[bytes] = []
    for entry in entries:
        _validate_manifest_entry(entry)
        if entry.chain_id in seen:
            raise ValueError(f"duplicate manifest chain_id: {entry.chain_id}")
        seen.add(entry.chain_id)
        lines.append(_canonical_json(asdict(entry)))
    return b"\n".join(lines) + b"\n"


@dataclass(frozen=True)
class ContactManifestReceipt:
    path: str
    manifest_sha256: str
    entry_count: int
    train_count: int
    eval_count: int
    ordered_chain_ids: tuple[str, ...]
    ordered_chain_ids_sha256: str


def verify_contact_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_chain_ids: Sequence[str] | None = None,
    source_payloads: Mapping[str, str | Path] | None = None,
) -> ContactManifestReceipt:
    """Verify canonical JSONL, exact ordering, and optional source payload bytes.

    ``source_payloads`` maps chain IDs to their immutable source files.  When it
    is supplied, every manifest chain must be present and its file is rehashed.
    """

    manifest_path = Path(path).resolve()
    payload = manifest_path.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed_sha256 != _validate_sha256(
        expected_sha256, field="expected_sha256"
    ):
        raise ValueError("contact manifest SHA-256 mismatch")
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("contact manifest must be non-empty canonical JSONL")

    entries: list[ContactManifestEntry] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        try:
            decoded = json.loads(raw_line)
            entry = ContactManifestEntry(**decoded)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid manifest row {line_number}") from error
        _validate_manifest_entry(entry)
        if raw_line != _canonical_json(asdict(entry)):
            raise ValueError(f"manifest row {line_number} is not canonical JSON")
        entries.append(entry)
    canonical = _manifest_bytes(entries)
    if canonical != payload:
        raise ValueError("contact manifest bytes are not canonical")

    chain_ids = tuple(entry.chain_id for entry in entries)
    if expected_chain_ids is not None and chain_ids != tuple(expected_chain_ids):
        raise ValueError("contact manifest ordered chain IDs differ")
    if source_payloads is not None:
        if set(source_payloads) != set(chain_ids):
            raise ValueError("source payload map must exactly cover manifest chain IDs")
        for entry in entries:
            if file_sha256(source_payloads[entry.chain_id]) != entry.source_payload_sha256:
                raise ValueError(f"source payload digest mismatch for {entry.chain_id}")

    ids_digest = hashlib.sha256(_canonical_json(list(chain_ids))).hexdigest()
    return ContactManifestReceipt(
        path=str(manifest_path),
        manifest_sha256=observed_sha256,
        entry_count=len(entries),
        train_count=sum(entry.role == "train" for entry in entries),
        eval_count=sum(entry.role == "eval" for entry in entries),
        ordered_chain_ids=chain_ids,
        ordered_chain_ids_sha256=ids_digest,
    )


def write_contact_manifest(
    entries: Sequence[ContactManifestEntry], path: str | Path
) -> ContactManifestReceipt:
    """Atomically publish an immutable canonical ordered JSONL manifest."""

    payload = _manifest_bytes(entries)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    manifest_path = Path(path)

    def verify_existing(existing: Path) -> None:
        receipt = verify_contact_manifest(existing)
        if receipt.manifest_sha256 != expected_sha256:
            raise FileExistsError(f"different immutable manifest already exists: {existing}")

    _publish_immutable(
        manifest_path,
        lambda handle: handle.write(payload),
        verify_existing=verify_existing,
    )
    return verify_contact_manifest(
        manifest_path,
        expected_sha256=expected_sha256,
        expected_chain_ids=[entry.chain_id for entry in entries],
    )


@dataclass(frozen=True)
class ContactFeatureBinding:
    """Pre-probe identity for extracted attention features.

    A feature artifact cannot bind a fitted probe receipt because that receipt
    is produced *from* the features.  Keeping this contract probe-free avoids
    a circular provenance dependency while still fixing the model, weights,
    and dataset used for extraction.
    """

    model_id: str
    model_revision: str
    weight_manifest_sha256: str
    dataset_manifest_sha256: str


def _validate_feature_binding(binding: ContactFeatureBinding) -> ContactFeatureBinding:
    model_id = _nonempty(binding.model_id, field="model_id")
    model_revision = _nonempty(binding.model_revision, field="model_revision")
    weight_manifest_sha256 = _validate_sha256(
        binding.weight_manifest_sha256, field="weight_manifest_sha256"
    )
    dataset_manifest_sha256 = _validate_sha256(
        binding.dataset_manifest_sha256, field="dataset_manifest_sha256"
    )
    # Normalize subclasses to the exact pre-probe schema before serialization.
    return ContactFeatureBinding(
        model_id=model_id,
        model_revision=model_revision,
        weight_manifest_sha256=weight_manifest_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )


@dataclass(frozen=True)
class ContactCheckpointBinding(ContactFeatureBinding):
    """Post-probe identity for evaluated per-chain results."""

    probe_receipt_sha256: str


def _validate_binding(binding: ContactCheckpointBinding) -> ContactCheckpointBinding:
    feature = _validate_feature_binding(binding)
    return ContactCheckpointBinding(
        model_id=feature.model_id,
        model_revision=feature.model_revision,
        weight_manifest_sha256=feature.weight_manifest_sha256,
        dataset_manifest_sha256=feature.dataset_manifest_sha256,
        probe_receipt_sha256=_validate_sha256(
            binding.probe_receipt_sha256, field="probe_receipt_sha256"
        ),
    )


@dataclass(frozen=True)
class ChainCheckpointIdentity:
    chain_id: str
    source_length: int
    evaluated_length: int
    sequence_sha256: str
    coordinate_sha256: str


def checkpoint_identity(
    chain: NormalizedContactChain,
    *,
    maximum_residues: int = MAXIMUM_RESIDUES,
) -> ChainCheckpointIdentity:
    maximum_residues = int(maximum_residues)
    if maximum_residues <= 0:
        raise ValueError("maximum_residues must be positive")
    return ChainCheckpointIdentity(
        chain_id=chain.chain_id,
        source_length=len(chain.sequence),
        evaluated_length=min(len(chain.sequence), maximum_residues),
        sequence_sha256=chain.sequence_sha256,
        coordinate_sha256=chain.coordinate_sha256,
    )


def _validate_identity(identity: ChainCheckpointIdentity) -> ChainCheckpointIdentity:
    _nonempty(identity.chain_id, field="chain_id")
    if identity.source_length <= 0:
        raise ValueError("source_length must be positive")
    if not 0 < identity.evaluated_length <= identity.source_length:
        raise ValueError("evaluated_length must be within the source chain")
    _validate_sha256(identity.sequence_sha256, field="sequence_sha256")
    _validate_sha256(identity.coordinate_sha256, field="coordinate_sha256")
    return identity


def _feature_sha256(features: np.ndarray) -> str:
    array = np.asarray(features, dtype=np.dtype("<f4"), order="C")
    descriptor = _canonical_json({"dtype": "float32", "shape": list(array.shape)})
    digest = hashlib.sha256(descriptor)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _attention_metadata(
    binding: ContactFeatureBinding,
    chain: ChainCheckpointIdentity,
    features: np.ndarray,
    *,
    transformation_id: str,
) -> dict[str, Any]:
    binding = _validate_feature_binding(binding)
    _validate_identity(chain)
    transformation_id = _nonempty(transformation_id, field="transformation_id")
    array = np.asarray(features, dtype=np.dtype("<f4"), order="C")
    expected = chain.evaluated_length
    if array.ndim != 3 or array.shape[0] != expected or array.shape[1] != expected:
        raise ValueError(
            f"attention features must have shape ({expected}, {expected}, feature_count)"
        )
    if array.shape[2] <= 0:
        raise ValueError("attention feature_count must be positive")
    if not np.isfinite(array).all():
        raise ValueError("attention features contain NaN or Inf")
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_ATTENTION,
        "binding": asdict(binding),
        "chain": asdict(chain),
        "transformation_id": transformation_id,
        "feature_dtype": "float32",
        "feature_shape": list(array.shape),
        "feature_sha256": _feature_sha256(array),
    }


@dataclass(frozen=True)
class AttentionCheckpointReceipt:
    path: str
    artifact_sha256: str
    feature_sha256: str
    feature_shape: tuple[int, int, int]
    binding: ContactFeatureBinding
    chain: ChainCheckpointIdentity
    transformation_id: str


def verify_attention_feature_checkpoint(
    path: str | Path,
    *,
    expected_binding: ContactFeatureBinding | None = None,
    expected_chain: ChainCheckpointIdentity | None = None,
) -> AttentionCheckpointReceipt:
    checkpoint_path = Path(path).resolve()
    try:
        with np.load(checkpoint_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            features = np.asarray(payload["features"], dtype=np.dtype("<f4"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid attention checkpoint: {checkpoint_path}") from error
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("attention checkpoint schema changed")
    if metadata.get("artifact_kind") != ARTIFACT_ATTENTION:
        raise ValueError("attention checkpoint artifact kind changed")
    binding = _validate_feature_binding(ContactFeatureBinding(**metadata["binding"]))
    chain = _validate_identity(ChainCheckpointIdentity(**metadata["chain"]))
    recomputed = _attention_metadata(
        binding,
        chain,
        features,
        transformation_id=metadata["transformation_id"],
    )
    if metadata != recomputed:
        raise ValueError("attention checkpoint metadata or feature digest mismatch")
    if expected_binding is not None and binding != _validate_feature_binding(expected_binding):
        raise ValueError("attention checkpoint model/weights/dataset binding differs")
    if expected_chain is not None and chain != _validate_identity(expected_chain):
        raise ValueError("attention checkpoint chain identity differs")
    return AttentionCheckpointReceipt(
        path=str(checkpoint_path),
        artifact_sha256=file_sha256(checkpoint_path),
        feature_sha256=metadata["feature_sha256"],
        feature_shape=tuple(metadata["feature_shape"]),
        binding=binding,
        chain=chain,
        transformation_id=metadata["transformation_id"],
    )


def write_attention_feature_checkpoint(
    path: str | Path,
    *,
    binding: ContactFeatureBinding,
    chain: ChainCheckpointIdentity,
    features: np.ndarray,
    transformation_id: str,
) -> AttentionCheckpointReceipt:
    """Atomically publish immutable float32 per-chain attention features."""

    array = np.asarray(features, dtype=np.dtype("<f4"), order="C")
    metadata = _attention_metadata(
        binding,
        chain,
        array,
        transformation_id=transformation_id,
    )
    checkpoint_path = Path(path)

    def verify_existing(existing: Path) -> None:
        receipt = verify_attention_feature_checkpoint(
            existing,
            expected_binding=binding,
            expected_chain=chain,
        )
        if receipt.feature_sha256 != metadata["feature_sha256"]:
            raise FileExistsError(f"different immutable attention checkpoint exists: {existing}")

    def writer(handle: BinaryIO) -> None:
        np.savez(
            handle,
            metadata_json=np.asarray(_canonical_json(metadata).decode("utf-8")),
            features=array,
        )

    _publish_immutable(checkpoint_path, writer, verify_existing=verify_existing)
    return verify_attention_feature_checkpoint(
        checkpoint_path,
        expected_binding=binding,
        expected_chain=chain,
    )


@dataclass(frozen=True)
class ContactChainResult:
    status: RESULT_STATUS
    eligible_pair_count: int
    true_long_range_contacts: int
    precision_at_l: float | None
    random_precision_at_l: float | None
    exclusion_reason: str | None = None


def _validate_result(
    result: ContactChainResult, chain: ChainCheckpointIdentity
) -> ContactChainResult:
    if result.status not in ("eligible", "excluded"):
        raise ValueError("result status must be 'eligible' or 'excluded'")
    if result.eligible_pair_count < 0 or result.true_long_range_contacts < 0:
        raise ValueError("contact counts cannot be negative")
    if result.true_long_range_contacts > result.eligible_pair_count:
        raise ValueError("true contacts cannot exceed eligible pairs")
    if result.status == "eligible":
        if result.eligible_pair_count < chain.evaluated_length:
            raise ValueError("eligible result has fewer than L resolved pairs")
        if result.true_long_range_contacts < chain.evaluated_length:
            raise ValueError("eligible result has fewer than L true contacts")
        if result.exclusion_reason is not None:
            raise ValueError("eligible result cannot have an exclusion reason")
        for field in ("precision_at_l", "random_precision_at_l"):
            value = getattr(result, field)
            if value is None or not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"eligible result {field} must be finite within [0, 1]")
    else:
        if result.precision_at_l is not None or result.random_precision_at_l is not None:
            raise ValueError("excluded result cannot contain precision values")
        _nonempty(result.exclusion_reason, field="exclusion_reason")
    return result


@dataclass(frozen=True)
class ResultCheckpointReceipt:
    path: str
    artifact_sha256: str
    attention_feature_sha256: str
    binding: ContactCheckpointBinding
    chain: ChainCheckpointIdentity
    result: ContactChainResult


def _result_payload(
    binding: ContactCheckpointBinding,
    chain: ChainCheckpointIdentity,
    attention_feature_sha256: str,
    result: ContactChainResult,
) -> dict[str, Any]:
    _validate_binding(binding)
    _validate_identity(chain)
    _validate_sha256(attention_feature_sha256, field="attention_feature_sha256")
    _validate_result(result, chain)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_RESULT,
        "binding": asdict(binding),
        "chain": asdict(chain),
        "attention_feature_sha256": attention_feature_sha256,
        "result": asdict(result),
    }


def verify_chain_result_checkpoint(
    path: str | Path,
    *,
    expected_binding: ContactCheckpointBinding | None = None,
    expected_chain: ChainCheckpointIdentity | None = None,
    expected_attention_feature_sha256: str | None = None,
) -> ResultCheckpointReceipt:
    checkpoint_path = Path(path).resolve()
    try:
        raw = checkpoint_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid result checkpoint: {checkpoint_path}") from error
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("result checkpoint schema changed")
    if payload.get("artifact_kind") != ARTIFACT_RESULT:
        raise ValueError("result checkpoint artifact kind changed")
    binding = _validate_binding(ContactCheckpointBinding(**payload["binding"]))
    chain = _validate_identity(ChainCheckpointIdentity(**payload["chain"]))
    result = ContactChainResult(**payload["result"])
    recomputed = _result_payload(
        binding,
        chain,
        payload["attention_feature_sha256"],
        result,
    )
    if raw != _canonical_json(recomputed) + b"\n":
        raise ValueError("result checkpoint is not canonical or its contract changed")
    if expected_binding is not None and binding != _validate_binding(expected_binding):
        raise ValueError("result checkpoint model/dataset/probe binding differs")
    if expected_chain is not None and chain != _validate_identity(expected_chain):
        raise ValueError("result checkpoint chain identity differs")
    if expected_attention_feature_sha256 is not None and payload[
        "attention_feature_sha256"
    ] != _validate_sha256(
        expected_attention_feature_sha256,
        field="expected_attention_feature_sha256",
    ):
        raise ValueError("result checkpoint attention feature binding differs")
    return ResultCheckpointReceipt(
        path=str(checkpoint_path),
        artifact_sha256=file_sha256(checkpoint_path),
        attention_feature_sha256=payload["attention_feature_sha256"],
        binding=binding,
        chain=chain,
        result=_validate_result(result, chain),
    )


def write_chain_result_checkpoint(
    path: str | Path,
    *,
    binding: ContactCheckpointBinding,
    chain: ChainCheckpointIdentity,
    attention_feature_sha256: str,
    result: ContactChainResult,
) -> ResultCheckpointReceipt:
    """Atomically publish an immutable eligible or excluded chain result."""

    payload = _result_payload(binding, chain, attention_feature_sha256, result)
    encoded = _canonical_json(payload) + b"\n"
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    checkpoint_path = Path(path)

    def verify_existing(existing: Path) -> None:
        receipt = verify_chain_result_checkpoint(
            existing,
            expected_binding=binding,
            expected_chain=chain,
            expected_attention_feature_sha256=attention_feature_sha256,
        )
        if receipt.artifact_sha256 != expected_sha256:
            raise FileExistsError(f"different immutable result checkpoint exists: {existing}")

    _publish_immutable(
        checkpoint_path,
        lambda handle: handle.write(encoded),
        verify_existing=verify_existing,
    )
    return verify_chain_result_checkpoint(
        checkpoint_path,
        expected_binding=binding,
        expected_chain=chain,
        expected_attention_feature_sha256=attention_feature_sha256,
    )
