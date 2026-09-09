"""Executable model/probe/runtime for the frozen ESMC paper-contact benchmark."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .paper_contact import EXPECTED_EVALUATION_CHAINS, PROTOCOL_ID, score_packet
from .paper_contact_data import (
    ContactChainResult,
    ContactCheckpointBinding,
    ContactFeatureBinding,
    ContactManifestEntry,
    checkpoint_identity,
    file_sha256,
    manifest_entry,
    parse_normalized_chain,
    verify_chain_result_checkpoint,
    verify_contact_manifest,
    write_chain_result_checkpoint,
)
from .paper_contact_model import (
    fit_logistic_probe,
    sampled_pair_feature_matrix,
    save_probe,
    score_and_digest_from_attentions,
)

RUNTIME_PROTOCOL_ID = "esmc-paper-contact-runtime-v1"
FEATURE_SCHEMA = "esmc-paper-contact-sampled-probe-features-v1"
EVAL_SHARD_SCHEMA = "esmc-paper-contact-eval-shard-v1"
DEFAULT_EVAL_SHARDS = 64
TRANSFORMATION_ID = "all_layer_all_head_symmetrize_only"
PAIR_SAMPLING_ID = "balanced-4096-per-class-seed-20260819"


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model_id: str
    revision: str
    weights_sha256: str
    hidden_size: int
    layers: int
    heads: int

    @property
    def channels(self) -> int:
        return self.layers * self.heads


MODEL_SPECS = {
    "300m": ModelSpec(
        "300m",
        "biohub/ESMC-300M",
        "a59b831785f907e96e6a246b1d142bfb76df31ee",
        "0772d8fe64bb25e14fe6f23b80e3c9a7d215d0da3c6cba5bd356d7c0e0bb22cc",
        960,
        30,
        15,
    ),
    "600m": ModelSpec(
        "600m",
        "biohub/ESMC-600M",
        "a7e82012c83126b9eedb055fea9fa84b6c02f094",
        "e4232c30fd35fe2f57051ec88a703996ac94520580b4b836894207a3d45d9ff8",
        1152,
        36,
        18,
    ),
    "6b": ModelSpec(
        "6b",
        "biohub/ESMC-6B",
        "45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a",
        "81dfcf064d3009893299773e0c6096de299c888938e8622545d0df5a66a6e9db",
        2560,
        80,
        40,
    ),
}


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_identical(path: Path, payload: bytes) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if file_sha256(path) != digest:
            raise FileExistsError(f"different immutable artifact already exists: {path}")
        return digest
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
            if file_sha256(path) != digest:
                raise FileExistsError(f"different immutable artifact already exists: {path}")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    payload = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"JSON digest mismatch: {path}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


class ContactDataset:
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        self.ready_path = self.root / "PDB_CONTACT_DATASET_READY.json"
        self.ready = _load_json(self.ready_path)
        if self.ready.get("event") != "paper_contact_dataset_ready":
            raise ValueError("contact dataset readiness event mismatch")
        manifest_path = self.root / str(self.ready["manifest_path"])
        self.manifest_receipt = verify_contact_manifest(
            manifest_path, expected_sha256=str(self.ready["manifest_sha256"])
        )
        self.entries: list[ContactManifestEntry] = []
        with manifest_path.open() as handle:
            for line in handle:
                self.entries.append(ContactManifestEntry(**json.loads(line)))
        inventory_path = self.root / "PAYLOAD_INVENTORY.json"
        if file_sha256(inventory_path) != self.ready["payload_inventory_sha256"]:
            raise ValueError("contact payload inventory digest mismatch")
        inventory = json.loads(inventory_path.read_bytes())
        if not isinstance(inventory, list):
            raise TypeError("contact payload inventory must be a list")
        self.payloads = {str(row["chain_id"]): row for row in inventory}
        if len(self.payloads) != len(inventory) or set(self.payloads) != {
            entry.chain_id for entry in self.entries
        }:
            raise ValueError("contact payload inventory does not exactly cover the manifest")
        self.entry_by_id = {entry.chain_id: entry for entry in self.entries}
        if len(self.entry_by_id) != len(self.entries):
            raise ValueError("contact manifest contains duplicate chain IDs")
        self.train_ids = [entry.chain_id for entry in self.entries if entry.role == "train"]
        self.eval_ids = [entry.chain_id for entry in self.entries if entry.role == "eval"]
        if len(self.train_ids) != 20 or len(self.eval_ids) != EXPECTED_EVALUATION_CHAINS:
            raise ValueError("contact dataset does not have exact 20/20,775 train/eval coverage")
        if self.train_ids[:16] != list(self.ready["probe_train_chain_ids"]):
            raise ValueError("probe training chain order changed")
        if self.train_ids[16:] != list(self.ready["probe_validation_chain_ids"]):
            raise ValueError("probe validation chain order changed")

    def load_payload(self, chain_id: str) -> tuple[dict[str, Any], Any]:
        row = self.payloads[chain_id]
        path = (self.root / str(row["path"])).resolve(strict=True)
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"payload escapes contact dataset root: {path}") from error
        digest = file_sha256(path)
        entry = self.entry_by_id[chain_id]
        if digest != row["sha256"] or digest != entry.source_payload_sha256:
            raise ValueError(f"contact source payload digest mismatch: {chain_id}")
        payload = _load_json(path)
        chain = parse_normalized_chain(payload)
        observed = manifest_entry(chain, source_payload_sha256=digest, role=entry.role)
        if observed != entry:
            raise ValueError(f"contact payload differs from manifest: {chain_id}")
        return payload, chain


def _array_sha256(array: np.ndarray, *, dtype: str) -> str:
    normalized = np.asarray(array, dtype=np.dtype(dtype), order="C")
    digest = hashlib.sha256(
        canonical_json({"dtype": np.dtype(dtype).name, "shape": list(normalized.shape)})
    )
    digest.update(memoryview(normalized).cast("B"))
    return digest.hexdigest()


def _feature_payload(
    features: np.ndarray,
    labels: np.ndarray,
    metadata: dict[str, Any],
) -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        metadata_json=np.asarray(canonical_json(metadata).decode("utf-8")),
        features=np.asarray(features, dtype=np.dtype("<f4"), order="C"),
        labels=np.asarray(labels, dtype=np.int8, order="C"),
    )
    return buffer.getvalue()


def _verify_feature(path: Path, binding: ContactFeatureBinding, chain_id: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as packet:
        metadata = json.loads(str(packet["metadata_json"].item()))
        features = np.asarray(packet["features"], dtype=np.dtype("<f4"))
        labels = np.asarray(packet["labels"], dtype=np.int8)
    if metadata.get("schema") != FEATURE_SCHEMA or metadata.get("chain_id") != chain_id:
        raise ValueError(f"probe feature schema/chain mismatch: {path}")
    if metadata.get("binding") != asdict(binding):
        raise ValueError(f"probe feature binding mismatch: {path}")
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.size:
        raise ValueError(f"probe feature shape mismatch: {path}")
    if metadata.get("feature_sha256") != _array_sha256(features, dtype="<f4"):
        raise ValueError(f"probe feature digest mismatch: {path}")
    if metadata.get("labels_sha256") != _array_sha256(labels, dtype="i1"):
        raise ValueError(f"probe labels digest mismatch: {path}")
    return {"metadata": metadata, "features": features, "labels": labels}


def _load_model(spec: ModelSpec):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    from .pcore_embed import _model_hidden_size, _snapshot_weights_hash

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("paper-contact model jobs require exactly one visible CUDA GPU")
    observed_weights, _snapshot = _snapshot_weights_hash(spec.model_id, spec.revision)
    if observed_weights != spec.weights_sha256:
        raise ValueError(
            f"model weight digest mismatch: expected={spec.weights_sha256}, "
            f"observed={observed_weights}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id, revision=spec.revision, trust_remote_code=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval().to(torch.device("cuda"))
    if _model_hidden_size(model) != spec.hidden_size:
        raise ValueError("model hidden-size contract changed")
    return model, tokenizer, torch.device("cuda")


def _infer_attentions(model: Any, tokenizer: Any, device: Any, sequence: str, spec: ModelSpec):
    import torch

    encoded = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    if int(encoded["input_ids"].shape[1]) != len(sequence) + 2:
        raise ValueError("paper-contact tokenizer did not produce exactly BOS+residues+EOS")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        try:
            output = model(
                **encoded,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
            )
        except TypeError:
            output = model(
                **encoded,
                output_attentions=True,
                output_hidden_states=False,
            )
    attentions = output.attentions
    if attentions is None or len(attentions) != spec.layers:
        raise ValueError("model did not return the frozen number of attention layers")
    first = attentions[0]
    if first.ndim != 4 or int(first.shape[1]) != spec.heads:
        raise ValueError("model did not return the frozen number of attention heads")
    return attentions


def extract_probe_features(dataset_root: Path, output_root: Path, spec: ModelSpec) -> dict[str, Any]:
    dataset = ContactDataset(dataset_root)
    binding = ContactFeatureBinding(
        model_id=spec.model_id,
        model_revision=spec.revision,
        weight_manifest_sha256=spec.weights_sha256,
        dataset_manifest_sha256=dataset.manifest_receipt.manifest_sha256,
    )
    output_root = output_root.resolve()
    feature_root = output_root / "features"
    complete = output_root / "PROBE_FEATURES_COMPLETE.json"
    valid_existing: dict[str, str] = {}
    for chain_id in dataset.train_ids:
        path = feature_root / f"{hashlib.sha256(chain_id.encode()).hexdigest()}.npz"
        if path.exists():
            packet = _verify_feature(path, binding, chain_id)
            if int(packet["metadata"]["channels"]) != spec.channels:
                raise ValueError(f"probe feature channel mismatch: {path}")
            valid_existing[chain_id] = file_sha256(path)
    if len(valid_existing) != len(dataset.train_ids):
        model, tokenizer, device = _load_model(spec)
        for chain_id in dataset.train_ids:
            path = feature_root / f"{hashlib.sha256(chain_id.encode()).hexdigest()}.npz"
            if chain_id in valid_existing:
                continue
            _payload, chain = dataset.load_payload(chain_id)
            attentions = _infer_attentions(model, tokenizer, device, chain.sequence, spec)
            features, labels, selection_sha256 = sampled_pair_feature_matrix(
                attentions,
                chain.cb_distances,
                chain_id=chain_id,
                seed=20_260_819,
                maximum_per_class=4_096,
            )
            if features.shape[1] != spec.channels:
                raise ValueError("probe feature channel count differs from model contract")
            metadata = {
                "schema": FEATURE_SCHEMA,
                "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
                "binding": asdict(binding),
                "chain_id": chain_id,
                "sequence_sha256": chain.sequence_sha256,
                "coordinate_sha256": chain.coordinate_sha256,
                "channels": spec.channels,
                "rows": int(features.shape[0]),
                "pair_sampling_id": PAIR_SAMPLING_ID,
                "pair_selection_sha256": selection_sha256,
                "feature_sha256": _array_sha256(features, dtype="<f4"),
                "labels_sha256": _array_sha256(labels, dtype="i1"),
            }
            _atomic_identical(path, _feature_payload(features, labels, metadata))
            _verify_feature(path, binding, chain_id)
            valid_existing[chain_id] = file_sha256(path)
    receipt = {
        "schema": "esmc-paper-contact-probe-features-complete-v1",
        "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
        "model": asdict(spec),
        "binding": asdict(binding),
        "train_chain_ids": dataset.train_ids,
        "feature_checkpoint_sha256s": valid_existing,
    }
    _atomic_identical(complete, canonical_json(receipt))
    return receipt


def fit_probe(dataset_root: Path, model_root: Path, spec: ModelSpec) -> dict[str, Any]:
    dataset = ContactDataset(dataset_root)
    feature_receipt_path = model_root / "PROBE_FEATURES_COMPLETE.json"
    feature_receipt = _load_json(feature_receipt_path)
    if feature_receipt.get("model") != asdict(spec):
        raise ValueError("probe feature model contract mismatch")
    binding = ContactFeatureBinding(**feature_receipt["binding"])
    train_features, train_labels, valid_features, valid_labels = [], [], [], []
    for index, chain_id in enumerate(dataset.train_ids):
        path = model_root / "features" / f"{hashlib.sha256(chain_id.encode()).hexdigest()}.npz"
        if file_sha256(path) != feature_receipt["feature_checkpoint_sha256s"][chain_id]:
            raise ValueError(f"probe feature artifact digest mismatch: {chain_id}")
        packet = _verify_feature(path, binding, chain_id)
        target_features = train_features if index < 16 else valid_features
        target_labels = train_labels if index < 16 else valid_labels
        target_features.append(packet["features"])
        target_labels.append(packet["labels"])
    coefficients, intercept, selected_c, trace = fit_logistic_probe(
        train_features, train_labels, valid_features, valid_labels, seed=20_260_819
    )
    if coefficients.size != spec.channels:
        raise ValueError("fitted probe channel count differs from model contract")
    receipt_sha256 = save_probe(
        model_root / "probe",
        coefficients,
        intercept,
        model_id=spec.model_id,
        model_revision=spec.revision,
        weight_manifest_sha256=spec.weights_sha256,
        dataset_manifest_sha256=dataset.manifest_receipt.manifest_sha256,
        train_chain_ids=dataset.train_ids[:16],
        validation_chain_ids=dataset.train_ids[16:],
        feature_checkpoint_sha256s=feature_receipt["feature_checkpoint_sha256s"],
        selected_C=selected_c,
        validation_trace=trace,
        seed=20_260_819,
    )
    return {"event": "paper_contact_probe_fitted", "receipt_sha256": receipt_sha256}


def _load_probe(model_root: Path, spec: ModelSpec, dataset: ContactDataset):
    receipt_path = model_root / "probe" / "PROBE_RECEIPT.json"
    receipt = _load_json(receipt_path)
    if (
        receipt.get("model_id") != spec.model_id
        or receipt.get("model_revision") != spec.revision
        or receipt.get("weight_manifest_sha256") != spec.weights_sha256
        or receipt.get("dataset_manifest_sha256") != dataset.manifest_receipt.manifest_sha256
        or int(receipt.get("channels", -1)) != spec.channels
    ):
        raise ValueError("fitted contact probe contract mismatch")
    coefficient_path = model_root / "probe" / "probe-coefficients.npz"
    if file_sha256(coefficient_path) != receipt["coefficients_sha256"]:
        raise ValueError("fitted contact probe coefficient digest mismatch")
    with np.load(coefficient_path, allow_pickle=False) as packet:
        coefficients = np.asarray(packet["coefficients"], dtype=np.float64)
        intercept = float(packet["intercept"])
    return receipt, file_sha256(receipt_path), coefficients, intercept


def evaluate_shard(
    dataset_root: Path,
    model_root: Path,
    spec: ModelSpec,
    *,
    shard: int,
    shards: int,
) -> dict[str, Any]:
    dataset = ContactDataset(dataset_root)
    if shard < 0 or shard >= shards:
        raise ValueError(f"shard must be in [0, {shards})")
    _probe, probe_sha256, coefficients, intercept = _load_probe(model_root, spec, dataset)
    binding = ContactCheckpointBinding(
        model_id=spec.model_id,
        model_revision=spec.revision,
        weight_manifest_sha256=spec.weights_sha256,
        dataset_manifest_sha256=dataset.manifest_receipt.manifest_sha256,
        probe_receipt_sha256=probe_sha256,
    )
    chain_ids = [value for index, value in enumerate(dataset.eval_ids) if index % shards == shard]
    result_root = model_root / "results" / f"shard-{shard:03d}"
    receipts: dict[str, str] = {}
    missing = []
    for chain_id in chain_ids:
        _payload, chain = dataset.load_payload(chain_id)
        identity = checkpoint_identity(chain)
        path = result_root / f"{hashlib.sha256(chain_id.encode()).hexdigest()}.json"
        if path.exists():
            verified = verify_chain_result_checkpoint(
                path, expected_binding=binding, expected_chain=identity
            )
            receipts[chain_id] = verified.artifact_sha256
        else:
            missing.append((chain_id, path, chain, identity))
    if missing:
        model, tokenizer, device = _load_model(spec)
        for chain_id, path, chain, identity in missing:
            payload = _load_json(
                (dataset.root / str(dataset.payloads[chain_id]["path"])).resolve(strict=True)
            )
            attentions = _infer_attentions(model, tokenizer, device, chain.sequence, spec)
            scores, attention_sha256 = score_and_digest_from_attentions(
                attentions,
                coefficients,
                intercept,
                residue_length=len(chain.sequence),
            )
            from .paper_contact import score_chain

            fused = score_chain(
                chain_id,
                scores,
                chain.cb_distances,
                source_length=int(payload["source_length"]),
            )
            if fused is None:
                raise ValueError("frozen manifest chain became ineligible during evaluation")
            result = ContactChainResult(
                status="eligible",
                eligible_pair_count=fused.eligible_pair_count,
                true_long_range_contacts=fused.true_long_range_contacts,
                precision_at_l=fused.precision_at_l,
                random_precision_at_l=fused.random_precision_at_l,
                exclusion_reason=None,
            )
            published = write_chain_result_checkpoint(
                path,
                binding=binding,
                chain=identity,
                attention_feature_sha256=attention_sha256,
                result=result,
            )
            receipts[chain_id] = published.artifact_sha256
    shard_receipt = {
        "schema": EVAL_SHARD_SCHEMA,
        "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
        "model": asdict(spec),
        "binding": asdict(binding),
        "shard": shard,
        "shards": shards,
        "chain_ids": chain_ids,
        "result_sha256s": receipts,
    }
    _atomic_identical(model_root / "results" / f"SHARD-{shard:03d}.json", canonical_json(shard_receipt))
    return shard_receipt


def reduce_model(dataset_root: Path, model_root: Path, spec: ModelSpec, *, shards: int) -> dict[str, Any]:
    dataset = ContactDataset(dataset_root)
    _probe, probe_sha256, _coefficients, _intercept = _load_probe(model_root, spec, dataset)
    binding = ContactCheckpointBinding(
        model_id=spec.model_id,
        model_revision=spec.revision,
        weight_manifest_sha256=spec.weights_sha256,
        dataset_manifest_sha256=dataset.manifest_receipt.manifest_sha256,
        probe_receipt_sha256=probe_sha256,
    )
    covered: list[str] = []
    for shard in range(shards):
        receipt = _load_json(model_root / "results" / f"SHARD-{shard:03d}.json")
        if receipt.get("schema") != EVAL_SHARD_SCHEMA or receipt.get("binding") != asdict(binding):
            raise ValueError(f"evaluation shard receipt contract mismatch: {shard}")
        expected_ids = [value for index, value in enumerate(dataset.eval_ids) if index % shards == shard]
        if receipt.get("chain_ids") != expected_ids:
            raise ValueError(f"evaluation shard chain coverage mismatch: {shard}")
        covered.extend(expected_ids)
    if sorted(covered) != sorted(dataset.eval_ids) or len(covered) != len(dataset.eval_ids):
        raise ValueError("evaluation shard coverage is not exact")

    rows = []
    for eval_index, chain_id in enumerate(dataset.eval_ids):
        payload, chain = dataset.load_payload(chain_id)
        identity = checkpoint_identity(chain)
        shard = eval_index % shards
        path = model_root / "results" / f"shard-{shard:03d}" / (
            hashlib.sha256(chain_id.encode()).hexdigest() + ".json"
        )
        receipt = verify_chain_result_checkpoint(
            path, expected_binding=binding, expected_chain=identity
        )
        if receipt.result.status != "eligible":
            raise ValueError(f"frozen evaluation chain became ineligible: {chain_id}")
        rows.append(
            {
                "chain_id": chain_id,
                "source_length": int(payload["source_length"]),
                "evaluated_length": min(int(payload["source_length"]), 510),
                "eligible_pair_count": receipt.result.eligible_pair_count,
                "true_long_range_contacts": receipt.result.true_long_range_contacts,
                "precision_at_l": receipt.result.precision_at_l,
                "random_precision_at_l": receipt.result.random_precision_at_l,
            }
        )
    packet = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "weights_sha256": spec.weights_sha256,
        "dataset_manifest_sha256": dataset.manifest_receipt.manifest_sha256,
        "probe_receipt_sha256": probe_sha256,
        "chains": rows,
    }
    report = score_packet(packet)
    packet_sha256 = _atomic_identical(model_root / "CONTACT_PACKET.json", canonical_json(packet))
    report_sha256 = _atomic_identical(model_root / "CONTACT_REPORT.json", canonical_json(report))
    complete = {
        "schema": "esmc-paper-contact-model-complete-v1",
        "event": "paper_contact_model_complete",
        "runtime_protocol_id": RUNTIME_PROTOCOL_ID,
        "model": asdict(spec),
        "packet_sha256": packet_sha256,
        "report_sha256": report_sha256,
        "point_estimate": report["metric"]["point_estimate"],
        "evaluation_chains": report["metric"]["evaluation_chains"],
    }
    _atomic_identical(model_root / "PAPER_CONTACT_COMPLETE.json", canonical_json(complete))
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m autoresearch_esm.paper_contact_runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("extract-probe", "fit-probe", "eval-shard", "reduce"):
        command = subparsers.add_parser(name)
        command.add_argument("--dataset-root", type=Path, required=True)
        command.add_argument("--model-root", type=Path, required=True)
        command.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
        if name == "eval-shard":
            command.add_argument("--shard", type=int, required=True)
            command.add_argument("--shards", type=int, default=DEFAULT_EVAL_SHARDS)
        elif name == "reduce":
            command.add_argument("--shards", type=int, default=DEFAULT_EVAL_SHARDS)
    args = parser.parse_args()
    spec = MODEL_SPECS[args.model]
    if args.command == "extract-probe":
        result = extract_probe_features(args.dataset_root, args.model_root, spec)
    elif args.command == "fit-probe":
        result = fit_probe(args.dataset_root, args.model_root, spec)
    elif args.command == "eval-shard":
        result = evaluate_shard(
            args.dataset_root,
            args.model_root,
            spec,
            shard=args.shard,
            shards=args.shards,
        )
    else:
        result = reduce_model(args.dataset_root, args.model_root, spec, shards=args.shards)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
