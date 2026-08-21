"""Checkpoint evaluation: held-out MLM, current P-CORE, and paper-style P@L."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import MixtureBatcher, file_sha256
from .model import ESMCConfig, ESMCForMaskedLM
from .tokenizer import ProteinTokenizer, mask_tokens


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
            embed_sequences,
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
    embedding_result = embed_sequences(
        model,
        ProteinTokenizer.esmc(),
        sequences,
        store,
        device=device,
        batch_residue_budget=batch_residues,
        include_residue=True,
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
        "embedding": embedding_result,
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
    parser.add_argument("--run-pcore", action="store_true")
    parser.add_argument("--run-contact", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    device = torch.device("cuda", 0)
    model, checkpoint_packet = load_checkpoint(args.checkpoint, device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_training_seconds": checkpoint_packet["training_seconds"],
        "validation_mlm": validation_mlm(
            model,
            data_root=args.data_root,
            device=device,
            context_length=args.validation_context,
            batch_size=args.validation_batch_size,
            batches=args.validation_batches,
            seed=20260821,
        ),
    }
    if args.run_contact:
        if args.external_src is None or args.contact_root is None:
            parser.error("--run-contact requires --external-src and --contact-root")
        report["contact"] = run_contact_lite(
            model,
            dataset_root=args.contact_root,
            external_src=args.external_src,
            device=device,
            evaluation_chains=args.contact_chains,
        )
    if args.run_pcore:
        if args.external_src is None or args.pcore_root is None:
            parser.error("--run-pcore requires --external-src and --pcore-root")
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
    report_path = args.output_root / "EVALUATION.json"
    temporary = report_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
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
