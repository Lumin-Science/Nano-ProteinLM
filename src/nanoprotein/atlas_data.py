"""Stream a pinned Atlas subset into mmap stores for an unscreened experiment.

Training requires explicit allow_unscreened_training_data: true. Validation
inputs remain unchanged. Lance is needed only for preparation, not training.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import struct
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .data import INDEX_DTYPE, SOURCES, TokenStore, file_sha256
from .tokenizer import CANONICAL_AAS, ProteinTokenizer

ATLAS_URI = "s3://esm-protein-atlas/v1/folds/folds_1B.lance"
NORMALIZE_TABLE = bytes(
    i if chr(i) in CANONICAL_AAS + "XBUZO" else ord("X") for i in range(256)
)
CANONICAL_BYTES = CANONICAL_AAS.encode("ascii")
INDEX_STRUCT = struct.Struct("<QI32s")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def normalize(sequence):
    sequence = sequence.upper().encode("ascii", errors="replace")
    sequence = sequence.translate(None, b" \t\n\r").rstrip(b"*").translate(NORMALIZE_TABLE)
    if not 60 <= len(sequence) <= 16384:
        return None
    if len(sequence.translate(None, CANONICAL_BYTES)) > len(sequence) * 0.2:
        return None
    return sequence


class StoreWriter:
    """Write packed residues and fixed-size index rows with bounded memory."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tokens = (self.root / "tokens.bin.partial").open("wb", buffering=8 << 20)
        self.index = (self.root / "index.raw.partial").open("wb", buffering=8 << 20)
        self.records = self.offset = 0
        tokenizer = ProteinTokenizer.esmc()
        self.lookup = bytes(
            tokenizer.token_to_id.get(chr(i), tokenizer.unk_id) for i in range(256)
        )

    def append(self, sequence, digest):
        self.append_encoded(sequence.translate(self.lookup), digest)

    def append_encoded(self, encoded, digest):
        self.tokens.write(encoded)
        self.index.write(INDEX_STRUCT.pack(self.offset, len(encoded), digest))
        self.offset += len(encoded)
        self.records += 1

    def finish(self):
        for handle in (self.tokens, self.index):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        if not self.records:
            raise ValueError("empty training partition")
        raw = np.memmap(self.root / "index.raw.partial", dtype=INDEX_DTYPE, mode="r")
        output = np.lib.format.open_memmap(
            self.root / "index.npy.partial", mode="w+", dtype=INDEX_DTYPE, shape=(self.records,)
        )
        for start in range(0, self.records, 1_000_000):
            output[start : start + 1_000_000] = raw[start : start + 1_000_000]
        output.flush()
        del raw, output
        (self.root / "tokens.bin.partial").replace(self.root / "tokens.bin")
        (self.root / "index.npy.partial").replace(self.root / "index.npy")
        (self.root / "index.raw.partial").unlink()
        return dict(
            records=self.records,
            residues=self.offset,
            tokens_sha256=file_sha256(self.root / "tokens.bin"),
            index_sha256=file_sha256(self.root / "index.npy"),
        )


def prepare_partition(ds, part, root):
    work = Path(root) / "partitions" / f"{part['part']:05d}"
    work.mkdir(parents=True, exist_ok=True)
    done = work / "PREPARED.json"
    if done.exists():
        receipt = json.loads(done.read_text())
        if receipt["fragment_ids"] != part["fragment_ids"]:
            raise ValueError("cached partition identity differs")
        for name, field in (("tokens.bin", "tokens_sha256"), ("index.npy", "index_sha256")):
            if file_sha256(work / "store" / name) != receipt["store"][field]:
                raise ValueError("cached partition bytes differ")
        return receipt
    started = time.monotonic()
    counts, seen = Counter(), set()
    writer = StoreWriter(work / "store")
    fragments = [ds.get_fragment(i) for i in part["fragment_ids"]]
    batches = ds.scanner(
        columns=["sequence"],
        fragments=fragments,
        batch_size=65536,
        batch_readahead=2,
        fragment_readahead=4,
    ).to_batches()
    for batch in batches:
        for raw in batch.column(0).to_pylist():
            counts["downloaded"] += 1
            sequence = normalize(raw)
            if sequence is None:
                counts["quality_excluded"] += 1
                continue
            digest = hashlib.sha256(sequence).digest()
            if digest in seen:
                counts["local_duplicates"] += 1
                continue
            seen.add(digest)
            writer.append(sequence, digest)
    receipt = dict(
        status="prepared_unscreened",
        part=part["part"],
        fragment_ids=part["fragment_ids"],
        counts=counts,
        store=writer.finish(),
        seconds=time.monotonic() - started,
    )
    save(done, receipt)
    print(json.dumps(dict(event="partition_prepared", **receipt)), flush=True)
    return receipt


def validate_atlas_manifest(root, manifest=None):
    root = Path(root)
    manifest = manifest or json.loads((root / "manifest.json").read_text())
    receipt = json.loads((root / "ATLAS_PREPARATION.json").read_text())
    proof = json.loads((root / "CORPUS_VERIFICATION.json").read_text())
    valid = (
        manifest.get("protocol") == "atlas-clustered-subset-mmap-v1"
        and set(manifest.get("sources", {})) == {*SOURCES, "esm_atlas"}
        and manifest.get("decontamination", {}).get("homology_exclusion") is False
        and manifest["decontamination"].get("status") == "not_performed"
        and receipt.get("status") == "prepared_unscreened"
        and receipt.get("global_exact_duplicates") == 0
        and receipt.get("training_store") == manifest["sources"]["esm_atlas"]["train"]
        and proof.get("status") == "integrity_verified_unscreened"
        and proof.get("manifest_sha256") == file_sha256(root / "manifest.json")
        and proof.get("preparation_sha256") == file_sha256(root / "ATLAS_PREPARATION.json")
        and proof.get("evaluation_contamination_checked") is False
    )
    if not valid:
        raise ValueError("Atlas preparation identity or unscreened status is invalid")
    return manifest


def finalize(root, parts, validation_root, plan):
    root, validation_root = Path(root), Path(validation_root)
    output = root / "data"
    old = json.loads((validation_root / "manifest.json").read_text())
    stores = [
        TokenStore.open(root / "partitions" / f"{p['part']:05d}" / "store") for p in parts
    ]
    all_digests = np.concatenate([s.index["digest"] for s in stores])
    _, unique_indices = np.unique(all_digests, return_index=True)
    keep = np.zeros(len(all_digests), dtype=np.bool_)
    keep[unique_indices] = True
    duplicate_count = len(all_digests) - len(unique_indices)
    del all_digests, unique_indices
    print(
        json.dumps(
            dict(
                event="assembling_mmap",
                records=int(keep.sum()),
                duplicates_removed=duplicate_count,
            )
        ),
        flush=True,
    )
    writer = StoreWriter(output / "esm_atlas" / "train")
    cursor = 0
    for store in stores:
        selected = keep[cursor : cursor + store.index.size]
        if selected.all():
            base = writer.offset
            for start in range(0, store.tokens.size, 8 << 20):
                writer.tokens.write(store.tokens[start : start + (8 << 20)].tobytes())
            for start in range(0, store.index.size, 100000):
                block = np.array(store.index[start : start + 100000])
                block["offset"] += base
                writer.index.write(block.tobytes())
            writer.offset += store.tokens.size
            writer.records += store.index.size
        else:
            for row in np.flatnonzero(selected):
                record = store.index[row]
                writer.append_encoded(
                    store.sequence(int(row)).tobytes(), bytes(record["digest"]).ljust(32, b"\0")
                )
        cursor += store.index.size
    training = writer.finish()
    sources = {"esm_atlas": {"train": training}}
    for source in SOURCES:
        destination = output / source / "validation"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("tokens.bin", "index.npy"):
            shutil.copy2(validation_root / source / "validation" / name, destination / name)
            expected = old["sources"][source]["validation"][
                "tokens_sha256" if name == "tokens.bin" else "index_sha256"
            ]
            if file_sha256(destination / name) != expected:
                raise ValueError("fixed validation copy differs")
        sources[source] = {"validation": old["sources"][source]["validation"]}
    receipt = dict(
        protocol="atlas-clustered-subset-preparation-v1",
        status="prepared_unscreened",
        source=plan,
        partitions=parts,
        training_store=training,
        cross_partition_duplicates_removed=duplicate_count,
        global_exact_duplicates=0,
        evaluation_contamination_checked=False,
        validation_unchanged=True,
        validation_parent_manifest_sha256=file_sha256(validation_root / "manifest.json"),
    )
    save(output / "ATLAS_PREPARATION.json", receipt)
    manifest = dict(
        schema_version=1,
        protocol="atlas-clustered-subset-mmap-v1",
        sources=sources,
        sampling_unit="seeded subset of Atlas 70-percent cluster representatives",
        decontamination=dict(
            homology_exclusion=False,
            status="not_performed",
            reason="Explicit user request to skip contamination screening for this experiment",
        ),
    )
    save(output / "manifest.json", manifest)
    save(
        output / "CORPUS_VERIFICATION.json",
        dict(
            status="integrity_verified_unscreened",
            manifest_sha256=file_sha256(output / "manifest.json"),
            preparation_sha256=file_sha256(output / "ATLAS_PREPARATION.json"),
            evaluation_contamination_checked=False,
            global_exact_duplicates=0,
            validation_unchanged=True,
        ),
    )
    validate_atlas_manifest(output, manifest)
    save(
        root / "DATA_READY.json",
        dict(
            status="prepared_unscreened",
            data_root=str(output),
            records=training["records"],
            residues=training["residues"],
            manifest_sha256=file_sha256(output / "manifest.json"),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--records", type=int, default=120_000_000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--partitions",
        type=int,
        help="Prepare only the first N planned partitions for qualification",
    )
    args = parser.parse_args()
    if min(args.records, args.workers) <= 0:
        raise ValueError("preparation budgets must be positive")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    import lance

    ds = lance.dataset(
        ATLAS_URI,
        version=3,
        storage_options={"aws_skip_signature": "true", "region": "us-west-2"},
    )
    fragments = ds.get_fragments()
    order = np.random.default_rng(args.seed).permutation(len(fragments))
    selected, records = [], 0
    for i in order:
        fragment = fragments[int(i)]
        selected.append(fragment.fragment_id)
        records += fragment.count_rows()
        if records >= args.records:
            break
    parts = [
        dict(part=i // 100, fragment_ids=selected[i : i + 100])
        for i in range(0, len(selected), 100)
    ]
    plan = dict(
        uri=ATLAS_URI,
        version=3,
        upstream_records=ds.count_rows(),
        seed=args.seed,
        requested_records=args.records,
        selected_records=records,
        selected_fragments=selected,
        partitions=parts,
        confidence_filter=False,
        contamination_screen=False,
        selection="seeded whole-fragment permutation, without confidence selection",
    )
    path = root / "DOWNLOAD_PLAN.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("refusing to change an existing download plan")
    save(path, plan)
    subset = parts[: args.partitions] if args.partitions else parts
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare_partition, ds, part, root) for part in subset]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
            save(
                root / "PREPARATION.json",
                dict(
                    completed_partitions=len(completed),
                    planned_partitions=len(subset),
                    prepared_records=sum(p["store"]["records"] for p in completed),
                ),
            )
    completed.sort(key=lambda p: p["part"])
    finalize(root, completed, args.validation_root, plan)


if __name__ == "__main__":
    main()
