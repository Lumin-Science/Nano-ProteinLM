#!/usr/bin/env python3
"""Reproduce the complete open-protein corpus and its release shards.

This deliberately lives in ``dev/``: it is the annotated, heavyweight build
pipeline, not code imported by model training.  Every subcommand is restartable
and writes a content-addressed JSON receipt.  The intended host has 64 CPU cores,
MMseqs2 v18 or newer, enough RAM for one of 256 hash partitions, and roughly
3 TB of temporary disk if rebuilding all sources from their raw distributions.

Pipeline, in order
------------------
1. ``download`` pins the official UniRef90 and MGnify snapshots and every OMG
   Hugging Face Parquet object.  Only OMG is natively a Hugging Face source.
2. ``normalize`` canonicalizes records, enforces source separation and quality
   gates, and writes 256 SHA-256 partitions.
3. ``deduplicate`` collapses exact sequences globally while retaining source
   membership, then emits a sorted FASTA view for each source arm.
4. ``cluster`` runs source-specific MMseqs2 Linclust at 70% identity / 80%
   shorter-sequence coverage.
5. ``evaluation-union`` freezes P@L, legacy P-CORE, and every Q9 P-CORE sequence,
   including blocked CAFA, MegaScale wild types, and both PRING partners.
6. ``delta-screen`` searches only Q9 queries absent from the already verified
   legacy query set against the *complete* representative databases.
7. ``finalize-screen`` validates every alignment and unions it with the verified
   legacy exclusion set.  This is mathematically equivalent to rerunning the
   union, while preserving the expensive parent receipt.
8. ``shard`` applies exact + homology exclusion, length gates and a globally
   disjoint validation set, then writes deterministic SHA-ordered Parquet shards.
9. ``verify-release`` independently re-reads every shard and freezes the release
   manifest consumed by ``scripts/download_data.py``.

Examples are in ``dev/data/DATA_PROCESSING_REPORT.md``.  Run this file only via
the repository's uv environment: ``uv run --frozen python ...``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

import pyarrow as pa
import pyarrow.parquet as pq

SOURCES = ("uniref90", "mgnify", "omg_img")
SOURCE_BITS = {source: 1 << index for index, source in enumerate(SOURCES)}
CANONICAL_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")
EXTENDED_AAS = CANONICAL_AAS | frozenset("BXZJUO")
DEFAULT_PARTITIONS = 256
DEFAULT_SHARD_RESIDUES = 256 * 1024 * 1024
RELEASE_PROTOCOL = "protein-corpus-parquet-shards-v1"
SCREEN_PROTOCOL = "mmseqs2-evaluation-homology-exclusion-v2"
LEGACY_SCREEN_PROTOCOL = "mmseqs2-evaluation-homology-exclusion-v1"
MMSEQS_THRESHOLDS = {
    "minimum_sequence_identity": 0.30,
    "minimum_query_coverage": 0.80,
    "minimum_target_coverage": 0.80,
    "coverage_mode": 0,
    "maximum_sequences_per_query": 1_000_000,
}
PRIMARY_SOURCES = {
    "uniref90": {
        "url": (
            "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/"
            "release-2023_02/uniref/uniref2023_02.tar.gz"
        ),
        "bytes": 211_819_312_677,
        "md5": "353681f464572bb199fa032f714d4669",
        "relative": "raw/uniref90_2023_02/uniref2023_02.tar.gz",
    },
    "mgnify": {
        "url": (
            "https://ftp.ebi.ac.uk/pub/databases/metagenomics/peptide_database/"
            "2023_02/mgy_clusters.fa.gz"
        ),
        "bytes": 83_473_342_442,
        "md5": "332d36d2a943bdb769237a03e050ed03",
        "relative": "raw/mgnify_2023_02/mgy_clusters.fa.gz",
    },
}
OMG_REPO_ID = "tattabio/OMG"
OMG_MGNIFY_RE = re.compile(r"(?:ERZ|ERR|ERS|ERP|MGY)[A-Z0-9_.-]*$", re.I)
OMG_IMG_RE = re.compile(r"(?:\d{7,}|(?:Ga|IMG|JGI)[A-Z0-9_.-]+)$", re.I)
Q9_TASKS = {
    "cafa5_mf_nk30_hard": ("sequence",),
    "caid3_disorder_pdb": ("sequence",),
    "cath44_remote_retrieval": ("sequence",),
    "flip2_shift": ("sequence",),
    "megascale_family30_ddg": ("sequence", "wildtype_sequence"),
    "pring_human_c3_30": ("sequence_a", "sequence_b"),
}


def canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(canonical_json(value))
    partial.replace(path)


def file_hash(path: Path, algorithm: str = "sha256", block_size: int = 8 << 20) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def valid_digest(value: str) -> str:
    value = value.removeprefix("sha256_")
    if len(value) != 64:
        raise ValueError(f"invalid SHA-256 identifier: {value!r}")
    bytes.fromhex(value)
    return value


def write_fasta(handle: IO[str], digest: str, sequence: str) -> None:
    handle.write(f">sha256_{digest}\n")
    for start in range(0, len(sequence), 80):
        handle.write(sequence[start : start + 80] + "\n")


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    header: str | None = None
    parts: list[str] = []
    with opener(path, "rt", encoding="ascii", errors="strict") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header = line[1:].split()[0]
                parts = []
            elif header is None:
                raise ValueError(f"sequence before header at {path}:{line_number}")
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def canonicalize(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("ascii", errors="ignore")
    sequence = "".join(str(raw).split()).upper().rstrip("*")
    return "".join(residue if residue in EXTENDED_AAS else "X" for residue in sequence)


def sanitized_id(raw: object) -> str:
    return " ".join(str(raw).replace("\t", " ").replace("\n", " ").split())


def omg_accession_class(identifier: object) -> str:
    accession = str(identifier).split("|", 1)[0]
    if OMG_MGNIFY_RE.fullmatch(accession):
        return "mgnify"
    if OMG_IMG_RE.fullmatch(accession):
        return "img"
    return "unknown"


@dataclass
class CleanStats:
    source: str
    input_records: int = 0
    accepted: int = 0
    accepted_residues: int = 0
    rejected_too_short: int = 0
    rejected_ambiguous: int = 0
    rejected_wrong_source: int = 0
    rejected_unknown_source: int = 0


class PartitionWriter:
    def __init__(self, root: Path, partitions: int) -> None:
        if partitions < 1 or partitions & (partitions - 1):
            raise ValueError("partitions must be a positive power of two")
        self.root = root
        self.partitions = partitions
        self.prefix_digits = max(1, (partitions.bit_length() - 1 + 3) // 4)
        root.mkdir(parents=True, exist_ok=False)
        self.stack = ExitStack()
        self.handles = [
            self.stack.enter_context(
                (root / f"part-{part:04d}.tsv").open("w", buffering=4 << 20)
            )
            for part in range(partitions)
        ]

    def write(self, digest: str, source: str, identifier: str, sequence: str) -> None:
        # Prefix buckets concatenate into global lexical SHA-256 order.
        partition = int(digest[: self.prefix_digits], 16) % self.partitions
        self.handles[partition].write(f"{digest}\t{source}\t{identifier}\t{sequence}\n")

    def close(self) -> None:
        self.stack.close()


def _download_with_curl(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "8",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    partial.replace(destination)


def download_raw(data_root: Path, omg_manifest: Path) -> dict[str, Any]:
    """Fetch immutable upstream objects and verify size plus content hashes."""

    artifacts: list[dict[str, Any]] = []
    for source, spec in PRIMARY_SOURCES.items():
        target = data_root / str(spec["relative"])
        if not target.exists():
            _download_with_curl(str(spec["url"]), target)
        if target.stat().st_size != spec["bytes"] or file_hash(target, "md5") != spec["md5"]:
            raise ValueError(f"{source} raw object fails its pinned size/MD5")
        artifacts.append({"source": source, "path": str(target), "sha256": file_hash(target)})

    from huggingface_hub import hf_hub_download

    with omg_manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 959:
        raise ValueError(f"expected 959 pinned OMG shards, found {len(rows)}")
    for row in rows:
        local = Path(
            hf_hub_download(
                repo_id=OMG_REPO_ID,
                repo_type="dataset",
                filename=row["path"],
                local_dir=data_root / "raw/omg",
            )
        )
        if local.stat().st_size != int(row["bytes"]) or file_hash(local) != row["sha256"]:
            raise ValueError(f"OMG object fails its pin: {row['path']}")
        artifacts.append({"source": "omg", "path": str(local), "sha256": row["sha256"]})
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "protocol": "open-protein-raw-download-v1",
        "omg_manifest_sha256": file_hash(omg_manifest),
        "artifacts": artifacts,
    }
    atomic_json(data_root / "RAW_DOWNLOAD_VERIFIED.json", receipt)
    return receipt


@contextmanager
def _uniref_xml_handle(path: Path) -> Iterator[IO[bytes]]:
    """Open release tar -> uniref90.tar -> uniref90.xml.gz without extraction."""

    with tarfile.open(path, "r:gz") as outer:
        member = next(item for item in outer if item.name.endswith("uniref90.tar"))
        nested = outer.extractfile(member)
        if nested is None:
            raise ValueError("could not read uniref90.tar member")
        with tarfile.open(fileobj=nested, mode="r|") as inner:
            xml_member = next(item for item in inner if item.name.endswith("uniref90.xml.gz"))
            compressed = inner.extractfile(xml_member)
            if compressed is None:
                raise ValueError("could not read uniref90.xml.gz member")
            with gzip.GzipFile(fileobj=compressed) as xml:
                yield xml


def _xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def iter_uniref(path: Path) -> Iterator[tuple[str, str]]:
    with _uniref_xml_handle(path) as handle:
        for _event, element in ET.iterparse(handle, events=("end",)):
            if _xml_name(element.tag) != "entry":
                continue
            identifier = element.attrib.get("id", "")
            sequence = next(
                (
                    child.text or ""
                    for child in element.iter()
                    if _xml_name(child.tag) == "sequence"
                ),
                "",
            )
            yield identifier, sequence
            element.clear()


def iter_omg(path: Path) -> Iterator[tuple[str, str]]:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    id_column = next(
        (name for name in ("CDS_ids", "cds_ids", "protein_ids") if name in names),
        None,
    )
    seq_column = next(
        (name for name in ("CDS_seqs", "cds_seqs", "protein_sequences") if name in names),
        None,
    )
    if id_column is None or seq_column is None:
        raise ValueError(f"missing OMG CDS ID/sequence columns in {path}: {sorted(names)}")
    for batch in parquet.iter_batches(columns=[id_column, seq_column], batch_size=2048):
        rows = batch.to_pydict()
        for identifiers, sequences in zip(rows[id_column], rows[seq_column], strict=True):
            for identifier, sequence in zip(identifiers or [], sequences or [], strict=True):
                yield str(identifier), str(sequence)


def normalize_source(
    *, source: str, inputs: Sequence[Path], output: Path, partitions: int
) -> dict[str, Any]:
    """Canonicalize and filter one source into deterministic hash partitions."""

    if source not in SOURCES:
        raise ValueError(source)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stats = CleanStats(source)
    writer = PartitionWriter(output, partitions)
    started = time.monotonic()
    try:
        if source == "uniref90":
            iterator: Iterable[tuple[str, str]] = iter_uniref(inputs[0])
        elif source == "mgnify":
            iterator = (row for path in inputs for row in iter_fasta(path))
        else:
            iterator = (row for path in inputs for row in iter_omg(path))
        for identifier, raw_sequence in iterator:
            stats.input_records += 1
            if source == "omg_img":
                origin = omg_accession_class(identifier)
                if origin == "mgnify":
                    stats.rejected_wrong_source += 1
                    continue
                if origin == "unknown":
                    stats.rejected_unknown_source += 1
                    continue
            sequence = canonicalize(raw_sequence)
            if len(sequence) < 60:
                stats.rejected_too_short += 1
                continue
            ambiguous = sum(residue not in CANONICAL_AAS for residue in sequence)
            if ambiguous / len(sequence) > 0.20:
                stats.rejected_ambiguous += 1
                continue
            digest = sequence_hash(sequence)
            writer.write(digest, source, sanitized_id(identifier), sequence)
            stats.accepted += 1
            stats.accepted_residues += len(sequence)
    finally:
        writer.close()
    accounted = sum(
        getattr(stats, field)
        for field in (
            "accepted",
            "rejected_too_short",
            "rejected_ambiguous",
            "rejected_wrong_source",
            "rejected_unknown_source",
        )
    )
    if accounted != stats.input_records:
        raise AssertionError("normalization accounting is incomplete")
    if source == "omg_img" and stats.rejected_unknown_source:
        raise ValueError(
            f"OMG contains {stats.rejected_unknown_source:,} unknown source accessions"
        )
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "protocol": "open-protein-normalization-v1",
        "policy": {"minimum_length": 60, "maximum_ambiguous_fraction": 0.20},
        "partitions": partitions,
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for path in inputs
        ],
        "stats": asdict(stats),
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output / "normalization_report.json", receipt)
    return receipt


def deduplicate(inputs: Sequence[Path], output: Path, partitions: int) -> dict[str, Any]:
    """Globally collapse exact strings but preserve every source membership."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    fasta_roots = {source: output / "fasta" / source for source in SOURCES}
    membership_root = output / "membership"
    for path in (*fasta_roots.values(), membership_root):
        path.mkdir(parents=True, exist_ok=True)
    source_memberships: Counter[str] = Counter()
    source_unique: Counter[str] = Counter()
    duplicates = 0
    cross_source = 0
    total_unique = 0
    artifacts: dict[str, str] = {}
    schema = pa.schema(
        [("sha256", pa.string()), ("source", pa.string()), ("source_id", pa.string())]
    )
    for partition in range(partitions):
        records: dict[str, tuple[str, int, int]] = {}
        memberships: list[dict[str, str]] = []
        for root in inputs:
            path = root / f"part-{partition:04d}.tsv"
            with path.open() as handle:
                for line_number, line in enumerate(handle, start=1):
                    digest, source, source_id, sequence = line.rstrip("\n").split("\t", 3)
                    if sequence_hash(sequence) != digest:
                        raise ValueError(f"digest mismatch at {path}:{line_number}")
                    source_memberships[source] += 1
                    memberships.append(
                        {"sha256": digest, "source": source, "source_id": source_id}
                    )
                    previous = records.get(digest)
                    bit = SOURCE_BITS[source]
                    if previous is None:
                        records[digest] = (sequence, bit, 1)
                    else:
                        if previous[0] != sequence:
                            raise ValueError(f"SHA-256 collision for {digest}")
                        records[digest] = (sequence, previous[1] | bit, previous[2] + 1)
                        duplicates += 1
        member_path = membership_root / f"part-{partition:04d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(memberships, schema=schema), member_path, compression="zstd"
        )
        artifacts[str(member_path.relative_to(output))] = file_hash(member_path)
        handles = {
            source: (fasta_roots[source] / f"part-{partition:04d}.fasta").open("w")
            for source in SOURCES
        }
        try:
            for digest in sorted(records):
                sequence, source_mask, _membership_count = records[digest]
                present = [source for source in SOURCES if source_mask & SOURCE_BITS[source]]
                cross_source += int(len(present) > 1)
                total_unique += 1
                for source in present:
                    write_fasta(handles[source], digest, sequence)
                    source_unique[source] += 1
        finally:
            for handle in handles.values():
                handle.close()
        for source in SOURCES:
            path = fasta_roots[source] / f"part-{partition:04d}.fasta"
            artifacts[str(path.relative_to(output))] = file_hash(path)
    report = {
        "schema_version": 1,
        "status": "verified",
        "protocol": "global-exact-protein-deduplication-v1",
        "partitions": partitions,
        "total_memberships": sum(source_memberships.values()),
        "total_unique_sequences": total_unique,
        "exact_duplicate_memberships": duplicates,
        "cross_source_unique_sequences": cross_source,
        "source_memberships": dict(source_memberships),
        "source_unique_presence": dict(source_unique),
        "artifact_sha256": artifacts,
    }
    atomic_json(output / "deduplication_report.json", report)
    return report


def run_cluster(
    *, dedup_root: Path, output: Path, source: str, mmseqs: Path, threads: int
) -> dict[str, Any]:
    """Run the frozen source-specific 70%-identity Linclust recipe."""

    fasta = sorted((dedup_root / "fasta" / source).glob("part-*.fasta"))
    if not fasta:
        raise FileNotFoundError(f"no deduplicated FASTAs for {source}")
    output.mkdir(parents=True, exist_ok=False)
    db = output / "db"
    temporary = output / "tmp"
    db.mkdir()
    temporary.mkdir()
    subprocess.run(
        [str(mmseqs), "createdb", *map(str, fasta), str(db / "sequences")], check=True
    )
    command = [
        str(mmseqs),
        "linclust",
        str(db / "sequences"),
        str(db / "clusters"),
        str(temporary),
        "--min-seq-id",
        "0.70",
        "-c",
        "0.80",
        "--cov-mode",
        "1",
        "--cluster-mode",
        "2",
        "--threads",
        str(threads),
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [
            str(mmseqs),
            "createtsv",
            str(db / "sequences"),
            str(db / "sequences"),
            str(db / "clusters"),
            str(output / "clusters.tsv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(mmseqs),
            "result2repseq",
            str(db / "sequences"),
            str(db / "clusters"),
            str(db / "representatives"),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(mmseqs),
            "result2flat",
            str(db / "sequences"),
            str(db / "sequences"),
            str(db / "representatives"),
            str(output / "representatives.fasta"),
            "--use-fasta-header",
        ],
        check=True,
    )
    clusters = sum(1 for _ in iter_fasta(output / "representatives.fasta"))
    report = {
        "schema_version": 1,
        "status": "verified",
        "protocol": "source-linclust-70pct-v1",
        "source": source,
        "clusters": clusters,
        "command": command,
        "mmseqs_version": subprocess.run(
            [str(mmseqs), "version"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "representative_fasta_sha256": file_hash(output / "representatives.fasta"),
        "cluster_tsv_sha256": file_hash(output / "clusters.tsv"),
    }
    atomic_json(output / "verification.json", report)
    return report


def _task_role(split: str) -> str:
    return {
        "train": "probe_fit",
        "valid": "validation",
        "validation": "validation",
        "gallery": "evaluation_reference",
        "test": "test",
    }.get(split, "protected")


def build_evaluation_union(*, legacy_root: Path, q9_root: Path, output: Path) -> dict[str, Any]:
    """Freeze the exact union used as decontamination queries."""

    output.mkdir(parents=True, exist_ok=False)
    sequences = {
        valid_digest(header): sequence
        for header, sequence in iter_fasta(legacy_root / "evaluation_all_splits.fasta")
    }
    memberships: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    with (legacy_root / "sequence_memberships.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            digest = valid_digest(row["sha256"])
            for item in row["memberships"]:
                protocol = "contact-p-at-l" if item["task"] == "contact" else "pcore-v0.2"
                memberships[digest].add((protocol, item["task"], item["split"], item["role"]))
    legacy_digests = set(sequences)
    receipt = json.loads((q9_root / "DATASET_RECEIPT.json").read_text())
    task_reports: dict[str, Any] = {}
    for task, fields in Q9_TASKS.items():
        path = q9_root / "tasks" / f"{task}.jsonl"
        expected = receipt["task_receipts"][task]
        if file_hash(path) != expected["sha256"]:
            raise ValueError(f"Q9 task changed after its receipt: {task}")
        occurrences = 0
        task_digests: set[str] = set()
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                split = str(row["split"])
                for field in fields:
                    raw = row.get(field)
                    if not raw:
                        continue
                    sequence = canonicalize(raw)
                    digest = sequence_hash(sequence)
                    declared = row.get(f"{field}_sha256")
                    if field == "sequence":
                        declared = row.get("sequence_sha256", declared)
                    if declared is not None and valid_digest(str(declared)) != digest:
                        raise ValueError(f"Q9 digest mismatch at {path}:{line_number}:{field}")
                    observed = sequences.setdefault(digest, sequence)
                    if observed != sequence:
                        raise ValueError(f"SHA-256 collision for {digest}")
                    memberships[digest].add(
                        ("pcore-v0.5-alpha-q9", task, split, _task_role(split))
                    )
                    task_digests.add(digest)
                    occurrences += 1
        task_reports[task] = {
            "task_file": str(path.resolve()),
            "task_file_sha256": file_hash(path),
            "sequence_occurrences": occurrences,
            "unique_sequences": len(task_digests),
            "blocked_for_scoring_but_protected": task == "cafa5_mf_nk30_hard",
        }
    all_path = output / "evaluation_all_splits.fasta"
    delta_path = output / "evaluation_q9_delta.fasta"
    membership_path = output / "sequence_memberships.jsonl"
    exact_path = output / "evaluation_exact_sha256.txt"
    with (
        all_path.open("w") as all_handle,
        delta_path.open("w") as delta_handle,
        membership_path.open("w") as member_handle,
        exact_path.open("w") as exact_handle,
    ):
        for digest in sorted(sequences):
            sequence = sequences[digest]
            write_fasta(all_handle, digest, sequence)
            if digest not in legacy_digests:
                write_fasta(delta_handle, digest, sequence)
            exact_handle.write(digest + "\n")
            member_handle.write(
                json.dumps(
                    {
                        "sha256": digest,
                        "length": len(sequence),
                        "memberships": [
                            {"protocol": protocol, "task": task, "split": split, "role": role}
                            for protocol, task, split, role in sorted(memberships[digest])
                        ],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    report = {
        "schema_version": 2,
        "status": "verified",
        "protocol": "protein-evaluation-split-ledger-v2",
        "policy": (
            "every fit, validation, reference, test, and blocked-candidate sequence "
            "is excluded from pretraining"
        ),
        "evaluation_protocols": ["contact-p-at-l", "pcore-v0.2", "pcore-v0.5-alpha-q9"],
        "legacy_unique_sequences": len(legacy_digests),
        "union_unique_sequences": len(sequences),
        "q9_delta_unique_sequences": len(sequences) - len(legacy_digests),
        "union_unique_residues": sum(map(len, sequences.values())),
        "tasks": task_reports,
        "inputs": {
            "legacy_ledger_sha256": file_hash(legacy_root / "EVALUATION_SPLIT_LEDGER.json"),
            "legacy_memberships_sha256": file_hash(legacy_root / "sequence_memberships.jsonl"),
            "q9_receipt_sha256": file_hash(q9_root / "DATASET_RECEIPT.json"),
        },
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
            for path in (all_path, delta_path, membership_path, exact_path)
        },
    }
    atomic_json(output / "EVALUATION_SPLIT_LEDGER.json", report)
    return report


def run_delta_screen(
    *,
    query_fasta: Path,
    target_db_root: Path,
    output: Path,
    mmseqs: Path,
    threads: int,
    concurrent_sources: bool = False,
) -> dict[str, Any]:
    """Search Q9-only queries against already-built complete target databases."""

    output.mkdir(parents=True, exist_ok=False)
    (output / "db").mkdir()
    (output / "results").mkdir()
    (output / "tmp").mkdir()
    query_db = output / "db/query"
    subprocess.run([str(mmseqs), "createdb", str(query_fasta), str(query_db)], check=True)
    threads_per_source = max(1, threads // len(SOURCES)) if concurrent_sources else threads

    def screen_source(source: str) -> tuple[str, list[str], dict[str, Any]]:
        target = target_db_root / source
        if not target.with_suffix(".dbtype").is_file():
            raise FileNotFoundError(f"missing reusable complete target database: {target}")
        result = output / f"results/{source}"
        temporary = output / f"tmp/{source}"
        hits = output / f"results/{source}.tsv"
        command = [
            str(mmseqs),
            "search",
            str(query_db),
            str(target),
            str(result),
            str(temporary),
            "--min-seq-id",
            "0.30",
            "-c",
            "0.80",
            "--cov-mode",
            "0",
            "--max-seqs",
            "1000000",
            "-s",
            "7.5",
            "--threads",
            str(threads_per_source),
        ]
        subprocess.run(command, check=True)
        subprocess.run(
            [
                str(mmseqs),
                "convertalis",
                str(query_db),
                str(target),
                str(result),
                str(hits),
                "--format-output",
                "query,target,pident,alnlen,qcov,tcov,evalue,bits",
            ],
            check=True,
        )
        artifact = {
            "path": str(hits.resolve()),
            "bytes": hits.stat().st_size,
            "sha256": file_hash(hits),
        }
        return source, command, artifact

    commands: list[list[str]] = []
    artifacts: dict[str, Any] = {}
    if concurrent_sources:
        # Exact but opt-in: peak RAM is the sum of all three target indexes and
        # exceeds the practical limit of a 1 TB host.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(SOURCES)) as executor:
            futures = [executor.submit(screen_source, source) for source in SOURCES]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
    else:
        results = [screen_source(source) for source in SOURCES]
    for source, command, artifact in results:
        commands.append(command)
        artifacts[source] = artifact
    report = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "mmseqs2-evaluation-delta-search-v1",
        "query_fasta_sha256": file_hash(query_fasta),
        "target_scope": (
            "complete 70%-identity representative FASTAs via verified parent databases"
        ),
        "thresholds": MMSEQS_THRESHOLDS,
        "mmseqs_version": subprocess.run(
            [str(mmseqs), "version"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "total_threads": threads,
        "concurrent_sources": concurrent_sources,
        "threads_per_source": threads_per_source,
        "commands": sorted(commands, key=lambda command: command[3]),
        "artifacts": dict(sorted(artifacts.items())),
    }
    atomic_json(output / "MMSEQS_DELTA_SEARCH_COMPLETE.json", report)
    return report


def _read_digest_file(path: Path) -> set[str]:
    with path.open() as handle:
        values = {valid_digest(line.strip()) for line in handle if line.strip()}
    return values


def finalize_screen(
    *,
    evaluation_root: Path,
    parent_receipt: Path,
    parent_exclusions: Path,
    delta_root: Path,
    cluster_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate delta hits and union them with the frozen full legacy screen."""

    parent = json.loads(parent_receipt.read_text())
    if (
        parent.get("status") != "verified"
        or parent.get("protocol") != LEGACY_SCREEN_PROTOCOL
        or parent.get("scope_used_for_training") != "all evaluation splits"
    ):
        raise ValueError("parent homology screen is not the verified legacy protocol")
    for name in (
        "minimum_sequence_identity",
        "minimum_query_coverage",
        "minimum_target_coverage",
        "coverage_mode",
    ):
        if parent.get("thresholds", {}).get(name) != MMSEQS_THRESHOLDS[name]:
            raise ValueError(f"parent homology threshold changed: {name}")
    if file_hash(parent_exclusions) != parent["excluded_digest_file_sha256"]:
        raise ValueError("parent exclusion file changed")
    delta = json.loads((delta_root / "MMSEQS_DELTA_SEARCH_COMPLETE.json").read_text())
    if delta.get("status") != "complete" or delta.get("thresholds") != MMSEQS_THRESHOLDS:
        raise ValueError("delta search receipt is incomplete or uses different thresholds")
    ledger = json.loads((evaluation_root / "EVALUATION_SPLIT_LEDGER.json").read_text())
    if (
        delta["query_fasta_sha256"]
        != ledger["artifacts"]["evaluation_q9_delta.fasta"]["sha256"]
    ):
        raise ValueError("delta screen did not use the frozen Q9 delta FASTA")
    queries = {
        valid_digest(header)
        for header, _sequence in iter_fasta(evaluation_root / "evaluation_q9_delta.fasta")
    }
    union = _read_digest_file(parent_exclusions)
    parent_count = len(union)
    source_reports: dict[str, Any] = {}
    delta_targets: set[str] = set()
    for source in SOURCES:
        path = delta_root / f"results/{source}.tsv"
        if file_hash(path) != delta["artifacts"][source]["sha256"]:
            raise ValueError(f"delta hit file changed: {source}")
        per_query: Counter[str] = Counter()
        targets: set[str] = set()
        rows = 0
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 8:
                    raise ValueError(f"malformed MMseqs row at {path}:{line_number}")
                query, target = valid_digest(fields[0]), valid_digest(fields[1])
                identity = float(fields[2]) / 100.0
                qcov, tcov = float(fields[4]), float(fields[5])
                qcov = qcov / 100 if qcov > 1 else qcov
                tcov = tcov / 100 if tcov > 1 else tcov
                if (
                    query not in queries
                    or identity + 1e-12 < 0.30
                    or qcov + 1e-12 < 0.80
                    or tcov + 1e-12 < 0.80
                ):
                    raise ValueError(
                        f"MMseqs threshold/query violation at {path}:{line_number}"
                    )
                per_query[query] += 1
                targets.add(target)
                rows += 1
        largest = max(per_query.values(), default=0)
        if largest >= 1_000_000:
            raise ValueError(f"{source} hit the max-seqs cap")
        verification = json.loads((cluster_root / source / "verification.json").read_text())
        if (
            parent.get("sources", {}).get(source, {}).get("representative_fasta_sha256")
            != verification["representative_fasta_sha256"]
        ):
            raise ValueError(f"reused {source} target DB is not bound to this FASTA")
        source_reports[source] = {
            "alignment_rows": rows,
            "matched_queries": len(per_query),
            "delta_excluded_representatives": len(targets),
            "maximum_hits_for_one_query": largest,
            "complete_representative_sequences": verification["clusters"],
            "representative_fasta_sha256": verification["representative_fasta_sha256"],
        }
        delta_targets.update(targets)
    union.update(delta_targets)
    output.mkdir(parents=True, exist_ok=False)
    exclusions = output / "homology_excluded_all_splits.txt"
    with exclusions.open("w") as handle:
        for digest in sorted(union):
            handle.write(digest + "\n")
    exact = evaluation_root / "evaluation_exact_sha256.txt"
    receipt = {
        "schema_version": 2,
        "status": "verified",
        "protocol": SCREEN_PROTOCOL,
        "scope_used_for_training": "all evaluation splits",
        "evaluation_protocols": ledger["evaluation_protocols"],
        "blocked_benchmark_candidates_are_protected": True,
        "thresholds": MMSEQS_THRESHOLDS,
        "evaluation_all_split_queries": ledger["union_unique_sequences"],
        "evaluation_q9_delta_queries": len(queries),
        "parent_excluded_representatives": parent_count,
        "delta_unique_targets": len(delta_targets),
        "excluded_training_representatives": len(union),
        "excluded_digest_file": str(exclusions.resolve()),
        "excluded_digest_file_sha256": file_hash(exclusions),
        "exact_exclusion_file": str(exact.resolve()),
        "exact_exclusion_file_sha256": file_hash(exact),
        "parent_receipt_sha256": file_hash(parent_receipt),
        "delta_receipt_sha256": file_hash(delta_root / "MMSEQS_DELTA_SEARCH_COMPLETE.json"),
        "evaluation_ledger_sha256": file_hash(evaluation_root / "EVALUATION_SPLIT_LEDGER.json"),
        "sources": source_reports,
    }
    atomic_json(output / "HOMOLOGY_EXCLUSION_VERIFIED.json", receipt)
    return receipt


def prepare_sha_buckets(cluster_root: Path, work_root: Path) -> dict[str, Any]:
    """Externally order representative FASTAs using 256 digest-prefix buckets.

    MMseqs2 emits representatives in a deterministic internal database order,
    not lexical SHA order.  Loading hundreds of millions of sequences for one
    in-memory sort is impossible, so this phase first partitions on the leading
    digest byte.  Each bucket is small enough to sort in memory; concatenating
    buckets 00..ff is then a global order.
    """

    root_receipt = work_root / "SHA_BUCKETS_VERIFIED.json"
    if root_receipt.is_file():
        receipt = json.loads(root_receipt.read_text())
        for source in SOURCES:
            source_receipt = work_root / source / "verification.json"
            cluster_receipt = json.loads(
                (cluster_root / source / "verification.json").read_text()
            )
            if (
                not source_receipt.is_file()
                or file_hash(source_receipt) != receipt["sources"][source]["receipt_sha256"]
                or receipt["sources"][source]["input_representative_fasta_sha256"]
                != cluster_receipt["representative_fasta_sha256"]
                or any(
                    not (work_root / source / f"bucket-{index:02x}.tsv").is_file()
                    for index in range(256)
                )
            ):
                raise ValueError(f"cached SHA buckets are incomplete for {source}")
        return receipt
    if work_root.exists():
        raise FileExistsError(f"unverified SHA-sort work directory exists: {work_root}")
    required = sum(
        (cluster_root / source / "representatives.fasta").stat().st_size for source in SOURCES
    )
    free = shutil.disk_usage(work_root.parent).free
    if free < required * 6 // 5:
        raise OSError(
            f"SHA-sort work needs at least {required * 6 // 5:,} free bytes; found {free:,}"
        )
    work_root.mkdir(parents=True)

    def partition_source(source: str) -> tuple[str, dict[str, Any]]:
        source_root = work_root / source
        source_root.mkdir()
        handles: list[IO[str]] = []
        counts = [0] * 256
        fasta = cluster_root / source / "representatives.fasta"
        verification = json.loads((cluster_root / source / "verification.json").read_text())
        try:
            handles = [
                (source_root / f"bucket-{index:02x}.tsv").open(
                    "w", encoding="ascii", buffering=1 << 20
                )
                for index in range(256)
            ]
            total = 0
            for header, sequence in iter_fasta(fasta):
                digest = valid_digest(header)
                if sequence_hash(sequence) != digest:
                    raise ValueError(f"representative/header mismatch in {source}: {digest}")
                bucket = int(digest[:2], 16)
                handles[bucket].write(f"{digest}\t{sequence}\n")
                counts[bucket] += 1
                total += 1
        finally:
            for handle in handles:
                handle.close()
        if total != int(verification["clusters"]):
            raise ValueError(f"representative count differs from cluster receipt: {source}")
        report = {
            "schema_version": 1,
            "status": "verified",
            "protocol": "representative-sha-prefix-buckets-v1",
            "source": source,
            "records": total,
            "bucket_records": counts,
            "input_representative_fasta_sha256": verification["representative_fasta_sha256"],
        }
        atomic_json(source_root / "verification.json", report)
        return source, report

    reports: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SOURCES)) as executor:
        futures = [executor.submit(partition_source, source) for source in SOURCES]
        for future in concurrent.futures.as_completed(futures):
            source, report = future.result()
            reports[source] = report
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "protocol": "representative-global-sha-order-v1",
        "method": "leading-byte partition followed by in-memory sort within each bucket",
        "sources": {
            source: {
                **reports[source],
                "receipt_sha256": file_hash(work_root / source / "verification.json"),
            }
            for source in SOURCES
        },
    }
    atomic_json(root_receipt, receipt)
    return receipt


def iter_sha_sorted_buckets(work_root: Path, source: str) -> Iterator[tuple[str, str]]:
    """Yield one source in strict global digest order from its external-sort buckets."""

    previous = ""
    for index in range(256):
        path = work_root / source / f"bucket-{index:02x}.tsv"
        rows: list[tuple[str, str]] = []
        with path.open(encoding="ascii") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.rstrip("\n").split("\t", 1)
                if len(fields) != 2:
                    raise ValueError(f"malformed SHA bucket row at {path}:{line_number}")
                digest = valid_digest(fields[0])
                if int(digest[:2], 16) != index:
                    raise ValueError(f"digest is in the wrong SHA bucket: {path}:{digest}")
                rows.append((digest, fields[1]))
        rows.sort(key=lambda row: row[0])
        for digest, sequence in rows:
            if previous and digest <= previous:
                raise ValueError(
                    f"duplicate/non-increasing representative in {source}: {digest}"
                )
            previous = digest
            yield digest, sequence


class ParquetShardWriter:
    def __init__(self, root: Path, source: str, split: str, target_residues: int) -> None:
        self.root = root / split / source
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.split = split
        self.target_residues = target_residues
        self.rows: list[dict[str, Any]] = []
        self.residues = 0
        self.index = 0
        self.receipts: list[dict[str, Any]] = []
        self.schema = pa.schema(
            [("sequence", pa.string()), ("sha256", pa.string()), ("length", pa.int32())]
        )

    def append(self, digest: str, sequence: str) -> None:
        if self.rows and self.residues + len(sequence) > self.target_residues:
            self.flush()
        self.rows.append({"sequence": sequence, "sha256": digest, "length": len(sequence)})
        self.residues += len(sequence)

    def flush(self) -> None:
        if not self.rows:
            return
        path = self.root / f"shard-{self.index:05d}.parquet"
        partial = path.with_suffix(".parquet.partial")
        pq.write_table(
            pa.Table.from_pylist(self.rows, schema=self.schema),
            partial,
            compression="zstd",
            compression_level=7,
            row_group_size=65_536,
            use_dictionary=False,
            write_statistics=True,
        )
        partial.replace(path)
        self.receipts.append(
            {
                "path": str(path),
                "records": len(self.rows),
                "residues": self.residues,
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
                "minimum_sequence_sha256": self.rows[0]["sha256"],
                "maximum_sequence_sha256": self.rows[-1]["sha256"],
            }
        )
        self.rows = []
        self.residues = 0
        self.index += 1

    def finish(self) -> list[dict[str, Any]]:
        self.flush()
        return self.receipts


def shard_release(
    *,
    cluster_root: Path,
    screen_root: Path,
    evaluation_root: Path,
    output: Path,
    validation_per_source: int,
    shard_residues: int,
) -> dict[str, Any]:
    """Write complete eligible representatives as deterministic Parquet shards."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    sort_work = output.parent / f"{output.name}.sha-sort-work"
    ordering_receipt = prepare_sha_buckets(cluster_root, sort_work)
    homology = _read_digest_file(screen_root / "homology_excluded_all_splits.txt")
    exact = _read_digest_file(evaluation_root / "evaluation_exact_sha256.txt")
    excluded = homology | exact
    validation: dict[str, list[tuple[str, str]]] = {}
    global_validation: set[str] = set()
    for source in SOURCES:
        selected: list[tuple[str, str]] = []
        for digest, sequence in iter_sha_sorted_buckets(sort_work, source):
            if sequence_hash(sequence) != digest:
                raise ValueError(f"representative/header mismatch in {source}: {digest}")
            if digest in excluded or not 32 <= len(sequence) <= 16_384:
                continue
            selected.append((digest, sequence))
            if len(selected) == validation_per_source:
                break
        if len(selected) != validation_per_source:
            raise ValueError(f"not enough validation representatives for {source}")
        validation[source] = selected
        global_validation.update(digest for digest, _sequence in selected)

    def shard_source(source: str) -> tuple[str, dict[str, Any], Counter[str]]:
        train_writer = ParquetShardWriter(output, source, "train", shard_residues)
        validation_writer = ParquetShardWriter(output, source, "validation", 1 << 62)
        for digest, sequence in validation[source]:
            validation_writer.append(digest, sequence)
        rejected: Counter[str] = Counter()
        previous = ""
        accepted = 0
        accepted_residues = 0
        scanned = 0
        for digest, sequence in iter_sha_sorted_buckets(sort_work, source):
            scanned += 1
            if previous and digest <= previous:
                raise ValueError(f"representatives are not strictly SHA-sorted: {source}")
            previous = digest
            if sequence_hash(sequence) != digest:
                raise ValueError(f"representative/header mismatch in {source}: {digest}")
            if digest in exact:
                rejected["evaluation_exact"] += 1
            elif digest in homology:
                rejected["evaluation_homology"] += 1
            elif not 32 <= len(sequence) <= 16_384:
                rejected["length"] += 1
            elif digest in global_validation:
                rejected["global_validation"] += 1
            else:
                train_writer.append(digest, sequence)
                accepted += 1
                accepted_residues += len(sequence)
        train = train_writer.finish()
        valid = validation_writer.finish()
        if not train or not valid:
            raise ValueError(f"empty release split for {source}")
        if scanned != accepted + sum(rejected.values()):
            raise AssertionError(f"incomplete representative accounting for {source}")
        cluster_receipt = json.loads((cluster_root / source / "verification.json").read_text())
        if scanned != int(cluster_receipt["clusters"]):
            raise ValueError(f"representative count differs from cluster receipt: {source}")
        for row in (*train, *valid):
            row["path"] = str(Path(row["path"]).relative_to(output))
        source_row = {
            "train": train,
            "validation": valid,
            "representative_records_scanned": scanned,
            "train_records": accepted,
            "train_residues": accepted_residues,
            "rejected": dict(rejected),
        }
        return source, source_row, rejected

    source_rows: dict[str, Any] = {}
    rejection_totals: Counter[str] = Counter()
    # A shared exclusion set avoids tripling tens of gigabytes of Python hash
    # state, while C-level Parquet compression can progress in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SOURCES)) as executor:
        futures = [executor.submit(shard_source, source) for source in SOURCES]
        for future in concurrent.futures.as_completed(futures):
            source, source_row, rejected = future.result()
            source_rows[source] = source_row
            rejection_totals.update(rejected)
    source_rows = {source: source_rows[source] for source in SOURCES}
    screen_receipt = json.loads((screen_root / "HOMOLOGY_EXCLUSION_VERIFIED.json").read_text())
    manifest = {
        "schema_version": 1,
        "status": "generated",
        "protocol": RELEASE_PROTOCOL,
        "release_id": "full-open-cluster-representatives-v2",
        "sampling_unit": "one_representative_per_source_specific_70pct_identity_cluster",
        "ordering": "ascending_sequence_sha256",
        "ordering_receipt_sha256": hashlib.sha256(
            canonical_json(ordering_receipt).encode()
        ).hexdigest(),
        "partial_download": (
            "smallest deterministic per-source shard prefix satisfying a run budget"
        ),
        "filters": {"minimum_length": 32, "maximum_length": 16_384},
        "validation": {
            "records_per_source": validation_per_source,
            "selection": "first eligible SHA-ordered representatives",
            "globally_excluded_from_all_training_arms": True,
        },
        "decontamination": {
            "scope": "all_evaluation_splits",
            "evaluation_protocols": screen_receipt["evaluation_protocols"],
            "blocked_benchmark_candidates_are_protected": True,
            "thresholds": screen_receipt["thresholds"],
            "homology_exclusion_receipt_sha256": file_hash(
                screen_root / "HOMOLOGY_EXCLUSION_VERIFIED.json"
            ),
            "exact_exclusion_file_sha256": file_hash(
                evaluation_root / "evaluation_exact_sha256.txt"
            ),
            "homology_exclusion_file_sha256": file_hash(
                screen_root / "homology_excluded_all_splits.txt"
            ),
        },
        "sources": source_rows,
        "rejected": dict(rejection_totals),
    }
    atomic_json(output / "manifest.generated.json", manifest)
    return manifest


def verify_release(root: Path, *, screen_root: Path, evaluation_root: Path) -> dict[str, Any]:
    """Independently re-read all Parquet rows and promote the generated manifest."""

    generated_path = root / "manifest.generated.json"
    manifest = json.loads(generated_path.read_text())
    homology_path = screen_root / "homology_excluded_all_splits.txt"
    exact_path = evaluation_root / "evaluation_exact_sha256.txt"
    if (
        file_hash(homology_path)
        != manifest["decontamination"]["homology_exclusion_file_sha256"]
    ):
        raise ValueError("homology exclusion file differs from the generated manifest")
    if file_hash(exact_path) != manifest["decontamination"]["exact_exclusion_file_sha256"]:
        raise ValueError("exact exclusion file differs from the generated manifest")
    excluded = _read_digest_file(homology_path) | _read_digest_file(exact_path)
    global_validation: set[str] = set()
    sources: dict[str, Any] = {}
    for source in SOURCES:
        observed: dict[str, Any] = {}
        for split in ("train", "validation"):
            validation_digests: set[str] = set()
            records = residues = 0
            previous = ""
            for shard in manifest["sources"][source][split]:
                path = root / shard["path"]
                if file_hash(path) != shard["sha256"]:
                    raise ValueError(f"shard checksum mismatch: {path}")
                table = pq.read_table(path, columns=["sequence", "sha256", "length"])
                rows = table.to_pydict()
                local_records = local_residues = 0
                for sequence, digest, length in zip(
                    rows["sequence"], rows["sha256"], rows["length"], strict=True
                ):
                    if sequence_hash(sequence) != digest or len(sequence) != int(length):
                        raise ValueError(f"row integrity failure: {path}:{digest}")
                    if previous and digest <= previous:
                        raise ValueError(f"non-increasing digest: {path}:{digest}")
                    previous = digest
                    if digest in excluded:
                        raise ValueError(
                            f"evaluation-contaminated digest in {source}/{split}: {digest}"
                        )
                    if split == "validation":
                        validation_digests.add(digest)
                    local_records += 1
                    local_residues += len(sequence)
                if local_records != shard["records"] or local_residues != shard["residues"]:
                    raise ValueError(f"shard accounting mismatch: {path}")
                records += local_records
                residues += local_residues
            observed[split] = {
                "records": records,
                "residues": residues,
                "shards": len(manifest["sources"][source][split]),
            }
            if split == "validation":
                global_validation.update(validation_digests)
        sources[source] = observed
    # Validation sequences from any source must be absent from every train arm.
    for source in SOURCES:
        for shard in manifest["sources"][source]["train"]:
            table = pq.read_table(root / shard["path"], columns=["sha256"])
            if global_validation & set(table["sha256"].to_pylist()):
                raise ValueError(f"global validation leaks into {source} training")
    manifest["status"] = "verified"
    manifest["verification"] = {
        "protocol": "protein-corpus-parquet-verification-v1",
        "generated_manifest_sha256": file_hash(generated_path),
        "all_shard_hashes_recomputed": True,
        "all_sequence_hashes_recomputed": True,
        "exact_and_homology_exclusion_intersection": 0,
        "global_train_validation_intersection": 0,
        "sources": sources,
    }
    manifest_path = root / "manifest.json"
    atomic_json(manifest_path, manifest)
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "protocol": "protein-corpus-parquet-verification-v1",
        "manifest_sha256": file_hash(manifest_path),
        "sources": sources,
    }
    atomic_json(root / "RELEASE_VERIFIED.json", receipt)
    return receipt


def _paths(values: Sequence[str]) -> list[Path]:
    paths = [Path(value) for value in values]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--data-root", type=Path, required=True)
    download.add_argument("--omg-manifest", type=Path, required=True)
    normalize = sub.add_parser("normalize")
    normalize.add_argument("--source", choices=SOURCES, required=True)
    normalize.add_argument("--input", action="append", required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--partitions", type=int, default=DEFAULT_PARTITIONS)
    dedup = sub.add_parser("deduplicate")
    dedup.add_argument("--input", action="append", required=True)
    dedup.add_argument("--output", type=Path, required=True)
    dedup.add_argument("--partitions", type=int, default=DEFAULT_PARTITIONS)
    cluster = sub.add_parser("cluster")
    cluster.add_argument("--dedup-root", type=Path, required=True)
    cluster.add_argument("--output", type=Path, required=True)
    cluster.add_argument("--source", choices=SOURCES, required=True)
    cluster.add_argument("--mmseqs", type=Path, required=True)
    cluster.add_argument("--threads", type=int, default=64)
    union = sub.add_parser("evaluation-union")
    union.add_argument("--legacy-root", type=Path, required=True)
    union.add_argument("--q9-root", type=Path, required=True)
    union.add_argument("--output", type=Path, required=True)
    screen = sub.add_parser("delta-screen")
    screen.add_argument("--query-fasta", type=Path, required=True)
    screen.add_argument("--target-db-root", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--mmseqs", type=Path, required=True)
    screen.add_argument("--threads", type=int, default=64)
    screen.add_argument(
        "--concurrent-sources",
        action="store_true",
        help="search all source DBs at once; requires more than 1 TB RAM",
    )
    finalize = sub.add_parser("finalize-screen")
    finalize.add_argument("--evaluation-root", type=Path, required=True)
    finalize.add_argument("--parent-receipt", type=Path, required=True)
    finalize.add_argument("--parent-exclusions", type=Path, required=True)
    finalize.add_argument("--delta-root", type=Path, required=True)
    finalize.add_argument("--cluster-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    shard = sub.add_parser("shard")
    shard.add_argument("--cluster-root", type=Path, required=True)
    shard.add_argument("--screen-root", type=Path, required=True)
    shard.add_argument("--evaluation-root", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--validation-per-source", type=int, default=4096)
    shard.add_argument("--shard-residues", type=int, default=DEFAULT_SHARD_RESIDUES)
    verify = sub.add_parser("verify-release")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--screen-root", type=Path, required=True)
    verify.add_argument("--evaluation-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        result = download_raw(args.data_root, args.omg_manifest)
    elif args.command == "normalize":
        result = normalize_source(
            source=args.source,
            inputs=_paths(args.input),
            output=args.output,
            partitions=args.partitions,
        )
    elif args.command == "deduplicate":
        result = deduplicate(_paths(args.input), args.output, args.partitions)
    elif args.command == "cluster":
        result = run_cluster(
            dedup_root=args.dedup_root,
            output=args.output,
            source=args.source,
            mmseqs=args.mmseqs,
            threads=args.threads,
            concurrent_sources=args.concurrent_sources,
        )
    elif args.command == "evaluation-union":
        result = build_evaluation_union(
            legacy_root=args.legacy_root, q9_root=args.q9_root, output=args.output
        )
    elif args.command == "delta-screen":
        result = run_delta_screen(
            query_fasta=args.query_fasta,
            target_db_root=args.target_db_root,
            output=args.output,
            mmseqs=args.mmseqs,
            threads=args.threads,
        )
    elif args.command == "finalize-screen":
        result = finalize_screen(
            evaluation_root=args.evaluation_root,
            parent_receipt=args.parent_receipt,
            parent_exclusions=args.parent_exclusions,
            delta_root=args.delta_root,
            cluster_root=args.cluster_root,
            output=args.output,
        )
    elif args.command == "shard":
        result = shard_release(
            cluster_root=args.cluster_root,
            screen_root=args.screen_root,
            evaluation_root=args.evaluation_root,
            output=args.output,
            validation_per_source=args.validation_per_source,
            shard_residues=args.shard_residues,
        )
    else:
        result = verify_release(
            args.root,
            screen_root=args.screen_root,
            evaluation_root=args.evaluation_root,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
