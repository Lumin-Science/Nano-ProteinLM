#!/usr/bin/env python3
"""Prepare a bounded mmap pilot corpus from Step-9 cluster representatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_protein.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pcore-index", type=Path, required=True)
    parser.add_argument("--contact-manifest", type=Path, required=True)
    parser.add_argument("--train-per-source", type=int, default=250_000)
    parser.add_argument("--validation-per-source", type=int, default=4_096)
    parser.add_argument("--skip-sequence-hash-verification", action="store_true")
    args = parser.parse_args()
    result = prepare_dataset(
        cluster_root=args.cluster_root,
        output_root=args.output_root,
        pcore_index=args.pcore_index,
        contact_manifest=args.contact_manifest,
        train_per_source=args.train_per_source,
        validation_per_source=args.validation_per_source,
        verify_sequence_hashes=not args.skip_sequence_hash_verification,
    )
    print(json.dumps({"event": "data_ready", **result}, sort_keys=True))


if __name__ == "__main__":
    main()
