#!/usr/bin/env python3
"""Build the receipt-bound static contact scoring cache once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_protein.contact_cache import build_contact_scoring_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--external-src", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    receipt = build_contact_scoring_cache(
        dataset_root=args.dataset_root,
        external_src=args.external_src,
        output_root=args.output_root,
        workers=args.workers,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
