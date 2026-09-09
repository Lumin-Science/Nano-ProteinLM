#!/usr/bin/env python3
"""Hash the contact scoring cache once before parallel P@L inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contact_cache import verify_contact_scoring_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify_contact_scoring_cache(
        cache_root=args.cache_root,
        output=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
