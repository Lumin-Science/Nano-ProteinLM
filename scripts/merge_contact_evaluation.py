#!/usr/bin/env python3
"""Merge exact deterministic P@L shards without running P-CORE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_protein.data import file_sha256
from nano_protein.evaluate import merge_contact_evaluation, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-report", action="append", type=Path, required=True)
    parser.add_argument("--expected-contact-chains", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = merge_contact_evaluation(
        contact_paths=args.contact_report,
        expected_contact_chains=args.expected_contact_chains,
    )
    write_json(args.output, receipt)
    print(
        json.dumps(
            {
                "event": "contact_evaluation_complete",
                "report": str(args.output.resolve()),
                "report_sha256": file_sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
