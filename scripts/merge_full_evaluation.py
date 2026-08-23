#!/usr/bin/env python3
"""Merge parallel exact P-CORE and contact shards into one evaluation receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_protein.data import file_sha256
from nano_protein.evaluate import merge_full_evaluation, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-report", action="append", type=Path, required=True)
    parser.add_argument("--pcore-report", type=Path, required=True)
    parser.add_argument("--expected-contact-chains", type=int, required=True)
    parser.add_argument("--contact-bootstrap", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = merge_full_evaluation(
        contact_paths=args.contact_report,
        pcore_path=args.pcore_report,
        expected_contact_chains=args.expected_contact_chains,
        contact_bootstrap=args.contact_bootstrap,
    )
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "event": "full_evaluation_complete",
                "report": str(args.output.resolve()),
                "report_sha256": file_sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
