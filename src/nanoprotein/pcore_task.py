"""Isolated, time-limitable workers for the bounded P-CORE diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=("remote_homology", "human_ppi", "flip2_hydro_low_to_high"),
    )
    parser.add_argument("--external-src", type=Path, required=True)
    parser.add_argument("--embedding-store", type=Path, required=True)
    parser.add_argument("--pcore-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.external_src))
    try:
        from autoresearch_esm.pcore_probe import (  # type: ignore[import-not-found]
            _fitness_probe,
            _ppi_probe,
            _remote_probe,
            load_store,
        )
    finally:
        sys.path.pop(0)

    store = load_store(args.embedding_store)
    processed = args.pcore_root / "processed"
    raw = args.pcore_root / "raw"
    probes: dict[str, Callable[[], dict[str, Any]]] = {
        "remote_homology": lambda: _remote_probe(
            store, processed, 20260819, 0, permute_targets=False
        ),
        "human_ppi": lambda: _ppi_probe(store, processed, 20260819, 0, permute_targets=False),
        "flip2_hydro_low_to_high": lambda: _fitness_probe(
            store,
            raw / "flip2_hydro_low_to_high.csv.gz",
            20260819,
            0,
            permute_targets=False,
        ),
    }
    result = probes[args.task]()
    _atomic_json(
        args.output,
        {
            "schema_version": 1,
            "protocol": "pcore-diagnostic-v1",
            "task": args.task,
            "bootstrap_replicates": 0,
            "result": result,
        },
    )
    print(json.dumps({"event": "pcore_diagnostic_task_complete", "task": args.task}))


if __name__ == "__main__":
    main()
