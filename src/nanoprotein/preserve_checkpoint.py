"""Copy audited milestone checkpoints atomically without overwriting history."""

import argparse
import json
import shutil
from pathlib import Path

from .data import file_sha256


def preserve(checkpoint, receipt, destination, interval):
    record = json.loads(receipt.read_text())
    if record["status"] != "passed":
        raise ValueError("checkpoint requires a passed audit")
    step = int(record["optimizer_steps"])
    if step % interval:
        return None
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / f"checkpoint-{step:06d}.pt"
    expected = record["checkpoint_sha256"]
    if final.exists():
        if file_sha256(final) != expected:
            raise FileExistsError("different checkpoint already occupies this milestone")
    else:
        partial = final.with_suffix(".pt.partial")
        shutil.copyfile(checkpoint, partial)
        if file_sha256(partial) != expected:
            raise ValueError("checkpoint copy checksum mismatch")
        partial.replace(final)
    result = {
        "status": "passed",
        "optimizer_steps": step,
        "checkpoint": str(final),
        "checkpoint_sha256": expected,
        "bytes": final.stat().st_size,
    }
    (destination / f"checkpoint-{step:06d}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--milestone-interval", type=int, default=100000)
    args = parser.parse_args()
    if args.milestone_interval <= 0:
        parser.error("milestone interval must be positive")
    print(
        json.dumps(
            preserve(args.checkpoint, args.receipt, args.destination, args.milestone_interval)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
