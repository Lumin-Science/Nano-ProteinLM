#!/usr/bin/env python3
"""Fit and receipt-bind the frozen P@L probe once for parallel inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import file_sha256
from .evaluate import fit_contact_probe_receipt, load_checkpoint, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--external-src", type=Path, required=True)
    parser.add_argument("--contact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("contact probe fitting requires CUDA")
    np.random.seed(20260821)
    torch.manual_seed(20260821)
    torch.cuda.manual_seed_all(20260821)
    device = torch.device("cuda", 0)
    model, _packet = load_checkpoint(args.checkpoint, device)
    receipt = fit_contact_probe_receipt(
        model,
        checkpoint_sha256=file_sha256(args.checkpoint),
        dataset_root=args.contact_root,
        external_src=args.external_src,
        device=device,
    )
    write_json(args.output, receipt)
    print(
        json.dumps(
            {
                "event": "contact_probe_fitted",
                "receipt": str(args.output.resolve()),
                "receipt_sha256": file_sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
