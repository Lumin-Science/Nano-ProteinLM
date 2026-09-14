"""Preserve the exact production recipe and source bundle on project storage."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-atlas-setting3-b1024-20260914")
DEST = Path(
    "/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-atlas-setting3-b1024-20260914/source"
)
DEST.mkdir(parents=True, exist_ok=True)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


files = {}
for relative in (
    "source.bundle",
    "full-training/config.yaml",
    "full-training/run_contract.json",
    "data/manifest.json",
    "DATA_PRESERVED.json",
):
    source = ROOT / relative
    destination = DEST / source.name
    expected = digest(source)
    if destination.exists():
        assert digest(destination) == expected
    else:
        partial = destination.with_suffix(destination.suffix + ".partial")
        shutil.copyfile(source, partial)
        assert digest(partial) == expected
        partial.replace(destination)
    files[relative] = dict(path=str(destination), sha256=expected)
receipt = dict(status="passed", files=files)
(ROOT / "SOURCE_PRESERVED.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
