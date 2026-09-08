import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from nano_protein.evaluate import bootstrap_mean_interval, write_json

root = Path(sys.argv[1])
out = root / "eval-p-at-l"
result = json.loads((out / "P_AT_L.json").read_text())
training = json.loads((root / "TRAINING_VERIFIED.json").read_text())
probe = json.loads((out / "CONTACT_PROBE.json").read_text())
assert (
    result["checkpoint_sha256"] == training["checkpoint_sha256"] == probe["checkpoint_sha256"]
)
assert (
    probe["dataset_manifest_sha256"]
    == "c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3"
)
rows = []
for component in result["components"]:
    p = Path(component["path"])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == component["sha256"]
    report = json.loads(p.read_text())
    assert report["checkpoint_sha256"] == training["checkpoint_sha256"]
    rows.extend(report["contact"]["rows"])
assert (
    len(rows) == len({row["chain_id"] for row in rows}) == result["evaluation_chains"] == 20775
)
values = np.array([row["precision_at_l"] for row in rows], dtype=np.float64)
assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
assert abs(values.mean() - result["p_at_l"]) < 1e-12
uncertainty = bootstrap_mean_interval(values, replicates=5000, seed=20260820)
write_json(
    out / "P_AT_L_UNCERTAINTY.json",
    {
        "checkpoint_sha256": result["checkpoint_sha256"],
        "p_at_l": result["p_at_l"],
        "uncertainty": uncertainty,
        "note": "Across-chain bootstrap confidence interval, not training-seed SD.",
    },
)
print(
    json.dumps({"status": "passed", "p_at_l": result["p_at_l"], "uncertainty": uncertainty}),
    flush=True,
)
