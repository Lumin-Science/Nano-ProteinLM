import datetime
import json
import math
from pathlib import Path

import torch

torch.set_num_threads(2)
ROOT = Path("/scratch/muchenli/Nano-Protein-LM-nibi-setting3-b2048-400k-20260910")
paths = [
    Path(
        "/project/def-lsigal/muchenli/Nano-Protein-LM/checkpoints/nibi-paired-unique-b2048-100k-20260909/setting3/checkpoint-final.pt"
    ),
    ROOT / "qualification/checkpoint-final.pt",
]
packets = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
assert [p["optimizer_step"] for p in packets] == [100000, 100200]
assert packets[0]["model_config"] == packets[1]["model_config"]
assert list(packets[0]["model"]) == list(packets[1]["model"])
results = []
for path, packet in zip(paths, packets, strict=True):
    item = {
        "checkpoint": str(path),
        "optimizer_step": packet["optimizer_step"],
        "world_size": packet["world_size"],
        "optimizers": [],
    }
    for child in packet["optimizer"]["optimizers"]:
        state = child["state_dict"]
        groups = []
        for group in state["param_groups"]:
            groups.append(
                {k: v for k, v in group.items() if k != "params"}
                | {"parameter_count": len(group["params"])}
            )
        steps = sorted({int(s["step"].item()) for s in state["state"].values() if "step" in s})
        if child["name"] == "adamw":
            assert steps == [packet["optimizer_step"]]
        tensors = [
            t
            for s in state["state"].values()
            for t in s.values()
            if torch.is_tensor(t) and t.ndim > 0
        ]
        assert all(torch.isfinite(t).all().item() for t in tensors)
        rms = math.sqrt(
            sum(t.float().square().sum().item() for t in tensors)
            / sum(t.numel() for t in tensors)
        )
        assert rms > 0
        item["optimizers"].append(
            {
                "name": child["name"],
                "state_entries": len(state["state"]),
                "groups": groups,
                "adam_steps": steps,
                "state_tensor_rms": rms,
            }
        )
    results.append(item)
for before, after in zip(
    packets[0]["optimizer"]["optimizers"], packets[1]["optimizer"]["optimizers"], strict=True
):
    assert before["name"] == after["name"]
    assert before["state_dict"]["param_groups"] == after["state_dict"]["param_groups"]
    assert before["state_dict"]["state"].keys() == after["state_dict"]["state"].keys()
    for key in before["state_dict"]["state"]:
        a = before["state_dict"]["state"][key]
        b = after["state_dict"]["state"][key]
        assert a.keys() == b.keys()
        assert all(a[k].shape == b[k].shape for k in a if torch.is_tensor(a[k]))
result = {
    "status": "passed",
    "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": (
        "read-only inspection of immutable 100k parent and 100200 qualification checkpoints; "
        "production is untouched"
    ),
    "identical_model_config_and_parameter_order": True,
    "optimizer_group_values_and_parameter_indices_unchanged": True,
    "optimizer_slot_keys_and_tensor_shapes_preserved": True,
    "adam_step_counter_continued_without_reset": True,
    "checkpoints": results,
}
(ROOT / "RESUME_STATE_INSPECTION.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
