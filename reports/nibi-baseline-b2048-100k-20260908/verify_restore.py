import json
from pathlib import Path

import torch

from nano_protein.evaluate import load_checkpoint
from nano_protein.train import build_optimizer

root = Path("/scratch/muchenli/Nano-Protein-LM-nibi-b2048-100k-20260908")
model, packet = load_checkpoint(
    root / "trial-8gpu/checkpoint-final.pt", torch.device("cuda", 0)
)
optimizer = build_optimizer(model, packet["train_config"])
optimizer.load_state_dict(packet["optimizer"])
for name, value in model.state_dict().items():
    torch.testing.assert_close(value.cpu(), packet["model"][name].cpu(), rtol=0, atol=0)
actual = optimizer.state_dict()
assert actual["param_groups"] == packet["optimizer"]["param_groups"]
for index, state in actual["state"].items():
    for name, value in state.items():
        expected = packet["optimizer"]["state"][index][name]
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value.cpu(), expected.cpu(), rtol=0, atol=0)
        else:
            assert value == expected
receipt = {
    "status": "passed",
    "all_model_tensors_restored_exactly": True,
    "all_adamw_tensors_restored_exactly": True,
    "adamw_parameter_states": len(actual["state"]),
}
del model, optimizer, actual
other = torch.load(
    root / "resume-8gpu/checkpoint-final.pt", map_location="cpu", weights_only=False
)
delta = sum(
    (packet["model"][k].cpu().double() - v.double()).square().sum().item()
    for k, v in other["model"].items()
)
norm = sum(v.double().square().sum().item() for v in other["model"].values())
receipt["after_100_resumed_updates_model_relative_l2_difference"] = (delta / norm) ** 0.5
(root / "EXACT_STATE_RESTORE.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
