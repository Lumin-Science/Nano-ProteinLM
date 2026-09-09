"""Pinned Hopper FA3 kernel, loaded independently of the training dependency lock."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

import torch

FA3_REPOSITORY = "kernels-community/flash-attn3"
FA3_REVISION = "e29f138fc363b396e5d2706c8a5f6fa7d36f41e0"
FA3_BUILDS = {
    "12.6": "build/torch-stable-abi29-cu126-x86_64-linux",
    "13.0": "build/torch-stable-abi210-cu130-x86_64-linux",
}
FA3_CUDA13_DIGESTS = {
    "__init__.py": "29756640933e2a159cd94a9625eb2ea4268ffa6a80bd3c7bc86b9e1404e09471",
    "_flash_attn3_cuda_477ab85.abi3.so": (
        "a79c38ca3cbf5f6630e96eb1ab6022b21be7d93e2aa2461d0da7341a0f243f30"
    ),
    "_ops.py": "85c709ad5f7948f011137ca71011e9c27cbf6055d9aa964a74675b4d39ca4ec7",
    "flash_attn3/__init__.py": (
        "0c560f96b857c188c4a8297ff27d1299619957c3459a5e433e132329fbfdf066"
    ),
    "flash_attn_config.py": "bb1a3e7b231c0e2b7fffc603692e11b64e414cc2b34d22bdb4e2938dd5957bfc",
    "flash_attn_interface.py": (
        "cb46c3d89606305817ea20eb7b7cc9376b013eeabfd7637a328e7967e1d008d4"
    ),
    "metadata.json": "393b22374fd625c20f1a046ca29cffe482410502dde8f27f72099b3c4fac0a31",
}


def fa3_build() -> str:
    if sys.platform != "linux" or torch.version.cuda not in FA3_BUILDS:
        raise RuntimeError("the pinned FA3 builds require Linux and CUDA 12.6 or 13.0 PyTorch")
    return FA3_BUILDS[torch.version.cuda]


@cache
def load_fa3() -> ModuleType:
    """Load the matching stable-ABI build after verifying its file hashes."""
    build = fa3_build()
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            FA3_REPOSITORY,
            revision=FA3_REVISION,
            allow_patterns=[f"{build}/*"],
        )
    )
    root = snapshot / build
    metadata = json.loads((root / "metadata.json").read_text())
    digests = (
        FA3_CUDA13_DIGESTS
        if torch.version.cuda == "13.0"
        else {
            name: base64.b64decode(digest).hex()
            for name, digest in metadata["digest"]["files"].items()
        }
    )
    for name, expected in digests.items():
        with (root / name).open("rb") as handle:
            observed = hashlib.file_digest(handle, "sha256").hexdigest()
        if observed != expected:
            raise RuntimeError(f"FA3 kernel checksum mismatch: {name}")
    module_name = "_nano_flash_attn3_" + FA3_REVISION[:12]
    spec = importlib.util.spec_from_file_location(
        module_name, root / "__init__.py", submodule_search_locations=[str(root)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned FA3 kernel")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def attention_receipt(backend: str) -> dict[str, object]:
    if backend == "flash3":
        return {
            "implementation": "FlashAttention-3",
            "repository": FA3_REPOSITORY,
            "revision": FA3_REVISION,
            "build": fa3_build(),
            "operator": (
                "_flash_attn3_cuda_477ab85"
                if torch.version.cuda == "13.0"
                else "_flash_attn3_cuda_47281f5"
            ),
        }
    return {
        "implementation": "FlashAttention-2" if backend == "flash" else backend,
        "operator": "aten::_flash_attention_forward" if backend == "flash" else "sdpa",
    }


def prepare_attention(backend: str, device: torch.device) -> dict[str, object]:
    if backend == "flash3":
        if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 9:
            raise RuntimeError("FlashAttention-3 requires a Hopper GPU (compute capability 9)")
        load_fa3()
    return attention_receipt(backend)
