#!/usr/bin/env bash
# Source inside a provisioned workspace to use its isolated dependency caches.
if [[ -d .bootstrap ]]; then
  export PATH="$PWD/.bootstrap/bin:$PWD/.bootstrap/venv/bin:$PATH"
  export UV_CACHE_DIR="$PWD/.bootstrap/uv-cache"
  export UV_PYTHON_INSTALL_DIR="$PWD/.bootstrap/python"
  export UV_MANAGED_PYTHON=1
  export UV_NO_DEV="${UV_NO_DEV:-1}"
  export HF_HOME="$PWD/.bootstrap/huggingface"
fi
