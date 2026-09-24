"""Provision a fresh, single-branch AutoResearch workspace before agent access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/Lumin-Science/Nano-ProteinLM.git"
UV_INSTALLER = "https://astral.sh/uv/0.11.31/install.sh"
UV_INSTALLER_SHA256 = "bd9a2739c49251c71fd3706ac00b1bb8582ea138433c6e52840de4aba646e46a"


def git_environment() -> dict[str, str]:
    context = {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
    }
    return {key: value for key, value in os.environ.items() if key not in context}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, env=git_environment()
    ).strip()


def verify_release(root: Path) -> str:
    if not (root / ".git").is_dir() or (root / ".git/objects/info/alternates").exists():
        raise ValueError("a standalone clone without shared Git objects is required")
    if git(root, "branch", "--show-current") != "autoresearch":
        raise ValueError("expected the released autoresearch branch")
    revision = git(root, "rev-parse", "HEAD")
    headers = git(root, "cat-file", "-p", "HEAD").split("\n\n", 1)[0]
    if any(line.startswith("parent ") for line in headers.splitlines()):
        raise ValueError("the published starter must be an independent root commit")
    if git(root, "rev-list", "--all", "--count") != "1":
        raise ValueError("unexpected additional Git history")
    if git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("release files must be unchanged before provisioning")
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text())
    tracked = set(git(root, "ls-files").splitlines())
    if tracked != set(manifest["files"]) | {"RELEASE_MANIFEST.json"}:
        raise ValueError("tracked files do not match the release manifest")
    for name, expected in manifest["files"].items():
        path = root / name
        path.resolve().relative_to(root.resolve())
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"release checksum mismatch: {name}")
    return revision


def create_workspace(destination: Path, repository: str, revision: str | None) -> str:
    if destination.exists():
        raise FileExistsError(f"destination exists; choose a fresh directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_git = subprocess.run(
        ["git", "-C", str(destination.parent), "rev-parse", "--show-toplevel"],
        capture_output=True,
        env=git_environment(),
    )
    if parent_git.returncode == 0:
        raise ValueError("create the agent workspace outside every existing Git checkout")
    subprocess.run(
        [
            "git",
            "clone",
            "--no-local",
            "--depth",
            "1",
            "--single-branch",
            "--no-tags",
            "--branch",
            "autoresearch",
            "--",
            repository,
            str(destination),
        ],
        check=True,
        env=git_environment(),
    )
    observed = verify_release(destination)
    if revision and observed != revision:
        raise ValueError(f"release revision differs: expected {revision}, found {observed}")
    git(destination, "remote", "remove", "origin")
    if git(destination, "for-each-ref", "--format=%(refname)") != "refs/heads/autoresearch":
        raise ValueError("unexpected references remain in the fresh workspace")
    state = destination / ".autoresearch"
    state.mkdir()
    (state / "PROVISIONING.json").write_text(
        json.dumps({"release_commit": observed, "status": "cloned"}, indent=2) + "\n"
    )
    return observed


def provision(root: Path, training_shards: int) -> None:
    if sys.platform != "linux":
        raise RuntimeError("environment/data preparation requires Linux with four CUDA GPUs")
    env = os.environ.copy()
    env["DATA_ROOT"] = str(root / "data")
    env["OUTPUT_ROOT"] = str(root / "outputs")
    env["UV_CACHE_DIR"] = str(root / ".bootstrap/uv-cache")
    env["UV_PYTHON_INSTALL_DIR"] = str(root / ".bootstrap/python")
    env["UV_MANAGED_PYTHON"] = "1"
    env.pop("UV_NO_MANAGED_PYTHON", None)
    env["HF_HOME"] = str(root / ".bootstrap/huggingface")
    uv = compatible_uv(root, env)
    env["UV_BIN"] = uv
    # uv installs the project's pinned Python when it is not already available.
    subprocess.run([uv, "sync", "--frozen", "--no-dev"], cwd=root, env=env, check=True)
    subprocess.run(
        [
            uv,
            "run",
            "--frozen",
            "--no-dev",
            "python",
            "-m",
            "nanoprotein.check_environment",
            "--require-gpus",
            "4",
            "--autoresearch",
            "--attention-backend",
            "auto",
            "--output",
            str(root / ".autoresearch/ENVIRONMENT.json"),
        ],
        cwd=root,
        env=env,
        check=True,
    )
    subprocess.run(
        ["bash", "runs/setup.sh", "--training-shards", str(training_shards)],
        cwd=root,
        env=env,
        check=True,
    )
    # Persist only the permitted data/output roots. Kernel and uv caches stay local.
    env_path = root / ".env"
    if not env_path.exists():
        env_path.write_text(
            "DATA_ROOT=" + shlex.quote(str(root / "data")) + "\n"
            "OUTPUT_ROOT=" + shlex.quote(str(root / "outputs")) + "\n"
        )
    receipt_path = root / ".autoresearch/PROVISIONING.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(status="ready", training_shards=training_shards)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


def compatible_uv(root: Path, env: dict[str, str]) -> str:
    local_uv = root / ".bootstrap/bin/uv"
    uv = env.get("UV_BIN") or (str(local_uv) if local_uv.is_file() else shutil.which("uv"))
    compatible = False
    if uv:
        try:
            version = subprocess.check_output([uv, "--version"], text=True).split()[1]
            compatible = (0, 11, 31) <= tuple(map(int, version.split("."))) < (0, 12)
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
            pass
    if not compatible:
        # Cluster Python installations may reject binary wheels and build uv from source.
        # The pinned standalone installer supplies its own platform-binary checksums.
        installer = root / ".bootstrap/install-uv-0.11.31.sh"
        installer.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--max-time",
                "60",
                UV_INSTALLER,
                "--output",
                str(installer),
            ],
            check=True,
        )
        script = installer.read_bytes()
        if hashlib.sha256(script).hexdigest() != UV_INSTALLER_SHA256:
            raise ValueError("uv installer checksum mismatch")
        installer_env = dict(
            env, UV_UNMANAGED_INSTALL=str(local_uv.parent), UV_NO_MODIFY_PATH="1"
        )
        subprocess.run(["sh", str(installer)], env=installer_env, check=True)
        uv = str(local_uv)
    return uv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination", type=Path, help="new directory outside existing checkouts"
    )
    parser.add_argument("--revision", help="require this exact published commit")
    parser.add_argument(
        "--repository", default=REPOSITORY, help="release Git URL; use SSH for private access"
    )
    parser.add_argument("--training-shards", type=int, default=30)
    parser.add_argument(
        "--clone-only", action="store_true", help="defer environment/data setup"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume this script's unchanged clone"
    )
    args = parser.parse_args()
    if not 3 <= args.training_shards <= 565:
        parser.error("--training-shards must be between 3 and 565")
    if args.revision and (
        len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision)
    ):
        parser.error("--revision must be a full lowercase Git commit SHA")
    root = args.destination.expanduser().resolve()
    if args.resume:
        saved = json.loads((root / ".autoresearch/PROVISIONING.json").read_text())
        revision = verify_release(root)
        if revision != saved["release_commit"] or (args.revision and revision != args.revision):
            raise ValueError("the workspace no longer matches its provisioned release")
        if git(root, "remote"):
            raise ValueError("remove external remotes before resuming preparation")
    else:
        revision = create_workspace(root, args.repository, args.revision)
    if not args.clone_only:
        provision(root, args.training_shards)
    print(
        json.dumps(
            {
                "workspace": str(root),
                "release_commit": revision,
                "status": "cloned" if args.clone_only else "ready",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
