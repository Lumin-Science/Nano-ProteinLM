"""Verify preparation cannot inherit the research checkout or overwrite existing work."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "prepare", ROOT / ".dev/scripts/prepare_autoresearch.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


class AutoResearchPreparationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        git(self.upstream, "init", "-b", "main")
        git(self.upstream, "config", "user.name", "Fixture")
        git(self.upstream, "config", "user.email", "fixture@example.invalid")
        (self.upstream / "prior.txt").write_text("excluded findings\n")
        git(self.upstream, "add", ".")
        git(self.upstream, "commit", "-m", "private comparison")
        self.excluded = git(self.upstream, "rev-parse", "HEAD")
        git(self.upstream, "checkout", "--orphan", "autoresearch")
        git(self.upstream, "rm", "-rf", ".")
        (self.upstream / "README.md").write_text("plain starting implementation\n")
        (self.upstream / ".gitignore").write_text(".autoresearch/\n.bootstrap/\n.env\n")
        files = {
            n: hashlib.sha256((self.upstream / n).read_bytes()).hexdigest()
            for n in ("README.md", ".gitignore")
        }
        (self.upstream / "RELEASE_MANIFEST.json").write_text(json.dumps({"files": files}))
        git(self.upstream, "add", ".")
        git(self.upstream, "commit", "-m", "plain release")
        self.revision = git(self.upstream, "rev-parse", "HEAD")

    def test_only_clean_root_objects_arrive_and_remote_is_removed(self):
        dest = self.root / "agent with spaces"
        self.assertEqual(
            prepare.create_workspace(dest, str(self.upstream), self.revision), self.revision
        )
        self.assertEqual(git(dest, "branch", "--format=%(refname)"), "refs/heads/autoresearch")
        self.assertEqual(git(dest, "remote"), "")
        self.assertEqual(prepare.verify_release(dest), self.revision)
        result = subprocess.run(
            ["git", "-C", str(dest), "cat-file", "-e", self.excluded], capture_output=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((dest / "prior.txt").exists())
        # The source repository and excluded branch stay intact.
        self.assertEqual(git(self.upstream, "rev-parse", "main"), self.excluded)

    def test_cli_accepts_an_authenticated_repository_alternative(self):
        dest = self.root / "cli-agent"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / ".dev/scripts/prepare_autoresearch.py"),
                str(dest),
                "--repository",
                str(self.upstream),
                "--revision",
                self.revision,
                "--clone-only",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout)["release_commit"], self.revision)
        self.assertEqual(git(dest, "remote"), "")

    def test_inherited_git_context_cannot_redirect_workspace_operations(self):
        dest = self.root / "isolated"
        with patch.dict(
            os.environ,
            {"GIT_DIR": str(self.upstream / ".git"), "GIT_WORK_TREE": str(self.upstream)},
        ):
            self.assertEqual(
                prepare.create_workspace(dest, str(self.upstream), None), self.revision
            )
        self.assertEqual(git(dest, "remote"), "")
        self.assertEqual(git(self.upstream, "rev-parse", "main"), self.excluded)

    def test_existing_or_nested_destinations_are_refused(self):
        for dest in (self.upstream, self.upstream / "nested"):
            with self.assertRaises((FileExistsError, ValueError)):
                prepare.create_workspace(dest, str(self.upstream), None)
        self.assertEqual(git(self.upstream, "status", "--porcelain"), "")

    def test_wrong_pin_dirty_files_and_nonroot_release_are_refused(self):
        with self.assertRaisesRegex(ValueError, "revision differs"):
            prepare.create_workspace(self.root / "wrong-pin", str(self.upstream), "0" * 40)
        dest = self.root / "agent"
        prepare.create_workspace(dest, str(self.upstream), None)
        (dest / "README.md").write_text("changed files")
        with self.assertRaisesRegex(ValueError, "unchanged"):
            prepare.verify_release(dest)
        git(self.upstream, "commit", "--allow-empty", "-m", "extra history")
        with self.assertRaisesRegex(ValueError, "root commit"):
            prepare.create_workspace(self.root / "nonroot", str(self.upstream), None)

    def test_failed_environment_setup_does_not_mark_workspace_ready(self):
        dest = self.root / "agent"
        prepare.create_workspace(dest, str(self.upstream), None)
        with (
            patch.object(prepare.sys, "platform", "linux"),
            patch.object(prepare, "compatible_uv", return_value="uv"),
            patch.object(
                prepare.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "uv")
            ),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                prepare.provision(dest, 30)
        state = json.loads((dest / ".autoresearch/PROVISIONING.json").read_text())
        self.assertEqual(state["status"], "cloned")

    def test_old_uv_gets_a_workspace_install_without_updating_global_tool(self):
        script = b"#!/bin/sh\n"
        installer = self.root / ".bootstrap/install-uv-0.11.31.sh"
        installer.parent.mkdir()
        installer.write_bytes(script)
        with (
            patch.object(prepare.subprocess, "check_output", return_value="uv 0.9.26\n"),
            patch.object(prepare, "UV_INSTALLER_SHA256", hashlib.sha256(script).hexdigest()),
            patch.object(prepare.subprocess, "run") as run,
        ):
            selected = prepare.compatible_uv(self.root, {"UV_BIN": "/usr/bin/uv"})
        self.assertEqual(selected, str(self.root / ".bootstrap/bin/uv"))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args.kwargs["env"]["UV_UNMANAGED_INSTALL"],
            str(self.root / ".bootstrap/bin"),
        )
        self.assertEqual(run.call_args.kwargs["env"]["UV_NO_MODIFY_PATH"], "1")

    def test_changed_uv_installer_is_not_executed(self):
        installer = self.root / ".bootstrap/install-uv-0.11.31.sh"
        installer.parent.mkdir()
        installer.write_bytes(b"changed")
        with (
            patch.object(prepare.shutil, "which", return_value=None),
            patch.object(prepare.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                prepare.compatible_uv(self.root, {})
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], "curl")

    def test_compatible_uv_is_reused(self):
        with (
            patch.object(prepare.subprocess, "check_output", return_value="uv 0.11.31\n"),
            patch.object(prepare.subprocess, "run") as run,
        ):
            self.assertEqual(
                prepare.compatible_uv(self.root, {"UV_BIN": "/usr/bin/uv"}), "/usr/bin/uv"
            )
        run.assert_not_called()
