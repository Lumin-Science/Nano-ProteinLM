"""Exercise shell control flow without CUDA; resolve recipes with the real training CLI."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MOCK_UV = r"""#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['LAUNCH_TEST_LOG']).open('a') as log:
    log.write(json.dumps(args) + '\n')
if args[0] == 'sync':
    sys.exit(0)
assert args[:4] == ['run', '--frozen', '--no-dev', 'python'], args
args = args[4:]
if args[:2] == ['-m', 'nanoprotein.sharded_data']:
    root = Path(args[args.index('--output-root') + 1])
    if root.exists():
        assert (root / 'manifest.json').is_file()
        assert (root / 'CORPUS_VERIFICATION.json').is_file()
        sys.exit(0)
    root.mkdir(parents=True)
    for name in ('manifest.json', 'CORPUS_VERIFICATION.json'):
        (root / name).write_text('{}')
elif args[:2] == ['-m', 'nanoprotein.setup_evaluation']:
    if os.environ.get('LAUNCH_TEST_EVALUATION_FAIL'):
        sys.exit(1)
    root = Path(args[args.index('--data-root') + 1]) / 'evaluation'
    (root / 'contact').mkdir(parents=True, exist_ok=True)
    (root / 'source').mkdir(exist_ok=True)
elif args[0] == '-':
    # Receipt validation is exercised by the real trainer, not this shell-flow test.
    assert Path(args[1], 'CORPUS_VERIFICATION.json').is_file()
    sys.stdin.read()
elif args[:2] == ['-m', 'nanoprotein.check_environment']:
    if os.environ.get('LAUNCH_TEST_QUALIFY_FAIL'):
        sys.exit(1)
    Path(args[args.index('--output') + 1]).write_text('{}')
elif args[:2] == ['-m', 'torch.distributed.run']:
    train_args = args[args.index('nanoprotein.train') + 1:]
    out = Path(train_args[train_args.index('--output-root') + 1])
    resolved = subprocess.check_output(
        [os.environ['LAUNCH_TEST_PYTHON'], '-m', 'nanoprotein.train',
         *train_args, '--print-config'], text=True)
    (out / 'resolved-test.yaml').write_text(resolved)
    for name in ('run_contract.json', 'TRAINING_COMPLETE.json', 'checkpoint-final.pt'):
        (out / name).write_text('test artifact')
else:
    sys.exit(subprocess.call([os.environ['LAUNCH_TEST_PYTHON'], *args]))
"""


class LaunchScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "runs").mkdir()
        (self.root / "configs").mkdir()
        for name in ("setup.sh", "speedrun.sh"):
            shutil.copy(ROOT / "runs" / name, self.root / "runs" / name)
        shutil.copy(ROOT / ".env.example", self.root / ".env.example")
        shutil.copy(ROOT / "configs/default.yaml", self.root / "configs/default.yaml")
        self.data = self.root / "data with spaces"
        self.output = self.root / "outputs with spaces"
        (self.root / ".env").write_text(
            f"DATA_ROOT={shlex.quote(str(self.data))}\n"
            f"OUTPUT_ROOT={shlex.quote(str(self.output))}\n"
        )
        uv = self.root / "mock-uv"
        uv.write_text(MOCK_UV)
        uv.chmod(0o755)
        self.log = self.root / "commands.jsonl"
        self.env = {
            **os.environ,
            "UV_BIN": str(uv),
            "PYTHONPATH": str(ROOT / "src"),
            "LAUNCH_TEST_LOG": str(self.log),
            "LAUNCH_TEST_PYTHON": sys.executable,
        }
        for name in ("DATA_ROOT", "OUTPUT_ROOT"):
            self.env.pop(name, None)

    def run_script(self, name, *args, success=True):
        result = subprocess.run(
            ["bash", str(self.root / "runs" / name), *args],
            cwd=self.root.parent,
            env=self.env,
            capture_output=True,
            text=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_speedrun_prepares_data_and_resolves_real_default(self):
        import yaml

        self.run_script("speedrun.sh")
        output = self.output / "setting3-100k"
        config = yaml.safe_load((output / "resolved-test.yaml").read_text())
        self.assertEqual(config["optimizer"], "muon")
        self.assertEqual(config["max_steps"], 100000)
        self.assertEqual(config["walltime_seconds"], 57600)
        self.assertEqual(config["attention_backend"], "flash3")
        self.assertEqual(config["stages"][0]["gradient_accumulation"], 4)
        self.assertTrue((output / "checkpoint-final.pt").is_file())
        before = len(self.commands())
        self.run_script("speedrun.sh", success=False)
        self.assertFalse(
            any("torch.distributed.run" in row for row in self.commands()[before:])
        )

    def test_setup_reuses_data_and_does_not_launch_training(self):
        self.run_script("setup.sh")
        self.run_script("setup.sh")
        commands = self.commands()
        self.assertEqual(sum("nanoprotein.sharded_data" in row for row in commands), 2)
        self.assertFalse(any("torch.distributed.run" in row for row in commands))
        self.assertEqual(sum("nanoprotein.setup_evaluation" in row for row in commands), 2)
        download = next(row for row in commands if "nanoprotein.sharded_data" in row)
        self.assertEqual(download[download.index("--training-shards") + 1], "30")
        self.assertTrue((self.data / "evaluation/contact").is_dir())

    def test_incomplete_data_and_failed_qualification_block_training(self):
        (self.data / "training").mkdir(parents=True)
        self.run_script("speedrun.sh", success=False)
        self.assertFalse(any("torch.distributed.run" in row for row in self.commands()))
        (self.data / "training").rmdir()
        self.env["LAUNCH_TEST_QUALIFY_FAIL"] = "1"
        self.run_script("speedrun.sh", success=False)
        self.assertFalse(any("torch.distributed.run" in row for row in self.commands()))

    def test_cli_overrides_reach_qualification_and_training(self):
        import yaml

        self.run_script(
            "speedrun.sh",
            "configs/default.yaml",
            "custom-run",
            "--attention-backend",
            "flash",
            "--seed",
            "43",
            "--max-steps",
            "none",
            "--walltime-seconds",
            "3600",
        )
        config = yaml.safe_load((self.output / "custom-run/resolved-test.yaml").read_text())
        self.assertEqual((config["seed"], config["attention_backend"]), (43, "flash"))
        self.assertIsNone(config["max_steps"])
        self.assertEqual(config["walltime_seconds"], 3600)
        check = next(row for row in self.commands() if "nanoprotein.check_environment" in row)
        self.assertEqual(check[check.index("--attention-backend") + 1], "flash")

    def test_setup_custom_shard_count_and_usage(self):
        self.run_script("setup.sh", "--training-shards", "105")
        download = next(row for row in self.commands() if "nanoprotein.sharded_data" in row)
        self.assertEqual(download[download.index("--training-shards") + 1], "105")
        self.run_script("setup.sh", "--training-shards", success=False)
        help_result = self.run_script("setup.sh", "--help")
        self.assertIn("565", help_result.stdout)

    def test_speedrun_preserves_existing_seven_shard_selection(self):
        self.run_script("setup.sh", "--training-shards", "7")
        (self.data / "training/download-plan.json").write_text(
            json.dumps(
                {
                    "sources": {
                        "uniref90": {"train": [0] * 3},
                        "mgnify": {"train": [0]},
                        "omg_img": {"train": [0] * 3},
                    }
                }
            )
        )
        self.run_script("speedrun.sh")
        downloads = [row for row in self.commands() if "nanoprotein.sharded_data" in row]
        self.assertEqual(downloads[-1][downloads[-1].index("--training-shards") + 1], "7")

    def test_incomplete_evaluation_blocks_training(self):
        self.env["LAUNCH_TEST_EVALUATION_FAIL"] = "1"
        self.run_script("speedrun.sh", success=False)
        self.assertFalse(any("torch.distributed.run" in row for row in self.commands()))
