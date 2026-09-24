"""Exercise the README's evaluation command without downloading data or using GPUs."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from nanoprotein.evaluate import parse_args

ROOT = Path(__file__).resolve().parents[2]


class SpeedrunEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "runs").mkdir()
        shutil.copy(ROOT / "runs/speedrun.sh", self.root / "runs/speedrun.sh")
        shutil.copy(ROOT / ".env.example", self.root / ".env.example")
        # Evaluation must never enter setup or training.
        (self.root / "runs/setup.sh").write_text("exit 97\n")
        self.capture = self.root / "command.json"
        self.uv = self.root / "fake-uv"
        self.uv.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
            "sys.exit(int(os.environ.get('EVALUATION_EXIT_CODE', '0')))\n"
        )
        self.uv.chmod(0o755)
        self.env = {
            "PATH": os.environ["PATH"],
            "UV_BIN": str(self.uv),
            "CAPTURE": str(self.capture),
        }

    def run_command(self, *args):
        return subprocess.run(
            ["bash", str(self.root / "runs/speedrun.sh"), "--evaluate", *args],
            cwd=self.root.parent,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def captured_args(self):
        command = json.loads(self.capture.read_text())
        self.assertEqual(
            command[:5], ["run", "--frozen", "python", "-m", "nanoprotein.evaluate"]
        )
        return parse_args(command[5:])

    def test_readme_command_uses_full_parallel_evaluation(self):
        result = self.run_command("default-100k")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.captured_args()
        self.assertEqual(
            args.checkpoint, self.root / "outputs/default-100k/checkpoint-final.pt"
        )
        self.assertEqual(args.output_root, self.root / "outputs/default-100k/evaluation")
        self.assertEqual(args.data_root, self.root / "data/training")
        self.assertEqual(args.contact_root, self.root / "data/evaluation/contact")
        self.assertEqual(args.external_src, self.root / "data/evaluation/source")
        self.assertEqual(args.validation_context, 512)
        self.assertTrue(args.run_contact)
        self.assertEqual(args.contact_chains, 20775)
        self.assertEqual(args.contact_bootstrap, 5000)
        self.assertEqual(args.contact_mode, "parallel")

    def test_custom_roots_options_and_evaluation_failure_are_preserved(self):
        (self.root / ".env").write_text('DATA_ROOT="prepared data"\nOUTPUT_ROOT="saved runs"\n')
        self.env["EVALUATION_EXIT_CODE"] = "23"
        result = self.run_command(
            "trial-007", "--contact-mode", "serial", "--validation-batch-size", "8"
        )
        self.assertEqual(result.returncode, 23, result.stderr)
        args = self.captured_args()
        self.assertEqual(args.checkpoint, Path("saved runs/trial-007/checkpoint-final.pt"))
        self.assertEqual(args.data_root, Path("prepared data/training"))
        self.assertEqual(args.contact_mode, "serial")
        self.assertEqual(args.validation_batch_size, 8)

    def test_global_checkpoint_helper_uses_the_same_mlm_settings(self):
        # Stop after capturing the real evaluator command, before the historical contact helper.
        py = self.root / "fake-python"
        py.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "if 'nanoprotein.evaluate' in sys.argv:\n"
            "    pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
            "    sys.exit(17)\n"
        )
        py.chmod(0o755)
        result = subprocess.run(
            [
                "bash",
                str(ROOT / "runs/evaluate_global_checkpoint.sh"),
                str(self.root / "checkpoint.pt"),
                str(self.root / "eval"),
            ],
            env={
                **self.env,
                "PAIR_ROOT": str(self.root),
                "PAIR_REPO": str(ROOT),
                "TRAIN_PYTHON": str(py),
                "DATA_ROOT": str(self.root / "data"),
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 17, result.stderr)
        command = json.loads(self.capture.read_text())
        args = parse_args(command[2:])
        self.assertEqual(args.validation_context, 512)

    def test_omitted_run_name_and_invalid_run_name(self):
        result = self.run_command("--contact-workers", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.captured_args().contact_workers, 2)
        self.capture.unlink()
        result = self.run_command("../outside")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
