"""Exercise task boundaries without GPUs, using the real trainer's config resolver."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "171m-validation-loss_ar.sh"
MOCK_UV = r"""
import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['TASK_TEST_LOG']).open('a') as log:
    log.write(json.dumps(args) + '\n')
# Measurements must not fetch dev tools.
assert args[:4] == ['run', '--frozen', '--no-dev', 'python'], args
args = args[4:]
if args[:2] == ['-m', 'nanoprotein.check_environment']:
    if os.environ.get('TASK_TEST_QUALIFY_FAIL'):
        sys.exit(1)
    from nanoprotein.check_environment import autoresearch_hardware
    l40s = bool(os.environ.get('TASK_TEST_L40'))
    hardware = autoresearch_hardware([dict(
        name='NVIDIA L40S' if l40s else 'NVIDIA H100 80GB HBM3',
        memory_bytes=(48 if l40s else 80) * 1024**3,
        capability=(8, 9) if l40s else (9, 0)) for _ in range(4)])
    Path(args[args.index('--output') + 1]).write_text(json.dumps(hardware))
elif args[0] == '-c':
    sys.exit(subprocess.call([sys.executable, *args]))
elif args[:2] == ['-m', 'torch.distributed.run']:
    if os.environ.get('TASK_TEST_TRAIN_FAIL'):
        sys.exit(1)
    train_args = args[args.index('nanoprotein.train') + 1:]
    out = Path(train_args[train_args.index('--output-root') + 1])
    data = Path(train_args[train_args.index('--data-root') + 1])
    assert (data / 'uniref90/train/tokens.bin').read_text() == 'tokens', data
    resolved = subprocess.check_output(
        [sys.executable, '-m', 'nanoprotein.train', *train_args, '--print-config'], text=True)
    (out / 'resolved-test.yaml').write_text(resolved)
    (out / 'checkpoint-final.pt').write_text('test checkpoint')
elif args[:2] == ['-m', 'nanoprotein.evaluate']:
    assert Path(args[args.index('--checkpoint') + 1]).is_file()
else:
    raise AssertionError(args)
"""


class TaskMeasurementTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "tasks").mkdir()
        (self.root / "scripts").mkdir()
        for name in (SCRIPT, "171m-p-at-l_ar.sh"):
            shutil.copyfile(ROOT / "tasks" / name, self.root / "tasks" / name)
        shutil.copyfile(ROOT / ".env.example", self.root / ".env.example")
        self.data = self.root / "data with spaces"
        self.output = self.root / "output with spaces"
        self.stage = self.root / "local disk"
        (self.data / "training/uniref90/train").mkdir(parents=True)
        (self.data / "training/manifest.json").write_text("manifest v1")
        (self.data / "training/uniref90/train/tokens.bin").write_text("tokens")
        (self.root / ".env").write_text(
            f"DATA_ROOT={shlex.quote(str(self.data))}\n"
            f"OUTPUT_ROOT={shlex.quote(str(self.output))}\n"
        )
        self.recipe = yaml.safe_load((ROOT / "configs/autoresearch/esmc-171m.yaml").read_text())
        self.recipe.update(
            max_steps=12,
            max_model_tokens=12345,
            schedule_steps=12,
            walltime_seconds=3600,
            attention_backend="flash",
            checkpoint_interval=10,
            periodic_evaluation_interval=10,
        )
        self.recipe["stages"][0]["micro_batch_size"] = 32
        self.recipe["stages"][0]["gradient_accumulation"] = 2
        (self.root / "recipe.yaml").write_text(yaml.safe_dump(self.recipe))
        uv = self.root / "mock uv"
        uv.write_text(f"#!{sys.executable}\n" + MOCK_UV)
        uv.chmod(0o755)
        self.log = self.root / "commands.jsonl"
        self.env = {
            **os.environ,
            "UV_BIN": str(uv),
            "PYTHONPATH": str(ROOT / "src"),
            "TASK_TEST_LOG": str(self.log),
            "CUDA_VISIBLE_DEVICES": "2,3,5,7",
            "NANOPROTEIN_STAGE_DIR": str(self.stage),
        }

    def run_script(self, *args, script=SCRIPT, success=True):
        result = subprocess.run(
            ["bash", str(self.root / "tasks" / script), *args],
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
        return [
            row
            for line in self.log.read_text().splitlines()
            if (row := json.loads(line))[4] == "-m"
        ]

    def test_one_caller_seed_and_fixed_budget_preserve_mutable_recipe_settings(self):
        self.run_script("recipe.yaml", "trial", "47")
        run = self.output / "trial"
        config = yaml.safe_load((run / "resolved-test.yaml").read_text())
        self.assertEqual(config["seed"], 47)
        self.assertEqual(config["walltime_seconds"], 1200)
        self.assertEqual(config["warmup_steps"], 500)
        self.assertEqual(config["attention_backend"], "flash3")
        for key in ("max_steps", "max_model_tokens", "schedule_steps"):
            self.assertIsNone(config[key])
        self.assertEqual(config["checkpoint_interval"], 0)
        self.assertEqual(config["periodic_evaluation_interval"], 0)
        self.assertEqual(config["prefetch_batches"], 2)
        self.assertEqual(config["stages"], self.recipe["stages"])
        self.assertEqual(config["learning_rate"], self.recipe["learning_rate"])
        self.assertEqual(
            (run / "recipe.yaml").read_bytes(), (self.root / "recipe.yaml").read_bytes()
        )
        commands = self.commands()
        self.assertEqual(len(commands), 3)
        self.assertIn("nanoprotein.check_environment", commands[0])
        self.assertIn("--autoresearch", commands[0])
        self.assertEqual(commands[0][commands[0].index("--attention-backend") + 1], "auto")
        self.assertIn("--nproc-per-node=4", commands[1])
        self.assertIn("nanoprotein.evaluate", commands[2])
        (staged,) = self.stage.glob("nanoprotein-*/training")
        for command in commands[1:]:
            self.assertEqual(command[command.index("--data-root") + 1], str(staged))
        self.assertEqual(commands[2][commands[2].index("--validation-context") + 1], "512")
        self.assertNotIn("--run-contact", commands[2])
        self.run_script("recipe.yaml", "trial", "48", success=False)
        self.assertEqual(self.commands(), commands)

    def test_local_data_copy_is_reused_until_the_manifest_changes(self):
        self.run_script("recipe.yaml", "first", "42")
        (staged,) = self.stage.glob("nanoprotein-*/training")
        marker = staged / "uniref90/train/marker"
        marker.write_text("kept")
        self.run_script("recipe.yaml", "second", "42")
        self.assertTrue(marker.exists())
        (self.data / "training/manifest.json").write_text("manifest v2")
        self.run_script("recipe.yaml", "third", "42")
        self.assertFalse(marker.exists())
        self.assertEqual((staged / "manifest.json").read_text(), "manifest v2")
        self.assertEqual(list(staged.parent.iterdir()), [staged])

    def test_l40s_measurement_uses_one_hour_and_qualified_backend(self):
        self.env["TASK_TEST_L40"] = "1"
        self.run_script("recipe.yaml", "l40-trial", "47")
        config = yaml.safe_load((self.output / "l40-trial/resolved-test.yaml").read_text())
        self.assertEqual(config["attention_backend"], "flash")
        self.assertEqual(config["peak_bf16_tflops_per_gpu"], 362.05)
        self.assertEqual(config["walltime_seconds"], 3600)

    def test_contact_entrypoint_uses_the_same_hardware_budget(self):
        self.env["TASK_TEST_L40"] = "1"
        self.run_script("recipe.yaml", "contact-trial", "47", script="171m-p-at-l_ar.sh")
        config = yaml.safe_load((self.output / "contact-trial/resolved-test.yaml").read_text())
        self.assertEqual(config["walltime_seconds"], 3600)
        self.assertEqual(config["attention_backend"], "flash")
        evaluation = self.commands()[-1]
        self.assertIn("nanoprotein.evaluate", evaluation)
        self.assertIn("--run-contact", evaluation)
        for option, value in (
            ("--validation-context", "512"),
            ("--contact-chains", "20775"),
            ("--contact-bootstrap", "5000"),
            ("--contact-gpus", "2,3,5,7"),
        ):
            self.assertEqual(evaluation[evaluation.index(option) + 1], value)

    def test_rejects_missing_invalid_arguments_and_wrong_gpu_count_before_compute(self):
        for args in (
            ("recipe.yaml", "trial"),
            ("recipe.yaml", "trial", "-1"),
            ("recipe.yaml", "trial", "x"),
            ("recipe.yaml", "../escape", "42"),
            ("missing.yaml", "trial", "42"),
        ):
            with self.subTest(args=args):
                self.run_script(*args, success=False)
        self.env["CUDA_VISIBLE_DEVICES"] = "0,1"
        self.run_script("recipe.yaml", "trial", "42", success=False)
        self.assertFalse(self.log.exists())
        self.assertIn("SEED", self.run_script("--help").stdout)

    def test_failed_qualification_or_training_never_evaluates(self):
        self.env["TASK_TEST_QUALIFY_FAIL"] = "1"
        self.run_script("recipe.yaml", "bad-environment", "42", success=False)
        self.assertEqual(len(self.commands()), 1)
        del self.env["TASK_TEST_QUALIFY_FAIL"]
        self.env["TASK_TEST_TRAIN_FAIL"] = "1"
        self.run_script("recipe.yaml", "bad-training", "42", success=False)
        self.assertEqual(len(self.commands()), 3)
        self.assertFalse(any("nanoprotein.evaluate" in row for row in self.commands()))
