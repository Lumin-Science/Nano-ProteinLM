import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import yaml

from nanoprotein import evaluate
from nanoprotein.contact_parallel import contact_devices, run_contact_parallel
from nanoprotein.data import file_sha256
from nanoprotein.summarize_training_runs import summarize


class ParallelContactEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"test checkpoint")
        self.digest = file_sha256(self.checkpoint)
        self.parse_args = evaluate.parse_args
        self.args = evaluate.parse_args(
            [
                "--checkpoint",
                str(self.checkpoint),
                "--data-root",
                str(self.root / "data"),
                "--output-root",
                str(self.root / "evaluation"),
                "--external-src",
                str(self.root / "external"),
                "--contact-root",
                str(self.root / "contact"),
                "--run-contact",
            ]
        )
        self.args.output_root.mkdir()
        self.rows = sorted(
            [
                {
                    "chain_id": f"chain-{index}",
                    "precision_at_l": (index % 101) / 100,
                    "random_precision_at_l": 0.02,
                }
                for index in range(self.args.contact_chains)
            ],
            key=lambda row: hashlib.sha256(f"20260820:{row['chain_id']}".encode()).digest(),
        )
        self.commands = []
        self.processes = []
        self.load_model = Mock(return_value=(Mock(), {"training_seconds": 3600}))

    def fit_probe(self, command, **kwargs):
        self.load_model.assert_not_called()
        path = Path(command[command.index("--output") + 1])
        path.write_text(json.dumps({"checkpoint_sha256": self.digest}))

    def start_shard(self, command, **kwargs):
        self.load_model.assert_not_called()
        self.commands.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        worker = self.parse_args(command[3:])
        self.assertEqual(worker.contact_mode, "serial")
        self.assertTrue(worker.skip_validation_mlm)
        self.assertEqual(worker.contact_bootstrap, 0)
        rows = self.rows[worker.contact_shard_index :: worker.contact_shard_count]
        report = {
            "checkpoint_sha256": self.digest,
            "checkpoint_training_seconds": 3600,
            "peak_cuda_memory_bytes": 100,
            "contact": {
                "protocol": "esmc-paper-contact-lite-v1",
                "selection": "sha256_rank_uniform_without_replacement",
                "selection_seed": 20260820,
                "selection_total_chains": worker.contact_chains,
                "shard_index": worker.contact_shard_index,
                "shard_count": worker.contact_shard_count,
                "evaluation_chains": len(rows),
                "precision_at_l_uncertainty": {"replicates": 0},
                "selected_C": 1.0,
                "validation_trace": [{"C": 1.0}],
                "probe_receipt_sha256": file_sha256(worker.contact_probe_receipt),
                "rows": rows,
            },
        }
        evaluate.write_json(worker.output_root / "EVALUATION.json", report)
        process = Mock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        self.processes.append(process)
        return process

    def run_parallel(self):
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,5,7"}, clear=True),
            patch(
                "nanoprotein.contact_parallel.subprocess.run", side_effect=self.fit_probe
            ) as fit,
            patch(
                "nanoprotein.contact_parallel.subprocess.Popen", side_effect=self.start_shard
            ),
        ):
            result = run_contact_parallel(self.args, self.digest)
        return result, fit

    def test_standard_command_defaults_merge_full_population_and_preserve_summary(self):
        self.assertEqual(self.args.contact_mode, "parallel")
        self.assertEqual(self.args.contact_chains, 20775)
        self.assertEqual(self.args.contact_bootstrap, 5000)
        self.assertEqual(self.args.validation_context, 512)
        validation = {"sequence_mean_nll": 2.5, "sequences": 12288}
        with (
            patch("nanoprotein.evaluate.parse_args", return_value=self.args),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.manual_seed_all"),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch("nanoprotein.evaluate.load_checkpoint", self.load_model),
            patch("nanoprotein.evaluate.validation_mlm", return_value=validation),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3,5,7"}, clear=True),
            patch(
                "nanoprotein.contact_parallel.subprocess.run", side_effect=self.fit_probe
            ) as fit,
            patch(
                "nanoprotein.contact_parallel.subprocess.Popen", side_effect=self.start_shard
            ),
        ):
            evaluate.main()
        fit.assert_called_once()
        self.load_model.assert_called_once()
        self.assertEqual(len(self.commands), 32)
        self.assertEqual([gpu for _, gpu in self.commands], ["2", "3", "5", "7"] * 8)
        report = json.loads((self.args.output_root / "EVALUATION.json").read_text())
        contact = report["contact"]
        self.assertEqual(contact["rows"], self.rows)
        values = np.asarray([row["precision_at_l"] for row in self.rows])
        self.assertEqual(contact["precision_at_l"], float(values.mean()))
        self.assertEqual(
            contact["precision_at_l_uncertainty"],
            evaluate.bootstrap_mean_interval(values, replicates=5000, seed=20260820),
        )
        self.assertEqual(report["validation_mlm"], validation)
        runs = []
        for seed in (42, 43):
            root = self.root / f"seed-{seed}"
            root.mkdir()
            (root / "config.yaml").write_text(yaml.safe_dump({"seed": seed}))
            evaluate.write_json(
                root / "TRAINING_COMPLETE.json", {"final_checkpoint": {"sha256": self.digest}}
            )
            evaluate.write_json(root / "evaluation/EVALUATION.json", report)
            runs.append(root)
        self.assertEqual(
            summarize(runs, 12288)["metrics"]["p_at_l"]["mean"], float(values.mean())
        )

    def test_mlm_resume_rejects_sampled_receipts_and_ignores_batch_size(self):
        self.args.run_contact = False
        self.args.resume_components = True
        validation_path = self.args.output_root / "VALIDATION_MLM.json"
        with (
            patch("nanoprotein.evaluate.parse_args", return_value=self.args),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.manual_seed_all"),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch("nanoprotein.evaluate.load_checkpoint", self.load_model),
            patch("nanoprotein.evaluate.validation_mlm") as score,
        ):
            evaluate.write_json(validation_path, {"sequence_mean_nll": 2.5, "sequences": 32})
            with self.assertRaisesRegex(ValueError, "different protocol or settings"):
                evaluate.main()
            self.assertFalse((self.args.output_root / "EVALUATION.json").exists())
            sampled = {
                "protocol": "heldout-cluster-representative-mlm-v1",
                "settings": {
                    "sampling_seed": 20260821,
                    "context_length": 512,
                    "batch_size": 4,
                    "batches": 1024,
                },
                "sequences": 4096,
            }
            evaluate.write_json(validation_path, sampled)
            with self.assertRaisesRegex(ValueError, "different protocol or settings"):
                evaluate.main()
            validation = {
                "protocol": evaluate.VALIDATION_MLM_PROTOCOL,
                "settings": evaluate.validation_settings(512),
                "sequences": 12288,
                "sequence_mean_nll": 2.5,
                "batch_size": 64,
            }
            evaluate.write_json(validation_path, validation)
            self.args.validation_batch_size = 8
            evaluate.main()
        score.assert_not_called()
        report = json.loads((self.args.output_root / "EVALUATION.json").read_text())
        self.assertEqual(report["validation_mlm"]["sequences"], 12288)
        self.assertEqual(report["resumed_components"], ["validation_mlm"])

    def test_resume_reuses_probe_and_shards_and_rejects_changed_request(self):
        self.args.contact_chains = 9
        self.args.contact_bootstrap = 17
        self.rows = self.rows[:9]
        first, _ = self.run_parallel()
        self.args.resume_components = True
        self.commands.clear()
        resumed, fit = self.run_parallel()
        fit.assert_not_called()
        self.assertEqual(self.commands, [])
        self.assertEqual(first["contact"], resumed["contact"])
        self.args.contact_chains = 8
        with self.assertRaisesRegex(ValueError, "identical request"):
            self.run_parallel()

    def test_failure_does_not_publish_contact_and_stops_remaining_workers(self):
        self.args.contact_workers = 2

        def fail_first(command, **kwargs):
            process = self.start_shard(command, **kwargs)
            if len(self.processes) == 1:
                process.wait.return_value = 7
            else:
                process.poll.return_value = None
            return process

        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=True),
            patch("nanoprotein.contact_parallel.subprocess.run", side_effect=self.fit_probe),
            patch("nanoprotein.contact_parallel.subprocess.Popen", side_effect=fail_first),
            self.assertRaisesRegex(RuntimeError, "contact shard failed"),
        ):
            run_contact_parallel(self.args, self.digest)
        self.processes[1].terminate.assert_called_once()
        self.assertFalse((self.args.output_root / "CONTACT.json").exists())
        self.assertFalse((self.args.output_root / "EVALUATION.json").exists())

    def test_merger_rejects_duplicate_chains_and_different_probe(self):
        self.args.contact_workers = 2
        self.args.contact_chains = 9
        self.args.contact_bootstrap = 17
        self.rows = self.rows[:9]
        self.run_parallel()
        paths = sorted((self.args.output_root / "components").glob("*/EVALUATION.json"))
        report = json.loads(paths[1].read_text())
        report["contact"]["probe_receipt_sha256"] = "wrong-probe"
        evaluate.write_json(paths[1], report)
        with self.assertRaisesRegex(ValueError, "different fitted probe"):
            evaluate.merge_contact_evaluation(
                contact_paths=paths, expected_contact_chains=9, contact_bootstrap=17
            )
        report["contact"]["rows"][0] = self.rows[0]
        evaluate.write_json(paths[1], report)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            evaluate.merge_contact_evaluation(
                contact_paths=paths, expected_contact_chains=9, contact_bootstrap=17
            )

    def test_gpu_selection_preserves_visible_identifiers(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}, clear=True):
            self.assertEqual(contact_devices(None), ["GPU-a", "GPU-b"])
            self.assertEqual(contact_devices("3,5"), ["3", "5"])
        for value in ("", "-1", "0,0"):
            with self.assertRaises(ValueError):
                contact_devices(value)

    def test_explicit_serial_mode_uses_the_same_scorer_without_workers(self):
        self.args.contact_mode = "serial"
        self.args.skip_validation_mlm = True
        with (
            patch("nanoprotein.evaluate.parse_args", return_value=self.args),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.manual_seed_all"),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch("nanoprotein.evaluate.load_checkpoint", self.load_model),
            patch("nanoprotein.evaluate.run_contact_lite", return_value={"rows": []}) as score,
            patch("nanoprotein.contact_parallel.run_contact_parallel") as parallel,
        ):
            evaluate.main()
        parallel.assert_not_called()
        score.assert_called_once()
        self.assertEqual(score.call_args.kwargs["evaluation_chains"], 20775)
        self.assertEqual(score.call_args.kwargs["bootstrap"], 5000)

    def test_cache_preflight_runs_once_and_is_shared_by_workers(self):
        self.args.contact_workers = 2
        self.args.contact_chains = 9
        self.args.contact_bootstrap = 17
        self.rows = self.rows[:9]
        self.args.contact_scoring_cache_root = self.root / "static-cache"
        with patch("nanoprotein.contact_parallel.verify_contact_scoring_cache") as verify:
            self.run_parallel()
        verify.assert_called_once_with(
            cache_root=self.args.contact_scoring_cache_root,
            output=self.args.output_root / "CONTACT_SCORING_CACHE_PREFLIGHT.json",
        )
        for command, _gpu in self.commands:
            worker = self.parse_args(command[3:])
            self.assertEqual(
                worker.contact_scoring_cache_root, self.args.contact_scoring_cache_root
            )
            self.assertEqual(
                worker.contact_scoring_cache_preflight,
                self.args.output_root / "CONTACT_SCORING_CACHE_PREFLIGHT.json",
            )

    def test_task_entry_point_evaluates_one_run_on_the_training_gpus(self):
        repo = Path(__file__).resolve().parents[2]
        checkout = self.root / "checkout"
        (checkout / "tasks").mkdir(parents=True)
        for name in ("171m-validation-loss_ar.sh", "171m-p-at-l_ar.sh"):
            shutil.copyfile(repo / "tasks" / name, checkout / "tasks" / name)
        shutil.copyfile(repo / ".env.example", checkout / ".env.example")
        (checkout / "recipe.yaml").write_text("seed: 42\n")
        (self.root / "data/training").mkdir(parents=True)
        (self.root / "data/training/manifest.json").write_text("{}")
        binary = self.root / "bin"
        binary.mkdir()
        uv = binary / "uv"
        log = self.root / "commands.jsonl"
        uv.write_text(
            f"#!{sys.executable}\nimport json,os,sys\n"
            "with open(os.environ['COMMAND_LOG'], 'a') as output:\n"
            "    output.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if '-c' in sys.argv: print('flash3 989.5')\n"
        )
        uv.chmod(0o755)
        subprocess.run(
            ["bash", "tasks/171m-p-at-l_ar.sh", "recipe.yaml", "trial", "47"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{binary}:{os.environ['PATH']}",
                "CUDA_VISIBLE_DEVICES": "2,3,5,7",
                "EVAL_GPUS": "8,9",
                "DATA_ROOT": str(self.root / "data"),
                "NANOPROTEIN_STAGE_DIR": str(self.root / "local"),
                "OUTPUT_ROOT": str(self.root / "runs"),
                "COMMAND_LOG": str(log),
            },
        )
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        evaluations = [
            self.parse_args(command[command.index("nanoprotein.evaluate") + 1 :])
            for command in commands
            if "nanoprotein.evaluate" in command
        ]
        self.assertEqual(len(evaluations), 1)
        for args in evaluations:
            self.assertEqual(args.contact_mode, "parallel")
            self.assertEqual(args.contact_gpus, "2,3,5,7")
            self.assertEqual(args.contact_workers, 32)
            self.assertEqual(args.contact_chains, 20775)
            self.assertEqual(args.contact_bootstrap, 5000)
            self.assertEqual(args.checkpoint.parent.name, "trial")
        training = next(command for command in commands if "nanoprotein.train" in command)
        self.assertEqual(training[training.index("--seed") + 1], "47")
        self.assertFalse(any("nanoprotein.summarize_training_runs" in row for row in commands))


if __name__ == "__main__":
    unittest.main()
