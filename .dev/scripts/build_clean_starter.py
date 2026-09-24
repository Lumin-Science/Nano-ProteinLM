"""Maintainer-only export: build a history-free ESMC benchmark starting point.

This builder and its removal rules must never be distributed to benchmark agents.
It reads the working tree so owner edits are included without touching research files.
"""

from __future__ import annotations

# ruff: noqa: E501 -- generated Markdown paragraphs must remain on one source line.
import argparse
import ast
import gzip
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Also the release tag; bump it for each published starter.
NAME = "autoresearch-v0"
MODULES = """
__init__ build_contact_scoring_cache check_environment contact_cache contact_parallel
data data_budget data_migration evaluate fit_contact_probe flash_attention global_sampling
merge_contact_evaluation merge_full_evaluation model pcore_task periodic_evaluation
released_contact resume schedule setup_evaluation sharded_data
summarize_training_runs tokenizer train verify_contact_scoring_cache
""".split()
TESTS = """
test_contact_cache test_evaluation_setup test_fast_contact_scoring test_parallel_contact_evaluation
test_shard_selection test_training_budgets test_task_measurement test_model_configs
test_autoresearch_hardware test_validation_mlm
""".split()
REMOVED_FIELDS = {
    "learned_residual_routing",
    "transformer_norm",
    "depth_scaled_residual_init",
    "ffn_hidden_dim",
    "tie_word_embeddings",
}


def replace(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"source drift: missing {old[:100]!r}")
    return text.replace(old, new)


def remove_nodes(text: str, predicate) -> str:
    nodes = [node for node in ast.walk(ast.parse(text)) if predicate(node)]
    spans = [(node.lineno - 1, node.end_lineno) for node in nodes]
    lines = text.splitlines(keepends=True)
    return "".join(
        line for i, line in enumerate(lines) if not any(a <= i < b for a, b in spans)
    )


def clean_model(text: str) -> str:
    text = remove_nodes(
        text,
        lambda n: (
            isinstance(n, ast.AnnAssign)
            and isinstance(n.target, ast.Name)
            and n.target.id in REMOVED_FIELDS
        )
        or (
            isinstance(n, ast.FunctionDef | ast.ClassDef)
            and n.name in {"ESMCRMSNorm", "_transformer_norm", "_route_residual"}
        )
        or (
            isinstance(n, ast.If)
            and any(
                isinstance(a, ast.Attribute) and a.attr in REMOVED_FIELDS
                for a in ast.walk(n.test)
            )
            and not ast.unparse(n.test) == "hidden is None"
        ),
    )
    text = replace(
        text,
        "_transformer_norm(config.d_model, config.transformer_norm, bias=False)",
        "nn.LayerNorm(config.d_model, bias=False)",
    )
    text = replace(
        text,
        "_transformer_norm(config.d_model, config.transformer_norm)",
        "nn.LayerNorm(config.d_model)",
    )
    text = replace(
        text,
        "        hidden = config.ffn_hidden_dim\n        if hidden is None:\n            hidden = ",
        "        hidden = ",
    )
    text = replace(
        text,
        "    hidden = config.ffn_hidden_dim\n    if hidden is None:\n        hidden = ",
        "    hidden = ",
    )
    text = replace(text, "        initial_hidden = hidden\n", "")
    text = replace(
        text, "for layer_index, block in enumerate(self.blocks):", "for block in self.blocks:"
    )
    text = replace(
        text,
        "            hidden = self._route_residual(hidden, initial_hidden, layer_index)\n",
        "",
    )
    text = replace(
        text,
        '    transformer_norm = 0 if config.transformer_norm == "rmsnorm" else 6 * width\n',
        "",
    )
    text = replace(
        text,
        "        transformer_norm  # attention, Q/K, and FFN pre-normalization",
        "        6 * width  # attention, Q/K, and FFN pre-normalization",
    )
    text = replace(
        text,
        '    final_norm = 0 if config.transformer_norm == "rmsnorm" else width',
        "    final_norm = width",
    )
    text = replace(
        text,
        "    routing = 2 * config.n_layers if config.learned_residual_routing else 0\n",
        "",
    )
    text = replace(
        text, "    tied_savings = embedding if config.tie_word_embeddings else 0\n", ""
    )
    text = replace(text, " + head + routing - tied_savings", " + head")
    return text


def clean_train(text: str) -> str:
    text = replace(text, "from .batch_balance import rebalance_masked_batch\n", "")
    text = remove_nodes(
        text,
        lambda n: isinstance(n, ast.FunctionDef | ast.ClassDef)
        and n.name
        in {
            "training_losses",
            "_OptimizerBundle",
            "_canonical_parameter_name",
            "_nonempty_parameter_groups",
            "muon_adamw_parameter_groups",
            "build_optimizer",
        },
    )
    plain_optimizer = """def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    name = str(config.get("optimizer", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"unknown optimizer {name!r}")
    return torch.optim.AdamW(
        parameter_groups(model, weight_decay=float(config["weight_decay"])),
        lr=float(config["learning_rate"]),
        betas=tuple(config.get("betas", (0.9, 0.95))),
        eps=1e-8,
        fused=True,
    )


"""
    text = replace(text, "def train(\n", plain_optimizer + "def train(\n")
    text = remove_nodes(
        text,
        lambda n: isinstance(n, ast.If)
        and (
            ast.unparse(n.test).startswith("loss_reduction ")
            or ast.unparse(n.test) in {"balance_batches", "balance_statistics is not None"}
            or "allow_unscreened" in ast.unparse(n.test)
        ),
    )
    text = (
        "\n".join(
            line
            for line in text.splitlines()
            if not any(
                phrase in line
                for phrase in (
                    "loss_reduction =",
                    "balance_batches =",
                    "step_objective =",
                    "balance_statistics:",
                    *(f'"{field}":' for field in REMOVED_FIELDS),
                )
            )
        )
        + "\n"
    )
    text = replace(
        text,
        'loss, diagnostic, objective = training_losses(\n                    output["logits"], labels, reduction=loss_reduction\n                )',
        'loss = sequence_mean_loss(output["logits"], labels)',
    )
    text = replace(text, "step_loss += float(diagnostic)", "step_loss += float(loss.detach())")
    text = replace(text, ' * float(group.get("lr_scale", 1.0))', "")
    text = replace(
        text,
        "    data_root: Path,\n    *,\n    allow_unscreened: bool = False,\n",
        "    data_root: Path,\n",
    )
    text = replace(
        text,
        '"""Validate a screened corpus, or an explicitly opted-in unscreened experiment."""',
        '"""Validate the screened training corpus and verification receipts."""',
    )
    text = replace(
        text,
        '    data_manifest = validate_data_manifest(\n        data_root, allow_unscreened=config.get("allow_unscreened_training_data") is True\n    )',
        "    data_manifest = validate_data_manifest(data_root)",
    )
    text = replace(
        text,
        "def _git_state(root: Path) -> dict[str, object]:\n",
        'def _git_state(root: Path) -> dict[str, object]:\n    if not (root / ".git").exists():\n        return {"git_commit": None, "git_dirty": None}\n',
    )
    return text


def clean_setup(text: str) -> str:
    source_manifest = json.loads(
        (ROOT / ".dev/scripts/clean_starter_templates/source_files.json").read_text()
    )
    clean_manifest = (
        json.dumps({"files": source_manifest["files"]}, indent=2, sort_keys=True) + "\n"
    )
    clean_sha = hashlib.sha256(clean_manifest.encode()).hexdigest()
    text = replace(
        text,
        f'SOURCE_MANIFEST_SHA256 = "{clean_sha}"',
        f'SOURCE_MANIFEST_SHA256 = "{clean_sha}"\n'
        "BUNDLE_SOURCE_MANIFEST_SHA256 = SOURCE_MANIFEST_SHA256",
    )
    text = remove_nodes(
        text,
        lambda n: isinstance(n, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "LEGACY_SOURCE_MANIFEST_SHA256"
            for target in n.targets
        ),
    )
    text = replace(
        text,
        "    # Existing research installations retain their original metadata and receipt.\n",
        "",
    )
    text = replace(
        text,
        "source_manifest_sha not in {SOURCE_MANIFEST_SHA256, LEGACY_SOURCE_MANIFEST_SHA256}",
        "source_manifest_sha != SOURCE_MANIFEST_SHA256",
    )
    text = replace(text, "import argparse\n", "import argparse\nimport hashlib\n")
    text = remove_nodes(
        text,
        lambda n: isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "extract_bundle",
            "prepare_evaluation",
        },
    )
    implementation = (
        ROOT / ".dev/scripts/clean_starter_templates/install_assets.py.txt"
    ).read_text()
    text = replace(text, "def main() -> None:\n", implementation + "\n\ndef main() -> None:\n")
    text = replace(
        text,
        '    ready = json.loads((contact / "PDB_CONTACT_DATASET_READY.json").read_text())',
        '    ready = json.loads((contact / "PDB_CONTACT_DATASET_READY.json").read_text())\n'
        "    if set(ready) - READY_FIELDS:\n"
        '        raise ValueError("use a fresh evaluation root with this release")',
    )
    return text


def wrap(*blocks: str) -> str:
    return (
        "\n\n".join(f'<div class="ai">\n\n{block.strip()}\n\n</div>' for block in blocks) + "\n"
    )


def preparation_blocks() -> list[str]:
    """Reuse the main protocol's preparation steps so both releases give the same commands."""
    text = (ROOT / "docs/AUTORESEARCH.md").read_text()
    section = text[
        text.index("## Preparation") + len("## Preparation") : text.index("## Design Space")
    ]
    blocks = re.findall(r'<div class="ai">\n\n(.*?)\n\n</div>', section, re.S)
    if not blocks:
        raise ValueError("source drift: missing preparation section in docs/AUTORESEARCH.md")
    return ["## Preparation", *blocks]


def information_rules() -> str:
    return (ROOT / ".dev/scripts/clean_starter_templates/information_rules.md").read_text()


def documents() -> dict[str, str]:
    return {
        "README.md": wrap(
            "# ESMC AutoResearch starter",
            "This release provides a plain ESMC training recipe and a fixed benchmark protocol for comparing AutoResearch algorithms. Every participant starts from the same files, training corpus and evaluation procedure. The package includes training, checkpointing, data preparation, evaluation and measurement commands; participants supply their own search algorithm.",
            "## Starting recipe",
            "The starting model has 24 transformer layers, width 768, 12 attention heads and 170,671,168 trainable parameters. It uses the ESMC tokenizer, LayerNorm, RoPE with base 10,000, SwiGLU and an MLM head. AdamW uses learning rate 0.0005, weight decay 0.01, betas (0.9, 0.95) and gradient clipping at 1.0, with 500 warmup steps followed by constant learning rate. Context is 512 tokens and the global batch is 256 proteins on four GPUs. The search recipe is [configs/autoresearch/esmc-171m.yaml](configs/autoresearch/esmc-171m.yaml); its final-evaluation version, [configs/test-100k/esmc-171m.yaml](configs/test-100k/esmc-171m.yaml), uses global batch 1,024 and 1,000 warmup steps.",
            "## Preparation",
            "Use Linux with four matching H100 or four matching L40S GPUs and a working CUDA driver. The organizer prepares a fresh workspace from the `autoresearch-v0` release before starting the agent. [Preparation instructions](docs/AUTORESEARCH.md#preparation) cover private-repository access and data sizing.",
            "```bash\ngit clone --depth 1 --single-branch --no-tags --branch autoresearch-v0 \\\n  https://github.com/Lumin-Science/Nano-ProteinLM.git nano-protein-autoresearch\ncd nano-protein-autoresearch\ngit remote remove origin\nbash runs/setup.sh\nbash tasks/171m-validation-loss_ar.sh configs/autoresearch/esmc-171m.yaml trial-001 42\n```",
            "The measurement command trains from scratch for 20 minutes on four H100s or one hour on four L40S GPUs, saves the final checkpoint and evaluates all 12,288 validation proteins plus all 20,775 contact chains. The seed is an argument supplied by the caller. Running this command once consumes one search round. It does not implement a search policy.",
            "## Optional sequential search",
            "[autoresearch/program.md](autoresearch/program.md) supplies a Karpathy-style sequential method with one training seed per candidate and an evidence-based keep/discard decision. The agent must explain its decisions; one seed does not establish statistical significance. The benchmark protocol does not require this method.",
            "Install the loop-and-sleep skill before starting the agent. Run Codex inside tmux on the allocated compute node so the skill can wake the same pane after training and evaluation. The command below enables automatic review of execution approvals, including GPU access outside the workspace sandbox.",
            "```bash\nnpx skills add Lumin-Science/Nano-AutoResearch-Skills --skill ar-loop-n-sleep -g -a codex\nnpx skills list -g  # confirm ar-loop-n-sleep is installed for Codex\ntmux new-session -s nanoprotein-ar\n# Inside tmux, in the prepared workspace:\ncodex --approve-for-me\n```",
            "```text\nUse $ar-loop-n-sleep. Read tasks/171m-validation-loss.md and autoresearch/program.md. Use the allocated four H100 GPUs, verify the live allocation, and run sequential AutoResearch with one seed per candidate. Preserve the full task evaluation, explain every keep/discard decision, and stop after the agreed round allowance.\n```",
            "Change the resource description to the actual allocation and state a smaller round limit for a qualification run. Other AutoResearch methods may use the same task without this program or skill.",
            "## Protocol and usage",
            "[AUTORESEARCH.md](docs/AUTORESEARCH.md) defines the 72-round search budget, permitted changes and final evaluation. [DATA.md](docs/DATA.md) describes the corpus and preparation. [EVALUATION.md](docs/EVALUATION.md) fixes the rewards. [USAGE.md](docs/USAGE.md) covers ordinary training and owner-run final evaluation. [ORGANIZER.md](docs/ORGANIZER.md) describes how to distribute identical workspaces and keep scoring under organizer control.",
            "## Checks",
            "```bash\nuv sync --frozen\nuv run --frozen python -m unittest discover -s .dev/tests -p 'test_*.py'\n```",
            "Code is distributed under the [MIT license](LICENSE). Dataset and kernel attribution is in [DATA.md](docs/DATA.md). Large datasets, dependencies and pretrained weights are not included in the released code.",
        ),
        "docs/AUTORESEARCH.md": wrap(
            "# AutoResearch benchmark protocol",
            "The benchmark compares algorithms that discover protein-model training recipes. Publish the objective, permitted design space, search allowance, final evaluation budget and infrastructure-failure policy before any method starts. Every method receives the same starting release and data access.",
            '```mermaid\nflowchart LR\n    B["72 rounds<br/>20 min on 4×H100 or 1 h on 4×L40S"] --> A["Participant search algorithm"]\n    A --> R["Selected recipe"]\n    R --> E["Owner-run evaluation<br/>24.2B tokens per recipe"]\n```',
            *preparation_blocks(),
            "## Information-access rules",
            *information_rules().split("\n\n")[1:],
            "## Immutable contract",
            "Use only the provided training corpus and train from scratch. Keep the tokenizer, maximum context of 512 tokens, global batch of 256 sequences and schedule of 500 linear warmup steps followed by constant learning rate, with no decay, fixed. Actual trainable parameters must remain within ±5% of 170,671,168; do not add unused parameters to meet this bound. Preserve evaluation data, sampling, masking, probe fitting, metric definitions, dependency pins and data-verification records. Do not train on held-out evaluation sequences or labels.",
            "Architecture, training objective, optimizer, learning rate, weight decay, implementation, source mixture and data selection within the supplied corpus may change within that contract. Declare the task objective before search: MLM validation loss is the default reward; contact P@L is an optional task objective. Report both measurements for every round.",
            "## Search allowance",
            "| Item | Fixed setting |\n|---|---|\n| Rounds | 72 |\n| Hardware per round | 4×H100 with FA3 or 4×L40S with FA2; fixed profile within a comparison |\n| Training time per round | 1,200 seconds on H100 or 3,600 seconds on L40S |\n| Global batch and schedule | 256 sequences; 500 linear warmup steps, then constant learning rate |\n| Total training allowance | 24 node-hours / 96 H100 GPU-hours, or 72 node-hours / 288 L40S GPU-hours |\n| Scored checkpoint | Final checkpoint at the training-time limit |\n| Validation | All 12,288 validation proteins; context 512 |\n| Contact evaluation | All 20,775 frozen chains; 5,000 bootstrap replicates |",
            "A round is one training run and its evaluation. Each seed repeat or participant-run reference measurement consumes another round. A method decides how to propose candidates, assign training seeds, reuse observations and choose its final recipe. No acceptance threshold, number of seeds per candidate or search order is prescribed here.",
            "The training clock includes batch loading, computation and synchronization. Setup, final checkpoint saving and evaluation are outside that clock; record their duration separately. Prepare inputs before timing and use the same storage arrangement across methods. Keep failed attempts and consumed compute in the ledger; apply only the replacement policy published before the comparison.",
            "Record the code snapshot, resolved configuration, training seed, data receipts, actual optimizer steps, non-padding model tokens, source exposure, elapsed time and both metrics for each round. Include time-limit overrun from the final optimizer update. The measurement scripts provide execution commands; the organizer verifies compliance and the complete round ledger.",
            "## Final evaluation",
            "Freeze the selected recipe before final evaluation. The owner trains the submitted recipe and the reference, [configs/test-100k/esmc-171m.yaml](../configs/test-100k/esmc-171m.yaml), from scratch to 24,200,224,761 non-padding model tokens each, using one common training seed: 42 unless another seed is declared before the comparison. Final training uses global batch 1,024 and 1,000 linear warmup steps followed by constant learning rate, with no decay; learning rate, weight decay and every other setting come from the submitted recipe. Hardware is not fixed because the token target defines the budget; the reference takes about 12 hours on four H100 GPUs. Count BOS/EOS and exclude padding. Stop at the first optimizer update reaching the token target and report actual tokens and overrun. This compute is separate from the search allowance.",
            "Score each final checkpoint using the same full MLM validation and contact evaluation used during search. Report loss, P@L and its chain-bootstrap 95% interval. One training seed does not estimate training-seed variability. These evaluation assets are available during search, so final evaluation tests transfer to a longer training budget rather than performance on a blind holdout.",
        ),
        "docs/DATA.md": wrap(
            "# Data and setup",
            "Training uses the processed LuminScience/LuminBench-Nano-ESMC corpus at immutable revision `bd38448d50d8f426d7b9bd4410b53159ea001259`. It contains 665,970,495 proteins across 565 training shards from UniRef90, MGnify and OMG/IMG. The starting mixture is 0.36:0.11:0.54, normalized by the sampler. Each source has a separate 4,096-protein validation shard; all three validation shards are always prepared.",
            "```bash\nbash runs/setup.sh --training-shards 30\n# Or size a download for an explicitly chosen sample budget:\n# bash runs/setup.sh --training-samples N\n```",
            "Setup downloads checksum-bound source prefixes, materializes local sequence stores and verifies the homology-exclusion receipts before training. The default 30 training shards contain 29,979,351 proteins; allow roughly 20 GB for compressed and prepared data plus separate space for dependencies, checkpoints and evaluation. Use a fresh DATA_ROOT when changing the shard selection. Existing prepared roots retain their saved selection.",
            "Size the corpus for the actual source mixture and training budget, including sampling headroom. Read DATA_COVERAGE.json before interpreting a run. With the default policy the trainer stops on source exhaustion. Record any explicitly enabled reuse. Stored residues and requested samples are distinct from model tokens: proteins are cropped to at most 510 residues, BOS/EOS are added and padding is excluded from the training-token meter.",
            "## Frozen evaluation assets",
            "Setup installs a checksum-verified contact bundle under `DATA_ROOT/evaluation`. Its numerical evaluator sources, contact splits and payload hashes are fixed. The installer retains only the required payloads, source modules and license; archive metadata is discarded. The organizer should finish setup and inspect the prepared workspace before granting an agent access.",
            "## Attribution",
            "UniRef90 is provided by the UniProt Consortium, MGnify by EMBL-EBI and OMG/IMG by their source data providers. The processed corpus carries source provenance and decontamination receipts. Contact chains derive from PDB structures; preserve the supplied dataset records and their applicable terms. Evaluator code is distributed under the accompanying MIT license. These source-code terms do not replace the original datasets' terms.",
            "On Hopper, the pinned FlashAttention-3 kernel comes from kernels-community/flash-attn3 at revision `e29f138fc363b396e5d2706c8a5f6fa7d36f41e0`. Its downloaded distribution retains its own license and metadata. Other supported GPUs use the FlashAttention-2 operators in the pinned PyTorch distribution. PyTorch and the remaining dependencies retain their upstream licenses.",
        ),
        "docs/EVALUATION.md": wrap(
            "# Evaluation",
            "## MLM validation loss",
            "Compute masked-token negative log-likelihood within each protein, then average over proteins. Lower is better. The evaluator scores every protein in the three held-out validation shards once: 12,288 proteins, 4,096 per source, at maximum context 512. Each protein's crop offset and mask positions come from mask seed 20260821 and the protein's SHA-256, so batch size and order do not change the score.",
            "Each receipt records the protocol, settings, a SHA-256 of the evaluated protein digests and per-source mean losses. Compare only receipts with identical protocol and settings. The score does not estimate variability across training seeds.",
            "## Contact P@L",
            "Contact P@L is the mean, over 20,775 frozen chains, of precision among the top L predicted long-range residue contacts. L is the evaluated chain length. Higher is better. Fit one contact probe per checkpoint using the frozen probe split and procedure, then evaluate all chains. Use 5,000 chain-bootstrap replicates for a 95% interval. The interval describes variation over chains.",
            "## Evaluate a checkpoint",
            "```bash\nuv run --frozen python -m nanoprotein.evaluate \\\n  --checkpoint outputs/trial-001/checkpoint-final.pt \\\n  --data-root data/training --output-root outputs/trial-001/evaluation \\\n  --validation-context 512 \\\n  --run-contact --contact-mode parallel --contact-chains 20775 --contact-bootstrap 5000 \\\n  --contact-gpus 0,1,2,3 --contact-workers 32 \\\n  --contact-root data/evaluation/contact --external-src data/evaluation/source\n```",
            "Adjust only the file paths and visible GPU identifiers to match the provisioned workspace. Use the final checkpoint from the declared training budget. The organizer evaluates with a trusted copy of the scoring code and records checkpoint and evaluator hashes.",
        ),
        "docs/USAGE.md": wrap(
            "# Training commands",
            "## Search round",
            "```bash\nbash tasks/171m-validation-loss_ar.sh configs/autoresearch/esmc-171m.yaml trial-001 42\n```",
            "The command requires prepared data and four matching H100 or L40S GPUs. It selects 1,200 seconds on H100 or 3,600 seconds on L40S. It refuses to overwrite a run directory and evaluates only after training succeeds. `tasks/171m-p-at-l_ar.sh` runs the same measurement for the P@L task. Each invocation is one round; the search algorithm owns its candidate and seed choices.",
            "## Ordinary training",
            "The commands below assume Hopper GPUs and FlashAttention-3. For the one-hour L40S search budget, use `--attention-backend flash --walltime-seconds 3600` in the ordinary training command below. Set `--peak-bf16-tflops-per-gpu` to the value recorded in a task run's `ENVIRONMENT.json` for hardware-specific utilization reporting. The benchmark task reads these settings automatically.",
            "```bash\nuv run --frozen python -m torch.distributed.run --standalone --nproc-per-node=4 \\\n  -m nanoprotein.train --config configs/autoresearch/esmc-171m.yaml \\\n  --data-root data/training --output-root outputs/training-001 \\\n  --walltime-seconds 1200 --seed 42\n```",
            "Ordinary training and benchmark measurements use the same API. Read the resolved recipe, run_contract.json, DATA_COVERAGE.json and TRAINING_COMPLETE.json to confirm the effective settings and stop reason. Successful training writes checkpoint-final.pt. The trainer supports time, step and token limits; the measurement command clears step and token caps so the round uses its training-time allowance.",
            "## Owner-run final evaluation",
            "Prepare enough data for the selected mixture and the 24,200,224,761-token target before starting. The example trains the reference on four Hopper GPUs with FlashAttention-3, where it takes about 12 hours. On other GPUs, use `--attention-backend flash` and raise the 16-hour safety time cap; completion is determined by the token target, not this cap. Keep the global batch at 1,024 and use the same seed for every recipe.",
            "```bash\nuv run --frozen python -m torch.distributed.run --standalone --nproc-per-node=4 \\\n  -m nanoprotein.train --config configs/test-100k/esmc-171m.yaml --seed 42 \\\n  --data-root data/training --output-root outputs/final-reference \\\n  --max-steps none --max-model-tokens 24200224761 --schedule-steps 100000 \\\n  --walltime-seconds 57600 --attention-backend flash3 --warmup-steps 1000 \\\n  --micro-batch-size 64 --gradient-accumulation 4 \\\n  --checkpoint-interval 0 --periodic-evaluation-interval 0\n```",
            "Verify that TRAINING_COMPLETE.json reports the token target as the stop reason and at least 24,200,224,761 model tokens. Record the final update's overrun. Repeat for the frozen submitted recipe with the same seed, then use the [evaluation command](EVALUATION.md#evaluate-a-checkpoint) on each final checkpoint. This owner-run comparison does not consume search rounds.",
        ),
        "docs/ORGANIZER.md": wrap(
            "# Organizing a comparison",
            "Provision every participant from the same `autoresearch-v0` release tag and record its commit. The documented clone downloads only that root commit, and the remote is removed afterwards. Give each agent a fresh context, prepared data and the same compute allowance. Keep prior experiments, checkpoint stores, private notes, shell history and other checkouts outside the agent workspace.",
            "Prepare dependencies and verified data before agent access. For a closed comparison, restrict network access after provisioning; an independent release commit and written rules do not block online access. Declare the external-information policy, agent/model version and proposal-generation budget before the comparison, and use the same policy for every method.",
            "Keep an organizer-owned copy of the protocol, evaluator, evaluation assets and release manifest outside the agent's writable workspace. Run candidate training and model code in an isolated process with only the required data and permissions. Score submitted checkpoints using the trusted evaluation procedure, and independently check architecture changes, data boundaries, parameter counts and the compute ledger. A manifest inside an editable workspace is a reference, not an enforcement mechanism.",
            "The release excludes previous research findings from its code and commit ancestry. It does not by itself prevent metric tampering or adaptation to repeatedly observed validation results. The published final evaluation shares the search evaluation assets; any blind generalization study needs a separately declared held-out set.",
        ),
    }


def build(destination: Path) -> Path:
    package = destination / NAME
    if package.exists():
        raise FileExistsError(f"choose a fresh destination: {package}")
    package.mkdir(parents=True)

    def write(name: str, text: str):
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def copy(name: str):
        source = ROOT / name
        assert source.is_file() and not source.is_symlink(), source
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    for name in ("LICENSE", "pyproject.toml", "uv.lock", ".env.example"):
        copy(name)
    for name in MODULES:
        copy(f"src/nanoprotein/{name}.py")
    for name in ("evaluate_full_parallel.sh", "evaluate_p_at_l_parallel.sh"):
        copy(f"src/{name}")
    for name in ("171m-validation-loss_ar.sh", "171m-p-at-l_ar.sh"):
        copy(f"tasks/{name}")
    copy("runs/setup.sh")
    write(
        "autoresearch/program.md",
        (ROOT / "autoresearch/program.md").read_text(),
    )
    for name in TESTS:
        copy(f".dev/tests/{name}.py")
    for name in ("test_plain_baseline", "test_asset_installation"):
        write(
            f".dev/tests/{name}.py",
            (ROOT / f".dev/scripts/clean_starter_templates/{name}.py.txt").read_text(),
        )
    write(
        "src/nanoprotein/model.py",
        clean_model((package / "src/nanoprotein/model.py").read_text()),
    )
    write(
        "src/nanoprotein/train.py",
        clean_train((package / "src/nanoprotein/train.py").read_text()),
    )
    write(
        "src/nanoprotein/setup_evaluation.py",
        clean_setup((package / "src/nanoprotein/setup_evaluation.py").read_text()),
    )
    budgets = (package / ".dev/tests/test_training_budgets.py").read_text()
    budgets = remove_nodes(
        budgets,
        lambda n: isinstance(n, ast.FunctionDef)
        and n.name == "test_public_best_alias_matches_canonical_incumbent",
    )
    budgets = (
        budgets.replace("24200224761", "24000000000")
        .replace("from pathlib import Path\n", "")
        .replace("import yaml\n", "")
    )
    write(".dev/tests/test_training_budgets.py", budgets)
    selection = (package / ".dev/tests/test_shard_selection.py").read_text()
    write(
        ".dev/tests/test_shard_selection.py",
        selection.replace(
            "test_seven_shards_preserve_benchmark_selection",
            "test_whole_shard_selection_is_deterministic",
        ),
    )
    # The generic tar helper now deliberately requires frozen-manifest member names.
    test = (package / ".dev/tests/test_evaluation_setup.py").read_text()
    test = remove_nodes(
        test,
        lambda n: isinstance(n, ast.FunctionDef)
        and n.name == "test_safe_files_extract_but_traversal_and_links_fail",
    )
    for line in (
        "import io\n",
        "import tarfile\n",
        "from unittest.mock import patch\n",
        "from nanoprotein.data import file_sha256\n",
    ):
        test = test.replace(line, "")
    write(".dev/tests/test_evaluation_setup.py", test)
    sharded = (package / "src/nanoprotein/sharded_data.py").read_text()
    sharded = replace(
        sharded,
        "This mirrors nanochat's operational model: immutable whole\nshards are cached locally; individual rows are not streamed over the network\ninside the training loop.",
        "Immutable whole shards are cached locally; individual rows are not\nstreamed over the network inside the training loop.",
    )
    write("src/nanoprotein/sharded_data.py", sharded)
    # Ship the plain reference configs as-is, minus internal provenance comments.
    for name in ("configs/autoresearch/esmc-171m.yaml", "configs/test-100k/esmc-171m.yaml"):
        lines = (ROOT / name).read_text().splitlines(keepends=True)
        write(name, "".join(line for line in lines if not line.startswith("# Source:")))
    write(
        "AGENTS.md",
        """# Working in the ESMC benchmark

Read the selected task before running research; it contains the complete scientific, evaluation, data, model-size and compute contract. Every training run consumes a round, including seed repeats. Record failed attempts and consumed compute. The organizer owns final evaluation.

Use src/nanoprotein for ordinary training and evaluation code, configs for recipes, tasks for measurement commands and OUTPUT_ROOT for artifacts. Keep your search algorithm and selection policy separate from task definitions. Do not change evaluation data, metric definitions or frozen dependencies. Use .dev/tests for checks.

Keep each prose paragraph on one source line when writing Markdown. Existing AI review wrappers are editorial annotations and do not affect the protocol.
""",
    )
    write(
        ".gitignore",
        ".venv/\n.env\ndata/\noutputs/\n.ar/\n__pycache__/\n*.pyc\n*.egg-info/\n.ruff_cache/\n",
    )
    for path, value in documents().items():
        write(path, value)
    for name in ("171m-validation-loss", "171m-p-at-l"):
        copy(f"tasks/{name}.md")
    path = package / "AGENTS.md"
    path.write_text(path.read_text().rstrip() + "\n\n" + information_rules())
    modified = [
        package / name
        for name in (
            "src/nanoprotein/model.py",
            "src/nanoprotein/train.py",
            "src/nanoprotein/setup_evaluation.py",
            ".dev/tests/test_task_measurement.py",
            ".dev/tests/test_plain_baseline.py",
            ".dev/tests/test_asset_installation.py",
            ".dev/tests/test_training_budgets.py",
            ".dev/tests/test_evaluation_setup.py",
        )
    ]
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--no-cache", *map(str, modified)], check=True
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--fix",
            "--select",
            "I",
            *map(str, modified),
        ],
        check=True,
    )
    return package


def audit(package: Path) -> dict:
    # Private denylist: never ship these clues or the builder in the agent package.
    forbidden = re.compile(
        r"muon|nanochat|rmsnorm|learned_residual|depth_scaled|ffn_hidden_dim|"
        r"tie_word_embeddings|sqrt_mask_count|balance_batches_across|"
        r"query.center|rms.restor|best.recipe|CURRENT_DEFAULT|BEST_RECIPE|"
        r"kn0[89]\d|ar-260913|human-ai-baseline|program2_h100|"
        r"62faec9|current.best|"
        r"canonical.incumbent|nanop-best|\.dev/configs/archive|\.dev/reports",
        re.I,
    )
    violations = []
    files = {}
    modules = {path.stem for path in (package / "src/nanoprotein").glob("*.py")}
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in release: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        if relative == "RELEASE_MANIFEST.json":
            continue
        if any(
            part in {".git", "__pycache__", "outputs", ".venv", "archive", "reports"}
            for part in Path(relative).parts
        ):
            violations.append(relative)
        content = path.read_bytes()
        matches = sorted(set(forbidden.findall(relative + "\n" + content.decode())))
        if matches:
            violations.append({"file": relative, "matches": matches})
        if path.suffix == ".py":
            tree = ast.parse(content, filename=relative)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level and node.module:
                    if node.module.split(".")[0] not in modules:
                        violations.append({"file": relative, "missing_import": node.module})
        if path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
        files[relative] = hashlib.sha256(content).hexdigest()
    if violations:
        raise ValueError(json.dumps(violations, indent=2))
    manifest = {"release": NAME, "files": files}
    (package / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"file_count": len(files) + 1, "content_scan": "passed", "python_parse": "passed"}


def archive(package: Path) -> Path:
    output = package.parent / (NAME + ".tar.gz")
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as handle:
                for path in sorted(package.rglob("*")):
                    if not path.is_file():
                        continue
                    data = path.read_bytes()
                    info = tarfile.TarInfo(f"{NAME}/{path.relative_to(package).as_posix()}")
                    info.size = len(data)
                    info.mode = 0o755 if path.suffix == ".sh" else 0o644
                    info.mtime = 0
                    handle.addfile(info, io.BytesIO(data))
    (package.parent / "SHA256SUMS").write_text(
        hashlib.sha256(output.read_bytes()).hexdigest() + "  " + output.name + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-existing", type=Path)
    args = parser.parse_args()
    package = args.package_existing or build(args.output.resolve())
    report = audit(package)
    output = archive(package)
    print(json.dumps({**report, "archive": str(output)}, indent=2))


if __name__ == "__main__":
    main()
