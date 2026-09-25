# ESMC AutoResearch starter

This release provides a plain ESMC training recipe and a fixed benchmark protocol for comparing AutoResearch algorithms. Every participant starts from the same files, training corpus and evaluation procedure. The package includes training, checkpointing, data preparation, evaluation and measurement commands; participants supply their own search algorithm.

## Starting recipe

The starting model has 24 transformer layers, width 768, 12 attention heads and 170,671,168 trainable parameters. It uses the ESMC tokenizer, LayerNorm, RoPE with base 10,000, SwiGLU and an MLM head. AdamW uses learning rate 0.0005, weight decay 0.01, betas (0.9, 0.95) and gradient clipping at 1.0, with 500 warmup steps followed by constant learning rate. Context is 512 tokens and the global batch is 256 proteins on four GPUs. The search recipe is [configs/autoresearch/esmc-171m.yaml](configs/autoresearch/esmc-171m.yaml); its final-evaluation version, [configs/test-100k/esmc-171m.yaml](configs/test-100k/esmc-171m.yaml), uses global batch 1,024 and 1,000 warmup steps.

## Preparation

Use Linux with four matching H100 or four matching L40S GPUs and a working CUDA driver. The organizer prepares a fresh workspace from the `autoresearch-v0` release before starting the agent. [Preparation instructions](docs/AUTORESEARCH.md#preparation) cover the release tag and data sizing.

```bash
git clone --depth 1 --single-branch --no-tags --branch autoresearch-v0 \
  https://github.com/Lumin-Science/Nano-ProteinLM.git nano-protein-autoresearch
cd nano-protein-autoresearch
git remote remove origin
bash scripts/setup.sh
bash tasks/171m-validation-loss_ar.sh configs/autoresearch/esmc-171m.yaml trial-001 42
```

The measurement command trains from scratch for 20 minutes on four H100s or one hour on four L40S GPUs, saves the final checkpoint and evaluates all 12,288 validation proteins; the P@L task also scores all 20,775 contact chains. The seed is an argument supplied by the caller. Running this command once consumes one search round. It does not implement a search policy.

## Optional sequential search

Two optional Karpathy-style sequential programs differ only in how they decide what to keep: [autoresearch/karpathy_ar_reward_gate.md](autoresearch/karpathy_ar_reward_gate.md) keeps a candidate by a fixed two-seed reward rule, and [autoresearch/karpathy_ar_agent_gate.md](autoresearch/karpathy_ar_agent_gate.md) leaves the decision to the agent's reasoning. The benchmark protocol does not require either program.

Install the loop-and-sleep skill for your coding agent, then start the agent inside tmux on the allocated compute node so the skill can wake the same pane after training and evaluation. [autoresearch/setup_karpathy_ar.txt](autoresearch/setup_karpathy_ar.txt) describes the full setup, which a coding agent can follow for you.

```bash
# Name your agent with -a, for example codex or claude-code:
npx skills add Lumin-Science/Nano-AutoResearch-Skills --skill ar-loop-n-sleep -g -a codex
tmux new-session -s nanoprotein-ar
# Inside tmux, in the prepared workspace, start your agent, for example:
codex --approve-for-me
```

```text
Use the ar-loop-n-sleep skill. Read tasks/171m-validation-loss.md and autoresearch/karpathy_ar_reward_gate.md. Use the allocated four H100 GPUs, verify the live allocation, and run sequential AutoResearch. Preserve the full task evaluation, explain every keep/discard decision, and stop after the agreed round allowance.
```

Change the resource description to the actual allocation and state a smaller round limit for a qualification run. Name the agent-gate program instead to let the agent decide. Other AutoResearch methods may use the same task without these programs or the skill.

## Protocol and usage

[AUTORESEARCH.md](docs/AUTORESEARCH.md) defines the 72-round search budget, permitted changes and final evaluation. [DATA.md](docs/DATA.md) describes the corpus and preparation. [EVALUATION.md](docs/EVALUATION.md) fixes the rewards. [USAGE.md](docs/USAGE.md) covers ordinary training and owner-run final evaluation. [ORGANIZER.md](docs/ORGANIZER.md) describes how to distribute identical workspaces and keep scoring under organizer control.

## Checks

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s .dev/tests -p 'test_*.py'
```

Code is distributed under the [MIT license](LICENSE). Dataset and kernel attribution is in [DATA.md](docs/DATA.md). Large datasets, dependencies and pretrained weights are not included in the released code.
