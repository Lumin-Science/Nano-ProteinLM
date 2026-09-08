# Protein embedding training recipes

This is the authoritative task definition. Commands run from the repository root
on the specified GPU host; replace the input paths and choose a fresh run name.
Existing training configs are executable recipes, not separate task definitions.

## Background

### Research question

Which training-recipe changes produce more useful protein representations at
approximately fixed model size and data? Search under a fixed training-time
budget, then verify progress with longer training at matched token exposure.
The comparator is the original ESMC-like AdamW baseline. MLM loss is the search
proxy; both MLM loss and contact P@L assess progress. An attention-based contact
probe alone does not establish improvement on every downstream embedding task.

### Established codebase

- **Baseline:** [ESMC-171M AdamW](configs/esmc-171m-original.yaml), 24 layers,
  width 768, 12 heads, FFN 2048, 170,671,168 parameters, LayerNorm, RoPE 10k.
- **Inputs:** pinned, decontaminated UniRef90/MGnify/OMG-IMG corpus, fixed
  tokenizer, masking and mixture sampling. This is a public ESMC-like reconstruction.
- **Training:** distributed BF16, FA2/FA3, time/step/token stopping and full
  model/optimizer checkpoints; model, optimizer, batching and backward-loss experiments.
- **Evaluation:** frozen sequence-mean MLM and full 20,775-chain contact P@L,
  with checkpoint-bound records and chain-bootstrap intervals. See [evaluation](docs/EVALUATION.md).
- **Evidence:** [research history](reports/program2/README.md) and the
  [Test Leaderboard](README.md#test-leaderboard). Historical protocols remain unchanged.

## Autoresearch

### Protocol

| Setting | Definition |
|---|---|
| Training per repeat | From scratch, 4 L40S, BF16, 3,600 synchronized training-loop seconds |
| Schedule / input | 554-step linear warmup, then constant LR; Stage 1, context 512; baseline batch 256 |
| Repeats | N=2 by default, seeds 42 and 43; same declared seed set for baseline, incumbent and candidate |
| Search score (reward) | Minimize mean `sequence_mean_nll` over all N runs; report sample SD (`ddof=1`) |
| MLM evaluation | 32 sequences, 8 batches × 4, context 512, evaluation seed 20260821, equal source weights |
| Other measurements | Full 20,775-chain P@L, training loss, steps/tokens, parameters, memory and timing |

Within each sequence, average cross-entropy over masked targets; then average
across sequences. Keep the evaluator fixed even when changing the backward loss.
Accept only when **incumbent mean loss − candidate mean loss > candidate SD**.
The incumbent is the last accepted recipe. Ties and single runs cannot qualify;
P@L does not select research candidates. This is a heuristic, not a significance test.

The clock includes batching, computation, communication, logging and lazy work
inside the loop; setup, final checkpoint and subsequent evaluation are excluded.
Stop at the first update boundary after the limit and record overrun. Require
`stop_reason=walltime`, complete evaluation and `ROUND_STATE=complete` for every
repeat. Record all seeds, failures and keep/discard decisions; do not rerun to
replace unfavorable completed scores. Use an isolated campaign worktree, one
conceptual change per round and fresh output paths; freeze code/input/environment
identities and preserve the original baseline and all previous evidence.

**Complete example: train, evaluate and read the two-seed score.** The protected
runner prepares/verifies the frozen training release, trains and runs both MLM
and full P@L evaluation. Change `AR_RECIPE` to a candidate recipe for another method.

```bash
set -euo pipefail
export CONTACT_ROOT=/path/to/frozen-contact-data
export EXTERNAL_SRC=/path/to/pinned-evaluation-source
export ARTIFACT_ROOT="$PWD/.exps/program2"
export UV_CACHE_DIR="$PWD/.uv-cache/program2"
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset DATA_ROOT DATA_CACHE_ROOT CONTACT_SCORING_CACHE_ROOT
AR_RECIPE="$PWD/configs/esmc-171m-original.yaml"
AR_RUN_ROOT="$PWD/outputs/program2/research-example-01"
mkdir -p "$(dirname "$AR_RUN_ROOT")"
mkdir "$AR_RUN_ROOT" # Refuse an existing method output directory.
uv sync --frozen
uv run --frozen ruff check nano_protein scripts tests
uv run --frozen ruff format --check nano_protein scripts tests
for AR_SEED in 42 43; do
  uv run --frozen python - "$AR_RECIPE" "$AR_RUN_ROOT" "$AR_SEED" <<'PY'
import pathlib, sys, yaml
recipe, root, seed = sys.argv[1:]
config = yaml.safe_load(pathlib.Path(recipe).read_text())
config['seed'] = int(seed)
pathlib.Path(root, f'config-{seed}.yaml').write_text(yaml.safe_dump(config))
PY
  CONFIG="$AR_RUN_ROOT/config-$AR_SEED.yaml" \
  OUTPUT_ROOT="$AR_RUN_ROOT/seed-$AR_SEED" \
    bash runs/autoresearch_4xl40s_1h.sh
done
uv run --frozen python - "$AR_RUN_ROOT" <<'PY'
import json, math, pathlib, statistics, sys
rows = [json.loads((pathlib.Path(sys.argv[1])/f'seed-{s}/ROUND_SUMMARY.json').read_text()) for s in (42, 43)]
for metric in ('validation_loss', 'p_at_l'):
    values = [r[metric] for r in rows]
    assert all(math.isfinite(v) for v in values), (metric, values)
    print(metric, 'per_seed=', values, 'mean=', statistics.mean(values), 'SD=', statistics.stdev(values))
PY
```

The search score is the printed mean validation loss. Compare it and its SD
with the incumbent's matching records using the acceptance rule above. Preserve
`ROUND_SUMMARY.json`, final checkpoints, configs, source revision, environment/data
receipts and a per-run ledger with seed, metrics, hypothesis and decision.

### Boundaries

**Disallowed changes:**

- Never alter evaluation code, metrics/reductions, examples, probes, seeds or
  correctness checks: `nano_protein/evaluate.py`, `nano_protein/pcore_task.py`,
  `nano_protein/contact_cache.py`, `scripts/**`, `runs/**`, and external evaluator assets.
- Never change data sources, release, mixture, splits, tokenizer, cropping or
  masking: `nano_protein/data.py`, `nano_protein/sharded_data.py`,
  `nano_protein/tokenizer.py`. Training weights remain UniRef90 0.36, MGnify 0.11,
  OMG/IMG 0.54, normalized by their sum. No held-out training, added data or pretrained weights.
- Never alter the budget meter, stopping/receipt logic, evaluation invocation or
  frozen schedule, even inside otherwise editable code. Protect
  `nano_protein/schedule.py`, `nano_protein/resume.py`, `nano_protein/periodic_evaluation.py`.
- Never change the task, existing tests, dependency lock or recorded evidence:
  `program.md`, `program2.md`, `README.md`, `docs/**`, `reports/**`, `tests/test_*.py`,
  `pyproject.toml`, `uv.lock`, original configs and previous run outputs.
- Never move trainable parameters outside **162,137,610–179,204,726** (±5% of
  170,671,168), add dummy parameters to satisfy this bound, change the declared
  hardware/context, shorten the budget or specialize model outputs to evaluation identities.

**Allowed changes:** model architecture/init/parameter grouping in
`nano_protein/model.py`; optimizer, backward loss and training execution in
`nano_protein/train.py`; rank redistribution in `nano_protein/batch_balance.py`;
attention execution in `nano_protein/flash_attention.py`; helpers/configs/tests
under `nano_protein/experiments/**`, `configs/candidates/**`, `tests/candidates/**`.
Hyperparameters and research batch/accumulation may vary within the frozen rules.
Redistribution preserves selected examples; filtering or resampling them is forbidden.
New campaign records may be written under `.dev/program2/**` and fresh
`outputs/program2/**`; verified input caches remain read-only.

Protected rules override allowed paths; unlisted source files remain protected.
These restrictions govern candidate experiments. A task revision between campaigns
must establish a new comparable baseline; shared editable code requires source audit.

## Test of Progress

Freeze the candidate and transfer recipe, then train from scratch under the
settings below. Run the original baseline and preceding cumulative recipe under
the same settings. **Progress requires both lower mean MLM loss and higher mean
P@L** against each named comparator; report ties, regressions and mixed results.

| Setting | Definition |
|---|---|
| Budget per seed | 24,200,224,761 global non-padding model tokens, including BOS/EOS; exclude padding/evaluation |
| Endpoint | First completed update reaching the token target; record actual count/overrun, less than one update (524,288 tokens maximum at batch 1024/context 512) |
| Hardware / repeats | 4 H100, BF16, FA3; N=2 from-scratch runs, seeds 42 and 43 |
| Transfer | Stage 1, context 512, batch 1024; base LR 5e-4, base WD 0.01, 1,000-step warmup then constant; retain declared optimizer-group multipliers |
| MLM | 4,096 sequences, 1,024 batches × 4, context 512, seed 20260821, sequence-mean NLL |
| P@L | All 20,775 chains; 16-chain probe fit / 4-chain selection / 20-chain refit; pair seed 20260819; fixed ordering and 5,000-resample 95% chain bootstrap, seed 20260820 |
| Output | Final budget checkpoint with full optimizer/runtime state; both full evaluations, per-seed values and mean ± sample SD |

The token count is summed across all ranks/microbatches before redistribution;
recomputation does not add tokens. The 16-hour guard is an emergency stop: an
under-budget run is incomplete. A 100k-step cap is removed for this protocol.

**Complete example: larger training, budget verification, full evaluation and
metric extraction.** Run on four H100s with the pinned FA3 environment and frozen
prepared corpus. Change `AR_RECIPE` to the accepted candidate's recipe; the block
applies the same transfer settings to every method. Use fresh outputs each time.

```bash
set -euo pipefail
export DATA_ROOT=/path/to/frozen-prepared-training-data
export CONTACT_ROOT=/path/to/frozen-contact-data
export EXTERNAL_SRC=/path/to/pinned-evaluation-source
export UV_CACHE_DIR="$PWD/.uv-cache/program2"
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset CONTACT_SCORING_CACHE_ROOT
AR_RECIPE="$PWD/configs/esmc-171m-original.yaml"
AR_RUN_ROOT="$PWD/outputs/program2/progress-example-01"
mkdir -p "$(dirname "$AR_RUN_ROOT")"
mkdir "$AR_RUN_ROOT"
uv sync --frozen
uv run --frozen python scripts/check_environment.py --require-gpus 4 \
  --attention-backend flash3 --output "$AR_RUN_ROOT/ENVIRONMENT.json"
uv run --frozen python - "$AR_RECIPE" "$AR_RUN_ROOT" <<'PY'
import copy, json, os, pathlib, sys, yaml
recipe, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
assert 'H100' in json.loads((root/'ENVIRONMENT.json').read_text())['gpu']
m = json.loads((pathlib.Path(os.environ['DATA_ROOT'])/'manifest.json').read_text())
assert m['release_manifest_sha256'] == 'fe1ac0657085ab19fe6f56786006e9eb004ca66bc6c5b81dfd8e6bc3dcfda6ff'
assert {k: m['sources'][k]['train']['records'] for k in ('uniref90','mgnify','omg_img')} == {'uniref90':2430914,'mgnify':1437829,'omg_img':3240726}
sys.path.insert(0, os.environ['EXTERNAL_SRC'])
from autoresearch_esm.paper_contact_runtime import ContactDataset
assert ContactDataset(pathlib.Path(os.environ['CONTACT_ROOT'])).manifest_receipt.manifest_sha256 == 'c135bc806b1a282ea3d38651d55e0cc799578047ca12855c518d77a9274e9ce3'
original = yaml.safe_load(recipe.read_text())
assert 162137610 <= original['expected_parameter_count'] <= 179204726
assert len(original['stages']) == 1 and original['stages'][0]['name'] == 'stage1'
for seed in (42, 43):
    config = copy.deepcopy(original)
    config.pop('max_steps', None)
    config.pop('periodic_evaluation_command', None)
    config.update(seed=seed, max_model_tokens=24200224761, schedule_steps=100000,
                  learning_rate=5e-4, weight_decay=0.01, warmup_steps=1000,
                  attention_backend='flash3', expected_world_size=4, walltime_seconds=57600,
                  stage1_cooldown_fraction=0, checkpoint_interval=0, periodic_evaluation_interval=0)
    config['stages'][0].update(context_length=512, micro_batch_size=64, gradient_accumulation=4,
                              mixture={'uniref90':0.36,'mgnify':0.11,'omg_img':0.54})
    (root/f'config-{seed}.yaml').write_text(yaml.safe_dump(config))
PY
for AR_SEED in 42 43; do
  AR_RUN="$AR_RUN_ROOT/seed-$AR_SEED"
  uv run --frozen python -m torch.distributed.run --standalone --nproc-per-node=4 \
    -m nano_protein.train --config "$AR_RUN_ROOT/config-$AR_SEED.yaml" \
    --data-root "$DATA_ROOT" --output-root "$AR_RUN" --walltime-seconds 57600
  uv run --frozen python - "$AR_RUN/TRAINING_COMPLETE.json" <<'PY'
import json, pathlib, sys
r = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert r['stop_reason'] == 'max_model_tokens' and r['target_model_tokens'] == 24200224761
assert r['model_token_budget_reached'] and r['model_tokens'] >= r['target_model_tokens']
assert 0 <= r['model_token_overrun'] < r['last_optimizer_step_model_tokens'] <= 524288
PY
  CUDA_VISIBLE_DEVICES=0 uv run --frozen python -m nano_protein.evaluate \
    --checkpoint "$AR_RUN/checkpoint-final.pt" --data-root "$DATA_ROOT" \
    --output-root "$AR_RUN/eval-validation" --validation-batches 1024 \
    --validation-batch-size 4 --validation-context 512
  OUTPUT_ROOT="$AR_RUN" CHECKPOINT="$AR_RUN/checkpoint-final.pt" \
  EVAL_OUTPUT_ROOT="$AR_RUN/eval-p-at-l" EVAL_GPUS=0,1,2,3 \
  CONTACT_CHAINS=20775 CONTACT_SHARDS=16 bash runs/evaluate_p_at_l_parallel.sh
done
uv run --frozen python - "$AR_RUN_ROOT" <<'PY'
import json, math, pathlib, statistics, sys
root = pathlib.Path(sys.argv[1])
for filename, metric in [('eval-validation/VALIDATION_MLM.json','sequence_mean_nll'), ('eval-p-at-l/P_AT_L.json','p_at_l')]:
    values = [json.loads((root/f'seed-{s}'/filename).read_text())[metric] for s in (42, 43)]
    assert all(math.isfinite(v) for v in values), (metric, values)
    print(metric, 'per_seed=', values, 'mean=', statistics.mean(values), 'SD=', statistics.stdev(values))
PY
```

Compare both printed means with the matching baseline and previous recipe;
report paired per-seed deltas and each run's P@L interval from `P_AT_L.json`.
Chain-bootstrap intervals are separate from training-seed SD and must not be
averaged into an across-seed CI. The directional progress rule is not a significance test.
Freeze recipes before verification; do not select checkpoints using these scores.
The evaluation assets overlap research, so this checks budget transfer, not a blind holdout.
The historical 100k-step/single-seed leaderboard and the old 142M FFN results
retain their original labels and are not new-protocol results.
