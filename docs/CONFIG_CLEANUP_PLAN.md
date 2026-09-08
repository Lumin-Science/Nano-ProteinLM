# Configuration and code cleanup plan

Status: task/API cleanup implemented; historical presets have not been removed.
[program.md](../program.md) is the agent entry point, and
[171M validation loss](../task/171m-validation-loss.md) defines the benchmark.
Its [research shell script](../task/171m-validation-loss_ar.sh) calls the ordinary
training/evaluation APIs. Test of Progress is manual, with commands in the
[evaluation documentation](EVALUATION.md#manual-test-of-progress); there is no verification launcher.
The two-root `.env` and `configs/default.yaml` are implemented. The obsolete
root `program2.md` is preserved in [the archive](archive/program2.md).

## Target configuration folder

Keep four maintained presets:

| File | Role | Source / architecture |
|---|---|---|
| `configs/default.yaml` | Setting 3, default training and research starting point | Current `program2_h100_100k/r10_sqrtloss.yaml`; 170,559,856 parameters |
| `configs/esmc-171m-original.yaml` | Original ESMC-like AdamW comparator | 24 layers, width 768, 12 heads; 170,671,168 parameters |
| `configs/esmc-300m-original.yaml` | Original ESMC-like 300M family preset | 30 layers, width 960, 15 heads; 332,997,184 parameters |
| `configs/esmc-600m-original.yaml` | Original ESMC-like 600M family preset | 36 layers, width 1152, 18 heads; 575,036,992 parameters |

Setting 3 retains Muon/AdamW parameter groups and their multipliers, RMSNorm,
learned residual routing, depth-scaled initialization, RoPE 10k, FFN width 2048,
batch balance and square-root masked-target weighting. It does not tie embeddings.
The original AdamW preset remains the comparator even though it is no longer the
default starting recipe. Details and measured results:
[best versus baseline](BEST_RECIPE_VS_BASELINE.md).

There are currently **29 training YAMLs**, including the new default, plus a scale-up manifest. The 600M
architecture already exists in code, but a 600M training YAML must be added.
The existing 300M original YAML is a single-stage 21k-step/4-hour pilot; its
`paper_reference` section is descriptive, not an implemented two-stage run.
Do not describe these as exact reproductions of the released models' full training.
Their base LR/WD use the disclosed transfer rule with explicitly assumed proxy
values; apply that same documented rule to 600M.

## Separate recipes from runs

- Presets own architecture, optimizer groups, backward-loss choices and baseline
  hyperparameters. Preserve Setting 3's completed recipe as the source of truth.
- Launch scripts own hardware/batch layout, budget, seeds, checkpoint/evaluation
  cadence and output paths. The research shell script declares fixed research
  settings; the owner uses explicit API arguments for manual verification.
- Save the fully resolved training config in every run. Generate seed and GPU
  variants there, instead of adding another permanent YAML per experiment.
- Use an explicit common Stage-1 profile for comparisons. Keep any full paper-style
  two-stage schedule as a separate execution profile; do not silently turn an old
  pilot preset into a claimed full paper reproduction.
- Keep candidate configs in campaign worktrees/run outputs; do not accumulate them
  as additional maintained presets on main.

## Migration order

1. Archive retired config bytes and the hash manifest with their matching reports,
   preserving paths inside the archive and immutable source revisions. Update live
   documentation links; keep historical measured results and audit receipts intact.
2. Finish consolidating the four presets; `configs/default.yaml` is already the
   default for direct examples, the research script and the speedrun helper.
   Move the current 100k-step H100 and Nibi launch details
   into execution scripts before removing their YAMLs. Preserve Nibi's evaluation
   every 10k steps and full final optimizer checkpoint.
3. The general speedrun, two-root `.env.example` and README now agree on Setting 3.
   The speedrun reads the configured attention backend, uses a 16-hour guard and
   prints the actual recipe. Retire or explicitly migrate the remaining specialized
   300M launch wrapper and old evaluation output defaults.
4. Remove obsolete 10k/12-hour variants, duplicate 300M best aliases, paired-seed
   research presets, scale-up variants and manifest from the active config folder
   after every live consumer has a replacement. Frozen remote checkouts and saved
   run configs remain usable for ongoing work and continuation.
5. Retarget config-dependent tests to the four maintained presets. Replace the
   old 300M best-alias equality test with default-recipe/parameter checks; preserve
   generic model, optimizer, schedule, evaluation and resume coverage.

The research script intentionally rejects 300M/600M candidates because this
research task has a fixed 171M size bound. Those presets remain available for
general training or a separately defined size-specific research campaign.

## Code cleanup assessment

| Area | Proposed treatment | Reason |
|---|---|---|
| Old campaign configs, duplicated launch wrappers and machine-specific eval commands | Remove/migrate after consumers and archives are updated | They are experiment records rather than maintained recipes |
| Original 300M `evaluation` and `paper_reference` config sections | Move to execution/docs | Current training does not consume them |
| `_OptimizerBundle.set_param_group_value` | Candidate for removal after a final reference check | No call sites found in current code |
| `muon_initial_momentum` fallback alias | Review historical config/checkpoint use before removal | No current preset sets it; compatibility may still matter |
| Tied embeddings and configurable FFN width | Retain model/checkpoint support | Recorded models require these to load, even though neither is a default experiment |
| `tiny`, math attention, FA2 and FA3 | Retain | Tests, numerical reference, environment qualification and the two hardware profiles use them |
| AdamW/LayerNorm and Muon/RMSNorm/routing/init/balance/sqrt loss | Retain | Required by the original comparators and Setting 3 |
| Resume, optimizer state portability and periodic evaluation | Retain | Required for Nibi runs and later continuation on four GPUs |
| Gradient checkpointing, compilation and Stage 2 schedules | Audit separately; retain for now | Being disabled in current YAMLs does not prove they are unused; larger models may need them |
| Data, contact/PCORE evaluation, probes and scoring caches | Retain frozen | Removing training presets does not authorize changing scientific evaluation |

Validate the cleanup with resolved-config comparisons, model parameter counts,
launcher preparation, focused model/optimizer/resume tests and broken-reference
checks. No new training is needed merely to move files; a GPU smoke test is needed
if execution behavior changes. Do not delete checkpoints, data caches, reports or
code solely because an option is disabled in the default recipe.
