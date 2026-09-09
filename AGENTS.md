# Working in NanoProteinLM

- `src/nanoprotein/`: models, training, data preparation, evaluation and reusable CLI code.
- `configs/`: research presets; `default.yaml` is the improved recipe; `esmc/` holds AdamW references. Keep settings in code or presets.
- `runs/`: readable setup and training commands (`setup.sh`, `speedrun.sh`); keep reusable logic in `src/`.
- `autoresearch/program.md`: reusable research-loop rules for the agent.
- `tasks/`: scientific task definitions and `<objective>_ar.sh` measurement commands.
- `docs/`: methods and usage. `.dev/tests/`, `.dev/scripts/` and `.dev/reports/`: regression tests, maintenance tools and curated evidence. Historical presets live in `.dev/configs/archive/`.
- `.env`: only `DATA_ROOT` and `OUTPUT_ROOT`; defaults are ignored `data/` and `outputs/`.

Use the same training and evaluation APIs for ordinary research and autoresearch.
For research changes, read the selected task and preserve its evaluation, data,
budget and model-size boundaries. The agent reviews compliance; task scripts
express the measurement commands. Keep loop policy in `autoresearch/program.md`
and scientific settings in the task/code. Test of Progress is owner-run.

Update affected commands and links when moving files, and run relevant checks in
`.dev/tests/`. Preserve historical results and checkpoint compatibility; write new
run artifacts to `OUTPUT_ROOT`.
