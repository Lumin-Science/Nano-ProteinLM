# Working in NanoProteinLM

- `src/nanoprotein/`: models, training, data preparation, evaluation and reusable CLI code.
- `configs/`: research presets; `default.yaml` is Setting 3. Keep settings in code or presets.
- `runs/`: readable setup and training commands (`setup.sh`, `speedrun.sh`); keep reusable logic in `src/`.
- `tasks/`: task definitions and `<objective>_ar.sh` measurement commands. `program.md` directs the research agent.
- `docs/`: methods and usage. `.dev/tests/`, `.dev/scripts/` and `.dev/reports/`: regression tests, maintenance tools and curated evidence.
- `.env`: only `DATA_ROOT` and `OUTPUT_ROOT`; defaults are ignored `data/` and `outputs/`.

Use the same training and evaluation APIs for ordinary research and autoresearch.
For research changes, read the selected task and preserve its evaluation, data,
budget and model-size boundaries. Keep search strategy out of task definitions;
Test of Progress is run manually by the owner.

Update affected commands and links when moving files, and run relevant checks in
`.dev/tests/`. Preserve historical results and checkpoint compatibility; write new
run artifacts to `OUTPUT_ROOT`.
