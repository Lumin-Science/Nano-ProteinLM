# Research repository layout

Use the same scientific implementation for ordinary research and autoresearch.
Standardize folder responsibilities and entry points; organize implementation
modules naturally for each project.

| Path | Responsibility |
|---|---|
| `README.md` | Goals, setup, training/simulation and evaluation, tasks, headline results |
| `program.md` | Direct an agent to a selected task; leave search strategy to the agent |
| `.env.example` | Two local paths: `DATA_ROOT` and `OUTPUT_ROOT` |
| `src/` | Scientific implementation, data preparation, evaluation and reusable CLI code |
| `configs/` | Optional reusable research presets; avoid duplicating executable settings |
| `runs/setup.sh` | Prepare the environment and required inputs |
| `runs/speedrun.sh` | Run a documented starting experiment using the ordinary APIs |
| `tasks/<objective>.md` | Background, Autoresearch protocol, Test of Progress |
| `tasks/<objective>_ar.sh` | Readable commands obtaining one research measurement |
| `docs/` | Methods, architecture and detailed usage |
| `.dev/tests/` | Developer regression tests, including tests used by qualification |
| `.dev/scripts/` | Plotting, release preparation and maintenance tools |
| `.dev/reports/` | Committed experiment summaries and provenance |
| `data/` | Ignored default for inputs, fixtures and data caches (`DATA_ROOT`) |
| `outputs/` | Ignored default for runs, checkpoints, builds and logs (`OUTPUT_ROOT`) |

Python code uses `src/<package>/`; native code and pinned upstream source may
also live under `src/`. In particular, OpenMM remains pinned at `src/openmm/`.
Use the project's package/build manifest and lockfile to capture dependencies.
Do not create an empty `configs/` directory when settings already live in code.

Launch scripts call the standard APIs. Reusable logic lives in `src/`; task
scripts fix the measurement protocol without implementing an agent loop.
Task documents define the score, budget, command and protected boundaries.
Concrete settings belong in code or presets. Test of Progress is owner-run;
a separate verification wrapper is optional, not required by this layout.

Evaluation and scientific accuracy gates remain runtime code. Their regression
tests may live in `.dev/tests/` and stay protected when a task requires them.
New run artifacts go to `OUTPUT_ROOT`. Only curated evidence is committed to
`.dev/reports/`; historical records retain their original paths and hashes.
A path migration that changes a benchmark's protected hashes requires a fresh
baseline contract, not rewritten historical evidence.
