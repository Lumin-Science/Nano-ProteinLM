# Working in NanoProteinLM

- `src/nanoprotein/`: models, training, data preparation, evaluation and reusable CLI code.
- `configs/`: `autoresearch/` holds search-setting recipes (global batch 256) and `test-100k/` holds final-evaluation recipes (global batch 1,024); each has the plain `esmc-171m` reference and `nanop-best-171m-round1`/`round2`. Keep settings in code or presets.
- `runs/`: readable setup and training commands (`setup.sh`, `speedrun.sh`); keep reusable logic in `src/`.
- `autoresearch/program.md`: reusable research-loop rules for the agent.
- `tasks/`: scientific task definitions and `<objective>_ar.sh` measurement commands.
- `docs/`: methods and usage; `docs/AUTORESEARCH.md` is the shared protocol; the README and task files restate it, and other docs link to it. `.dev/tests/`, `.dev/scripts/` and `.dev/reports/`: regression tests, maintenance tools and curated evidence. Historical presets live in `.dev/configs/archive/`.
- `.dev/scripts/build_clean_starter.py` builds the history-free AutoResearch starter from the plain reference, published as a release tag such as `autoresearch-v0`; keep improvements and prior findings out of what it ships.
- `.env`: only `DATA_ROOT` and `OUTPUT_ROOT`; defaults are ignored `data/` and `outputs/`.

Use the same training and evaluation APIs for ordinary research and autoresearch. For research changes, read the selected task and preserve its evaluation, data, budget and model-size boundaries. The agent reviews compliance; task scripts express the measurement commands. Keep loop policy in `autoresearch/program.md` and scientific settings in the task/code. Final evaluation is owner-run.

Update affected commands and links when moving files, and run relevant checks in `.dev/tests/`. Preserve historical results and checkpoint compatibility; write new run artifacts to `OUTPUT_ROOT`.

## General Rules

- Keep `src/` limited to code necessary for ordinary research and evaluation. Other development tools belong in `.dev/`; the design principle is "less is more".
- Mark each agent-authored or edited paragraph in `README.md` and `docs/` with its own `<div class="ai">` / `</div>` wrapper on separate lines, with blank lines around the Markdown inside. Never group multiple paragraphs or an entire section/document in one wrapper. Mark changed headings, tables and code blocks separately; use `<span class="ai">...</span>` for inline text or individual list/table items. Keep wrappers outside copyable commands, mark only changed content, and leave each wrapper for the owner to review and remove independently. The tracked `.vscode/settings.json` and `.vscode/ai-review.css` display these changes in blue in the local VS Code Markdown preview; see [AI review](docs/AI_REVIEW.md).
- Treat unresolved `<aitofix>...</aitofix>` notes in the requested files as owner fix requests; read the nearby paragraph for context and ignore examples inside code blocks and notes with the `resolved` attribute. After implementing a note and verifying the result, add `resolved` to its opening tag, preserve the original request and append a brief `Fixed:` explanation. Leave incomplete or blocked notes unresolved and retain all notes for owner review; do not resolve one merely because it was read or discussed.
- When writing Markdown or LaTeX, keep each prose paragraph on one source line.
