# Working in NanoProteinLM

- Ordinary research and AutoResearch use the same training and evaluation APIs in `src/nanoprotein/`. Keep `src/` to code they need; put other tools in `.dev/`.
- `docs/AUTORESEARCH.md` is the AutoResearch protocol; the README and `tasks/` restate it, and other docs link to it. Keep search policy in the `autoresearch/` programs, not in tasks. For research changes, read the selected task and keep its evaluation, data, budget and model-size boundaries. Final evaluation is owner-run.
- `.dev/` is local except `LOG.md`, `TODO.md` and `report/`; keep experiment records, receipts, cluster launchers, tests and maintenance scripts there. The local `.dev/scripts/build_clean_starter.py` builds the released starter (`autoresearch-v*` tags); never let it ship improvements or prior findings.
- `.env` holds only `DATA_ROOT` and `OUTPUT_ROOT`. Write run artifacts to `OUTPUT_ROOT`, and preserve historical results and checkpoint compatibility.
- When moving files, update the affected commands and links. Run the relevant tests in `.dev/tests/`.

## Writing docs

- Write README and docs edits as plain Markdown, without review tags.
- Treat unresolved `<aitofix>` notes as fix requests; once a fix is verified, add `resolved` and append `Fixed:` with what changed.
- Keep each Markdown or LaTeX prose paragraph on one source line.
