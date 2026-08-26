# Development archive

This directory keeps research state out of the supported `main` surface.

| Path | Contents |
|---|---|
| `configs/` | proposed benchmark configurations; obsolete training presets are not archived here |
| `runs/` | non-default experiment launchers |
| `docs/` | roadmaps, experiment narratives, and proposed evaluation designs |
| `plans/` | benchmark formulation and validation packets |
| `report/` | NeurIPS-style paper source |
| `results/` | current Q9 evaluation evidence, split ledgers, and compact scientific receipts |
| `data/` | annotated raw-to-shard corpus builder, source pins, Slurm launchers, and processing report |

Production code, the frozen evaluator, supported production configurations,
and release metadata stay at the repository root. Pre-Q9 training receipts and
obsolete two-hour/16-hour presets are intentionally excluded from this archive.
The canonical AutoResearch program, end-to-end round runner, and experiment
ledger remain on the dedicated `auto-research` branch until a selected setting
is promoted. Moving a research change from `dev/` into the production surface
requires metric parity checks and a focused commit that does not pull report or
experiment state with it.
