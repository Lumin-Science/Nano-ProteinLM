# Development archive

This directory keeps research state out of the supported `main` surface.

| Path | Contents |
|---|---|
| `program.md` | AutoResearch contract for four GPUs and a two-hour training-loop budget |
| `configs/` | AutoResearch, long-run, and proposed benchmark configurations |
| `runs/` | non-default experiment launchers |
| `docs/` | roadmaps, experiment narratives, and proposed evaluation designs |
| `plans/` | benchmark formulation and validation packets |
| `report/` | NeurIPS-style paper source |
| `results/` | completed-run logs, reports, and compact scientific receipts |

Production code, the frozen evaluator, the four-hour baseline configuration,
tests, and release metadata stay at the repository root. Moving a research
change from `dev/` into the production surface requires metric parity tests and
a focused commit that does not pull report or experiment state with it.
