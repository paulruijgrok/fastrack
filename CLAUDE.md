# FASTrack — project guide for Claude

Python 3 port of Tural Aksel's **FAST / FASTrack** for automated analysis of
in-vitro actin gliding-assay movies. `src/` layout, pluggable detection /
tracking / output seams behind a registry + `Settings`.

## Pipelines / CLIs

- `fast` — gliding (single-colour) analysis. Detectors: `entropy` (default),
  `ridge`, `ridge-fast`.
- `lima` — loaded in-vitro motility analysis (LIMA).
- `fastplus` — directional / two-colour polarity-aware analysis. Accepts either a
  pre-registered RGB movie or a raw spatially-packed movie aligned in-process via
  the optional **optomerge** feature aligner (`--register`).
- `stack2tifs` — TIFF-stack → per-frame converter.

## Environment

- Main env: repo `.venv` (`source .venv/bin/activate`), or a conda env. Install
  with `pip install -e '.[plus]'`; optional extras `[ridge]` / `[ridge-fast]` /
  `[plus-register]` (optomerge) / `[batch]` / `[plus-laptrack]`.
- The `ridge` detector needs numba/llvmlite; if they won't `pip`-build on macOS,
  use a conda-forge env (`conda create -n fastrack-ridge -c conda-forge
  python=3.11 numba llvmlite`, then `pip install -e '.[ridge]'`). Use **one**
  environment at a time — don't stack `.venv` on top of a conda env.

## Conventions

- Follow the **pipeline-engineering-standards** skill: run the full `pytest`
  suite before every commit; real commit/PR descriptions (what/why, verification,
  future work); sweep for dead code before committing; correctness → single-core →
  parallel; ship a fail-isolated batch mode for real pipelines.
- Docs live in `docs/`; the README is a map that links out to `docs/<topic>.md`.

## Session journaling (standard operating procedure)

At the end of any session that did non-trivial design/pipeline work (config/script
changes, pipeline debugging, running analyses, dependency/packaging changes),
record it — **default behaviour, don't wait to be asked**:

1. **Local journal** — a markdown file under `journals/` in this repo, named
   `journals/YYYY-MM-DD_<short-slug>.md`. Full detail: what changed and why, exact
   commands, files/commits touched, and an explicit *open items / next steps*
   section. This is the source of truth.
2. **Notion mirror** — a dated sub-page under the **Fastrack** hub page
   (<https://app.notion.com/p/39edaf1f293b81179192ea41a1110d6b>), titled
   `YYYY/MM/DD <short title>`, and add a link to it under the hub's *Session logs*
   section. Mirror the local journal's structure: **TL;DR**, **Detailed
   narrative**, a **Files / commits** table, and **Open items / next steps**, with
   a pointer back to the local journal file.

Keep both in sync. The dated title format is `YYYY/MM/DD` on Notion and
`YYYY-MM-DD` in the filename, matching the other project hubs in the workspace.
