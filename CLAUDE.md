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
- `fast-preview` — few-seconds quick look (velocity, lengths, stuck %, ranked
  image-quality metrics) on 20 binned frames; see `docs/preview.md`.

## Environment

- **Recommended (Mac): `fastrack-arm`** — a native Apple Silicon (osx-arm64)
  conda-forge env made with **Miniforge**. The shell's default `conda` is
  Anaconda's (Intel/x86_64, runs under Rosetta and can't see Miniforge envs), so
  activate it through Miniforge's conda:
  ```bash
  eval "$(~/miniforge3/bin/conda shell.bash hook)"
  conda activate fastrack-arm
  python -c "import platform; print(platform.machine())"   # arm64
  ```
  Created with `~/miniforge3/bin/conda create -n fastrack-arm -c conda-forge
  python=3.11 numba llvmlite`, then `pip install -e '.[ridge,ridge-fast,batch,test]' tifffile`.
  Full suite there: 172 passed, 2 skipped (golden masters included). Native is
  markedly faster than Rosetta (fast-preview 3.8 s vs 5.5 s on the densest Taka
  movie).
- Older envs: `fastrack-ridge` (Anaconda, x86_64 under Rosetta — works, slower)
  and the repo `.venv`. Use **one** environment at a time — don't stack `.venv`
  on top of a conda env.
- **Never `pip install` into Anaconda `base`** (check the prompt says the env you
  mean): an accidental `pip install -e .` there upgraded numpy/scipy and broke
  numba in base.
- Install extras: `[plus]`, `[ridge]` / `[ridge-fast]` (ridge detectors; need
  numba/llvmlite — use a conda-forge env on macOS), `[plus-register]`
  (optomerge), `[batch]`, `[plus-laptrack]`, `[test]`.
- Run the tests with `python -m pytest -q` (not bare `pytest`, which may pick up
  another environment's pytest).

## Conventions

- Follow the **pipeline-engineering-standards** skill: run the full `pytest`
  suite before every commit; real commit/PR descriptions (what/why, verification,
  future work); sweep for dead code before committing; correctness → single-core →
  parallel; ship a fail-isolated batch mode for real pipelines.
- Docs live in `docs/`; the README is a map that links out to `docs/<topic>.md`.
- **Figures** follow the plot theme (`viz/theme.py`, `viz/themes/*.toml`,
  catalogue in `viz/figures.py`; see `docs/plotting.md`): draw inside
  `as_theme(theme).figure("<name>")`, take colours / dpi / formats / options
  from the context, save with `fx.save`, never touch global rcParams. New
  figure → register it in `viz/figures.py` + a `[figures.<name>]` section in
  `default.toml` (tests check they agree, and that the docs table is current).

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
