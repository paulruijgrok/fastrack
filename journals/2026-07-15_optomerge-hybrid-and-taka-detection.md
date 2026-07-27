# 2026-07-15 — optomerge hybrid registration + taka low-SNR detection diagnosis

Mirror of the Notion session log:
<https://app.notion.com/p/39edaf1f293b8187b2d3e650c3f759db>
(hub: **Fastrack** — <https://app.notion.com/p/39edaf1f293b81179192ea41a1110d6b>)

## TL;DR

- Resumed FASTrack; confirmed local `main` == `origin/main` (`0761913`); reviewed
  env setup (conda env **or** repo `.venv`; `pip install -e '.[plus]'`).
- **optomerge integration — shipped, commit `16e0c90`, pushed.** Rewrote the
  `io/dual_channel` adapter (previously written against a guessed, non-existent
  optomerge API) into a **hybrid**: FASTplus takes either a pre-registered RGB
  movie (default) or a raw spatially-packed movie it aligns+merges in-process via
  optomerge's feature aligner (`--register`). optomerge is an **optional** dep
  (`[plus-register]`, pinned) that falls back to RGB when absent.
- **Ran** the gold `unloaded_motility` set and the new `examples/taka/.../slide_01`
  set; fixed environment issues (venv/conda stacking; numba/llvmlite source-build
  failures → conda-forge `fastrack-ridge` env).
- **Diagnosed** the taka detection problem (low contrast + illumination gradient +
  static top junk); documented in `docs/preprocessing_notes.md`, implementation
  deferred to a cross-pipeline strategy.
- Set up this **session-journaling SOP**.

## Environment notes

- Repo: `~/Documents/Claude/Projects/FASTrack`. Two working environments in play:
  - `.venv` (repo virtualenv) — main env; full test suite passes here.
  - `fastrack-ridge` (conda-forge env) — created for the `ridge` detector because
    numba/llvmlite would not `pip`-build from source (broken MacPorts `cmake` at
    `/opt/local`, missing `iomp5`/OpenMP). `conda create -n fastrack-ridge -c
    conda-forge python=3.11 numba llvmlite` then `pip install -e '.[ridge]'`.
  - **Do not stack them** — activating `.venv` on top of the conda env left
    `.venv` shadowing conda on PATH and installs went to the wrong place. One
    environment at a time; `(base)` in the background is fine.

## optomerge integration (commit `16e0c90`)

Real optomerge is an OOP API (`RawMovie`/`RGBMovie`, `Channel`, `Aligner` /
`PhaseCorrelationAligner` / `FeatureDistanceAligner`, `Transform`,
`MergePipeline`, `Settings`, `Calibrator`s). The old adapter guessed
`optomerge.register()` / `optomerge.apply_transform()` — neither exists.

Scope evolved over three iterations with the user, landing on the **hybrid**:

- `TwoChannelMovie(register=False)` → read RGB, split colour planes (no optomerge).
- `TwoChannelMovie(register=True)` → drive `MergePipeline(...).run()` with the
  feature aligner via `Settings(method="feature", channel_order=...,
  max_shift=..., projection_frames=...)`; convert the `(H,W,3,T)` float RGB to a
  `(T,H,W,3)` uint8 stack **bit-identical to optomerge's `save_rgb_tiff`**; split.
- optomerge absent + `register=True` → warn and fall back to RGB.

Verified end-to-end against real optomerge on a synthetic known-shift packed movie
(feature aligner recovered the injected shift; red=heads / green=fils).

Other changes: CLI `--register` / `--channel-order` / `--register-max-shift` /
`--register-frames` / `--movie-suffix`; registration params folded into the
detection-cache key; **movie-discovery fix** (raw movies aren't `*RGB.tif`, so
`--register` auto-broadens discovery to `*.tif`); `optomerge` moved to a pinned
`[plus-register]` extra (`[plus]` = `tifffile` only); 7 new tests; README /
`docs/fastplus.md` / PR notes / `config.plus.example.toml` / CHANGELOG updated.

Test suite on macOS: **115 passed, 2 skipped** (skips are the ridge/numba tests;
heavy `test_golden` / `test_stack_pipeline` pass). Committed `16e0c90`, pushed to
`origin/main`.

## taka dataset detection diagnosis (deferred)

Dataset: `examples/taka/04302026 Motility Experiment/slide_01` — 8 MicroManager
movies (E11K/WT × 300/500 nM × 2), 60 frames, 1002×1004 uint16, 561 nm, under
`<movie>/Pos0/`.

Findings (full write-up in `docs/preprocessing_notes.md`):

- Empty velocity output is a **detection symptom**, not a writer bug — combined
  MEAN/SEM are header-only because too little tracked.
- Filaments are field-wide but **low contrast** (~120–150 counts over ~318 bg),
  with a strong **top→bottom illumination gradient** and heavy **static junk at
  the top**. Entropy's quality gate rejects these (needs >1000 span, 16-bit).
- Ridge on raw + defaults: 227 top vs 28 bottom (0.12) — reproduces the reported
  "only near the top" symptom.
- **Fix (demonstrated):** temporal-median subtraction + illumination flat-field +
  mild denoise, then ridge with `line_widths=[3,5]`, `low/high_contrast≈18/50`,
  `min_len≈12` → top/bottom **0.64**, full-field. Diagnostic figures in
  `docs/assets/taka_lowsnr_{preproc,detect,flatfield}.png`.
- Implementation **held** pending a comprehensive cross-pipeline preprocessing
  strategy (options A: bake into pipeline / B: standalone script; plus the
  quality-gate, cache-key, and whole-stack considerations noted in the doc).

## Open items / next steps

- Cross-pipeline preprocessing strategy (user scoping) → then implement the
  flat-field / temporal-median preprocessing + low-SNR ridge tuning.
- Repin the optomerge dependency from a commit to a tagged release when available.
- Expose optomerge's robust calibrator / per-set alignment reuse through FASTplus
  (not exposed yet).
- Uncommitted: `docs/preprocessing_notes.md` + `docs/assets/`, and the journaling
  files (`CLAUDE.md`, `journals/`, this file).
- Leftover `examples/taka/_diag/` couldn't be auto-removed (mount quirk) — images
  are preserved in `docs/assets/`; delete `_diag/` manually if desired.
