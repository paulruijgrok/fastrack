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
  static top junk); documented in `docs/preprocessing_notes.md`.
- Set up this **session-journaling SOP**.
- **Solved the taka detection problem** with a standalone preprocessor
  (`tools/preprocess_lowsnr.py`). The *dominant* issue turned out to be FASTrack's
  16->8-bit `cv2.IMREAD_GRAYSCALE` read crushing the low-valued frames to ~2 grey
  levels before detection; median-subtract + flat-field + denoise + full-range
  8-bit stretch fixes it. All 8 slide_01 movies now yield velocities (were empty):
  MVEL ~1.5-1.7 um/s. Parallelised across movies (~5x). Script uncommitted.

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

## taka preprocessing solution (standalone script — option B)

User chose the standalone-script route: get it working well on this data first,
generalise to a cross-pipeline stage later.

**Key extra root cause found:** FASTrack reads frames with
`cv2.imread(..., IMREAD_GRAYSCALE)`, which downconverts 16-bit TIFFs to 8-bit by a
divide-by-256 bit-shift. The taka raw values only span ~295-905, so a raw frame
collapses to **2 grey levels** before detection — the filament signal is destroyed
regardless of the gradient. Verified: `cv2` read of a raw frame → `n_unique == 2`.
So the corrected output must be **intensity-stretched to the full 8-bit range**,
not merely gradient-corrected.

**`tools/preprocess_lowsnr.py`** (new, uncommitted). Per movie, in float at full
16-bit precision, then a final 8-bit cast:

1. read raw frames at full precision (`tifffile`, not the crushing cv2 path);
2. subtract the temporal-median background (static bg / gradient / stuck junk);
3. flat-field (divide by `gaussian_filter(mean, sigma=50)`, normalised);
4. mild denoise (`gaussian sigma=0.8`); clip negatives to 0;
5. **stretch** 0 -> the movie-wide 99.9th percentile onto 0-255; cast to uint8.

Writes **drop-in** MicroManager movies (same folders/filenames + copied
`metadata.txt`), fail-isolated per movie, resumable, and **parallel across movies**
(`ProcessPoolExecutor`, inner BLAS/OpenMP threads capped to 1 per worker).

**Validation.** `cv2` read of a corrected frame → **256** grey levels (vs 2); ridge
on it → **746 filaments, full-field** (top/bottom 0.68 vs 0.12 raw). End-to-end
`fast --detector ridge` on the corrected `E11K_300nM_2` (480x480 crop, sandbox) →
**172 trajectories** where the raw movie produced nothing. Parallel run: 6.4s vs
32s serial on 4 cores, output **byte-identical** to serial (md5 verified).

**8-movie results** (user ran `fast --detector ridge` on the full preprocessed set):
all 8 now produce data. MVEL-filtered ~1.5-1.7 um/s (E11K avg ~1593, WT avg ~1607
nm/s — no clear protein or 300/500 nM trend); top-5 ~2.1-2.4 um/s; stuck 0-2.7%.
Mean filament length differs: WT ~3.0 um vs E11K ~2.0 um. `WT_500nM_1` thin (83
points). See `outputs/slide_01_preprocessed__..._det_ridge/combined/MEAN_values.txt`.

**Commands.**

```bash
conda activate fastrack-ridge
cd ~/Documents/Claude/Projects/FASTrack
python tools/preprocess_lowsnr.py \
    -i 'examples/taka/04302026 Motility Experiment/slide_01' \
    -o 'examples/taka/04302026 Motility Experiment/slide_01_preprocessed' --force
fast -d 'examples/taka/04302026 Motility Experiment/slide_01_preprocessed' -f \
    --detector ridge --ridge-line-widths 3 5 \
    --ridge-low-contrast 18 --ridge-high-contrast 50 --ridge-min-len 12
```

Tuning knobs: `--flat-sigma` (residual gradient), `--clip-hi-pct` (stretch),
`--ridge-*` thresholds; `-j N` for worker count.

## Open items / next steps

- **Calibration before trusting the numbers:** velocities use `pixel_size_nm =
  80.65` and the frame interval from `metadata.txt` — confirm the pixel size
  matches Taka's objective/camera (all velocities scale with it). Also
  `utrophin(nM)` prints 0.000 (folder-name concentration not parsed) — cosmetic.
- **`fast` vs `lima`:** the nM in the names may be a utrophin *load*; if so `lima`
  (loaded motility) may be the intended pipeline rather than `fast` (unloaded).
- **WT vs E11K mean length** (~3.0 vs ~2.0 um) — check whether real or a
  detection/threshold artifact.
- **Commit `tools/preprocess_lowsnr.py`** (new, uncommitted).
- Fold the standalone preprocessing into the cross-pipeline strategy later
  (option A) once it's proven on more datasets — see `docs/preprocessing_notes.md`.
- Repin the optomerge dependency from a commit to a tagged release when available.
- Expose optomerge's robust calibrator / per-set alignment reuse through FASTplus.
- Earlier docs/journaling commit was handed to the user to run on their Mac
  (`docs/preprocessing_notes.md` + assets, `CLAUDE.md`, `journals/`); confirm it
  landed. Stray 0-byte files + `examples/taka/_diag*/` couldn't be removed from the
  sandbox (mount quirk) — delete on the Mac.
