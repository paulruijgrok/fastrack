# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Ridge signal-to-noise filter (`min_snr`).** `[ridge] min_snr` /
  `--ridge-min-snr` (default 0 = off) drops ridges whose mean centerline
  intensity is below `min_snr` x the frame's pixel noise. With `--preprocess`
  the noise comes from the preprocessing model, which now measures it
  (`PreprocessModel.noise_sd`, frame-difference MAD; old saved models are
  refitted once); otherwise it is estimated per frame. Fixes over-detection of
  background noise on dim movies, where fixed 8-bit contrast thresholds can't
  work because the per-movie stretch puts the same camera noise at ~0.5-5 grey
  levels. Plumbed through `FrameSource.noise_sd()` -> `Frame.noise_sd` (gliding)
  and `detect_filaments_in_stack(noise_sd=...)` (FASTplus). Docs:
  `docs/preprocessing.md` ("Detection on corrected data").

### Changed

- The `fast` filament cache tag for non-entropy detectors now includes a hash of
  the detector parameters (`_ridge_d<hash>...`), so re-running with different
  thresholds re-detects instead of silently reusing the previous filaments.
  Existing ridge caches are not reused (one-time re-detection). Entropy names are
  unchanged. Factored into `gliding.detection_tags()`.

- **In-pipeline preprocessing (`--preprocess`).** Opt-in illumination / low-SNR
  correction before detection, for both `fast` and `fastplus`: temporal-median
  background subtraction → flat-field (Gaussian gain field) → mild denoise →
  movie-wide full-range 8-bit stretch. Implemented at the frame-source seam as a
  `PreprocessedSource` wrapper (gliding) and applied to the split filament stack
  (directional); the model is fitted once per movie, saved next to the detection
  cache, and loaded by the worker processes. `--pp-model-frames N` fits on N
  evenly spaced frames for movies too long to hold in memory. New `[preprocess]`
  config section and `--pp-*` flags; recipe hashed into output-tree names and
  cache keys so raw and corrected runs never collide. `tools/preprocess_lowsnr.py`
  now delegates to the shared `fastrack.preprocess` module. Docs:
  `docs/preprocessing.md`.
- `FrameSource.read_raw()` — native-bit-depth page read (no 16→8→16
  quantisation) on both frame-folder and TIFF-stack sources.

- **FASTplus two-channel input registration.** `fastplus` now accepts two input
  forms: an already-registered RGB movie (default), or a **raw, spatially-packed**
  movie (`--register`) whose two channels occupy different regions of one camera
  frame. In `--register` mode FASTplus aligns + merges the channels into an RGB
  movie in-process using **optomerge**'s feature aligner (built for point-like
  heads vs line-like filaments), then analyses it.
- New flags: `--register`/`--no-register`, `--channel-order` (optomerge spatial
  layout, e.g. `top_green_fils_bottom_red_heads`), `--register-max-shift`
  (feature-aligner translation bound, px), `--register-frames` (frames used for
  alignment detection), and `--movie-suffix` (override movie discovery).
- `optomerge` is an **optional** dependency, isolated in the new
  `[plus-register]` extra (pinned to a specific commit). When it is absent and
  `--register` is requested, FASTplus warns and falls back to reading the input
  as a pre-registered RGB movie — the base install always works on aligned data.

### Changed

- `register_channels` defaults to **off** (input assumed pre-registered RGB). The
  registration parameters fold into the detection-cache key, so changing them
  invalidates cached detections.
- Movie discovery auto-broadens to all `*.tif` when `--register` is set (raw
  movies are not named `*RGB.tif`); overridable with `--movie-suffix`.
- `[plus]` no longer pulls in `optomerge` (moved to `[plus-register]`); it now
  only adds `tifffile` for RGB TIFF IO.

## [3.0.0] - 2026-06-08

Major restructure into a `src/`-layout package. **The numerical algorithm is
preserved verbatim** — a golden-master regression test reproduces the
pre-refactor `MEAN`/`SEM` outputs bit-for-bit on the example dataset. User-facing
commands (`fast`, `lima`, `stack2tifs`) are unchanged.

### Added

- Logical sub-packages: `core` (`frame`, `island`, `filament`, `link`,
  `motility`), `analysis`, `io`, `viz`, `pipelines`, `cli`.
- Pluggable strategy seams, each a small interface + registry with the existing
  algorithm as the first implementation:
  - **Detection** — `core.detection.Detector` (entropy/watershed impl).
  - **Tracking** — `core.tracking.Linker` (greedy impl, with the `legacy`
    partner-recovery variant).
  - **Output** — `io.movie.MovieWriter` (ffmpeg H.264), `io.stores.FilamentStore`
    (npy), and `io.export` (CSV).
- `datamodel.py`: `FilamentRecord` and cross-frame `FilamentTable` for
  storing/querying/exporting individual filaments across all frames.
- `config.py`: layered `Settings` (hardware / analysis / plotting / runtime) with
  `from_sources(*layers)` merging and a `from_toml` adapter.
- `registry.py`: the name→factory registry behind every seam.
- Test suite: unit tests (`test_config`, `test_registry`, `test_datamodel`,
  `test_tracking`) and a golden-master regression (`test_golden`) diffing against
  `tests/baseline/`.
- `tools/` helper scripts (`profile_frame`, `compare_fast_rank`,
  `compare_linking`, `capture_baseline`).

### Changed

- `pyproject.toml` switched to a `src/` layout; version bumped to `3.0.0`.
- The monolithic `motility.py` was split across the sub-packages; a
  `fastrack.motility` compatibility shim re-exports the old names.
- Replaced deprecated `skimage.morphology.square` with the identical numpy array.

### Preserved (deliberate parity quirks)

- The driver consults `dif_log_area_score_cutoff` (single `f`); the
  `diff_log_area_score_cutoff` value was never read in the original and remains
  unused, documented in `Motility.get_linker`.
- The dataset still emits a duplicate flattened-path output tree; the golden test
  compares the canonical tree only.

## [2.0.0]

- Python 3 port of Tural Aksel's FAST/FASTrack (FAST/LIMA/stack2tifs); replaced
  `ppss` with `multiprocessing` and `avconv` with `ffmpeg`.
- Performance: `fast_rank` (8-bit rank filters, default on), `--morph-contrast`
  (one-pass morphological-gradient contrast), all-cores default, H.264/MP4 movie
  output.
