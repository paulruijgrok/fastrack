# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
