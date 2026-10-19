# 2026-10-19 — Preview mode (`fast-preview`): a few-seconds quick look at the microscope

Mirror of the Notion session log: *2026/10/19 Preview mode (fast-preview)*
(hub: **Fastrack** — <https://app.notion.com/p/39edaf1f293b81179192ea41a1110d6b>).

## TL;DR

- New `fast-preview` command + `pipelines/preview.py` on branch **`preview-mode`**
  (commit `de8e483`). From 20 frames at 2x2 binning it reports **velocity,
  length distribution, stuck %** and **ranks image-quality metrics** (SNR,
  focus, flatness, saturation, background, density, stuck, floppiness,
  bleaching, displacement/length) worst first with advice, plus a GOOD /
  MARGINAL / POOR verdict. JSON + dashboard PNG + summary CSV per run.
- **Speed (Taka, 38 movies, 2 cloud cores):** median **5.7 s**, max 14 s (the
  densest movie, ~530 filaments/frame). Detection ≈ 60 % of the time and
  scales with cores, so the Mac should be well under 5 s.
- **Accuracy vs the full analysis:** velocity +0.4 % vs length-matched MVEL
  (IQR −2.8 … +2.9 %), mean length +0.8 % (r = 0.93), stuck r = 0.77.
- **Two findings that also matter for the full pipeline:**
  1. Background subtraction removes stuck filaments from the frames, so the
     tracking-based stuck % under-counts. Preview counts them directly on the
     background image.
  2. A short-window temporal **median** background contains slowly-moving
     filaments and cuts them short (synthetic: half length, half velocity).
     Preview uses the per-pixel **2nd-smallest** value instead.
- **Linker speed-up** (shared with the full pipeline): vectorized candidate
  gates in `GreedyLinker.link` and `Motility.wire_frame_links`; bit-identical
  results, dense 20-frame linking 14.4 s → 2.1 s.

## Detailed narrative

**Design.** Fastest defaults, everything overridable: `-n 20` frames (or
`--latest`: newest 20, skipping the newest file, which may still be being
written), `--bin 2` (ridge line widths and min length scaled by 1/bin),
`ridge-fast` detector with `min_snr 3`, a multiprocessing pool, no caches.
Preprocessing model fitted on the preview frames only. Linking and paths use
the standard FAST code (`Motility.make_frame_links` / `process_frame_links`)
and the stats come from `plot_length_velocity(print_plot=False)`, so velocity
filtering matches the full analysis.

**Profiling → speedups.** On the dense movie E11K_300nM_2: detection 7.8 s,
linking 14.4 s at first. The linker evaluated its distance/length gate with a
per-filament Python comprehension → vectorized (2.3 s). Then
`wire_frame_links` (gap closing) did the same per link → vectorized with a
per-frame array cache (link 3.1 → 2.05 s). Both verified bit-identical (same
velocity/length/point counts on Taka; full suite green).

**Focus.** At bin 2 the ridge width barely responds to defocus (391 → 470 nm
for σ = 3 px blur), so focus is measured on a full-res 512 px centre crop of
the middle frame (in focus ≈ 265–275 nm; σ = 3 px blur ≈ 367 nm). Other
simulations: cutting the signal 8× drops SNR ~12 → ~4.

**Background / stuck (the main correctness fix).** A synthetic movie (6 filaments
gliding 4 px/frame, 3 stuck, 60 px long) gave half the true velocity and length
and counted moving filaments as stuck: with 20 frames a filament moving only
1.3 lengths covers pixels in >half the frames, so the temporal median contains
it. Switched to the per-pixel 2nd-smallest value of the raw frames (must be the
unsmoothed frames — `apply()` denoises after subtracting, so a pre-smoothed
background leaves ridges with negative flanks). Stuck filaments are detected on
that same image (smoothed, flattened, noise = MAD of the image itself). Now
exact on the synthetic (323 nm/s, 3 stuck / 6 moving, ~5 µm). On Taka this
improved velocity agreement (+3.0 % → +0.4 %) and length (−2.5 % → +0.8 %);
median stuck dropped from 16 % to 10 %.

**Thresholds** (`QUALITY_SPECS`, good → poor): SNR 12 → 4, focus 300 → 380 nm,
flatness 0.8 → 0.5, saturation 0.01 → 0.5 %, background/signal 6 → 15,
density too few 1 → 0.3 and crowded 8 → 14 per 100 µm², stuck 15 → 40 %,
floppy 15 → 35 %, bleaching 1 → 3 %/frame, step/length 0.6 → 1.0.
Taka verdicts: 29 GOOD, 3 MARGINAL, 6 POOR.

**Floppiness caveat.** Floppiness (tracks with net/total centre-of-mass path
< 0.5) correlates with density (r ≈ 0.6): mislinks in crowded fields. Switching
from the contour midpoint to the centre of mass did not change this (kept for
robustness). E11K_150nM_2 (34 % floppy, 35 % stuck, bimodal velocity, many
filaments both static and moving = pivoting) looks like a genuine
poor-attachment case.

## Mac timing (afternoon, same day)

- First Mac runs used `fastrack-ridge` without `ridge-fast` → fell back to
  `ridge`. Identical numbers to the cloud with `ridge` (deterministic across
  machines), but **focus width is ~55–60 nm larger with `ridge` than
  `ridge-fast`** (342 vs 284 nm on E11K_300nM_1) → false MARGINAL focus flags.
  The "numerically identical" claim for ridge-fast holds for contours, not
  widths. Thresholds are calibrated for `ridge-fast`.
- Accidental `pip install -e .` in Anaconda **base** upgraded numpy 1.26 → 2.4
  and scipy (breaks numba 0.57 there); fix given: reinstall numpy 1.26.4 /
  scipy 1.11.1, uninstall opencv-python + fastrack from base.
- `fastrack-ridge` runs **x86_64 under Rosetta** on an arm64 Mac (8P + 4E).
  Worker scaling was poor (1/4/8 workers: 13.1/8.4/7.3 s): macOS spawns workers,
  and each imported the whole gliding pipeline (~2.5–3 s).
- Fix 1 (commit `564d0b3`): import-light worker `core/detection/worker.py` +
  pool initializer → Rosetta 1/4/8/12: 12.2/6.6/5.5/5.7 s.
- Fix 2: native env `fastrack-arm` (Miniforge, osx-arm64; activate via
  `eval "$(~/miniforge3/bin/conda shell.bash hook)"` because the shell's conda
  is Anaconda's) → 1/4/8/12: 12.0*/4.5/**3.8**/4.1 s (*first run, numba compile).
- Default workers now = performance cores on Apple Silicon (`default_workers()`),
  since the efficiency cores made 12 slower than 8.

## Commands

```bash
fast-preview /path/to/movie                    # one movie
fast-preview /path/to/slide -o outputs/preview/slide
fast-preview /path/to/movie --latest           # while acquiring
fast-preview /path/to/movie --config config.taka.toml
```

Validation scripts (cloud, not in the repo): `validate.py` (preview on all
slides → JSON), `compare.py` (vs `summary_per_movie.csv` of the full run).

## Files / commits

| file | change |
|---|---|
| `src/fastrack/pipelines/preview.py` | new pipeline |
| `src/fastrack/cli/__init__.py`, `pyproject.toml` | `fast-preview` entry point |
| `src/fastrack/core/tracking/greedy.py` | vectorized candidate gate |
| `src/fastrack/core/motility.py` | vectorized gate in `wire_frame_links` (+ `_gate_arrays` cache) |
| `tests/test_preview.py` | 7 tests (helpers, scoring, synthetic end-to-end, `--latest`, CLI) |
| `docs/preview.md`, README, CHANGELOG, CLAUDE.md | docs |

Commit `de8e483` on `preview-mode` (from main `7ee7fbb`). Suite: 162 passed,
5 skipped.

## Open items / next steps

- Make the focus metric detector-independent (measure the width from the image
  profile across each ridge) — `ridge` vs `ridge-fast` widths differ by ~60 nm.
- Consider moving day-to-day FASTrack use to the native `fastrack-arm` env
  (run the test suite there first).
- Optional `--watch` mode with a persistent pool for use while acquiring.

- Time it on the Mac (more cores) and at the microscope; decide whether more
  speed is needed (options: fewer frames, bin 3, sending uint8 instead of
  float images to the pool, persistent worker pool between runs, skipping the
  focus crop).
- Tune `QUALITY_SPECS` on other setups / deliberately bad acquisitions.
- Floppiness vs crowding: consider excluding tracks near other filaments.
- Consider the low-order background and direct stuck counting for the full
  pipeline (its stuck % under-counts for the same reason).
- Push `preview-mode`, open a PR, merge after review.
