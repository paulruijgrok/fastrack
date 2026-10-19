# Preview mode (`fast-preview`): a few-seconds quick look for use at the microscope

## Summary

Adds **`fast-preview`**, a quick-look analysis meant to run at the microscope
while acquiring, to decide whether the imaging conditions are good enough for
the full analysis. From 20 frames at 2x2 binning it reports **gliding velocity,
the length distribution and the stuck fraction**, and **ranks image-quality
problems worst first with advice** — focus, signal-to-noise, illumination
flatness, background, saturation, filament density (too few / crowded), stuck
and floppy filaments, bleaching, and displacement per frame (frame rate) —
ending in a GOOD / MARGINAL / POOR verdict.

**Speed:** 3.8 s on the densest Taka movie (~530 filaments/frame) on an Apple
Silicon Mac with a native (arm64) environment; ~2–3 s on typical movies.
**Accuracy** vs the full analysis (38 Taka movies): velocity +0.4 % against the
length-matched MVEL, mean length +0.8 % (r = 0.93), stuck r = 0.77.

Also includes a **linking speed-up that benefits the full pipeline**
(bit-identical results, dense 20-frame linking 14.4 s → 2.1 s).

```bash
fast-preview /path/to/movie                  # one movie, or a folder of movies
fast-preview /path/to/movie --latest         # newest frames while still acquiring
fast-preview /path/to/movie --config config.taka.toml
```

Writes a per-movie JSON report, a dashboard PNG (overlay of moving/stuck
filaments, length and velocity histograms, quality bars) and
`preview_summary.csv`. Docs: [`docs/preview.md`](docs/preview.md).

## What changed

- **`pipelines/preview.py` + `fast-preview` CLI** (`cli/__init__.py`,
  `pyproject.toml`). Fastest settings by default, all overridable: `-n 20`
  frames (or `--latest`: the newest 20, skipping the newest file, which may still
  be being written), `--bin 2` (ridge line widths / min length scaled), the
  `ridge-fast` detector with `min_snr 3` (falls back to `ridge`), a process pool
  (`-j`), no caches. Linking, paths and statistics use the standard FAST code,
  so velocity filtering matches the full analysis. Fail-isolated per movie.
- **Low-order background instead of the temporal median.** Over only 20 frames,
  a filament that moves less than ~2 of its lengths covers some pixels in more
  than half the frames, so a median background contains it and subtracting it
  cut moving filaments short (synthetic movie: half the true length and
  velocity). Preview uses the per-pixel 2nd-smallest value of the frames.
- **Stuck filaments counted directly.** Background subtraction removes stuck
  filaments from the frames, so tracking under-counts them (the full pipeline
  does too). Preview detects them on that low-order background image; "static"
  means moved less than ~1 filament length during the window (~75 nm/s at
  20 frames × 1 s for 1.5 µm filaments, close to the stuck threshold).
- **Focus measured from the image** (`analysis/profile_width.py`). The
  detector's own width estimate depends on the implementation (`ridge` reads
  ~60 nm wider than `ridge-fast`). Instead, for the 40 longest filaments in a
  full-resolution centre crop, perpendicular profiles are taken from the raw
  pixels, re-centred to sub-pixel precision, median-combined and fitted with a
  Gaussian on a sloped background → FWHM. Profiles crossing another filament are
  dropped; the window widens for blurred filaments. Computed while the workers
  detect the other frames.
- **Import-light detection worker** (`core/detection/worker.py`) + pool
  initializer. On macOS, workers are spawned and re-import the module holding
  the task function; from `fastrack.pipelines` that pulled in the whole gliding
  pipeline (~3 s per run under Rosetta). Default worker count = performance
  cores on Apple Silicon (12 workers on an 8P+4E Mac were slower than 8).
- **Vectorized candidate gates** in `GreedyLinker.link` and
  `Motility.wire_frame_links` (array arithmetic instead of per-filament Python
  comprehensions; same arithmetic and candidate order).
- Docs: `docs/preview.md`, README section, CHANGELOG, CLAUDE.md;
  journal `journals/2026-10-19_preview-mode.md`.

## Quality thresholds

Scores are linear between a "good" and a "poor" anchor; the verdict is the
worst metric's status. Calibrated on the 38 Taka movies plus noise-aware
blur/dim simulations — starting points to tune per setup (`QUALITY_SPECS`).

| metric | good → poor |
|---|---|
| Filament signal-to-noise | 12 → 4 |
| Focus (cross-section FWHM) | 290 → 380 nm (in focus 250–290; 1 px defocus 320–370; 2 px 430–600) |
| Illumination flatness (p5/p95 gain) | 0.8 → 0.5 |
| Saturated pixels | 0.01 → 0.5 % |
| Background / filament signal | 6 → 15 |
| Filaments per 100 µm² | too few 1 → 0.3; crowded 8 → 14 |
| Stuck filaments | 15 → 40 % |
| Floppy tracks | 15 → 35 % |
| Bleaching | 1 → 3 %/frame |
| Displacement per frame / length | 0.6 → 1.0 |

## Verification

- **Tests:** `tests/test_preview.py` (helpers, scoring/ranking, end-to-end on a
  synthetic movie with known answers — 323 nm/s, 3 stuck / 6 moving, 60 px
  filaments, which a median background would get wrong — `--latest` frame
  selection, CLI) and `tests/test_profile_width.py` (±3 % on synthetic lines at
  any angle and sub-pixel offset, wide blur, neighbour rejection). Full suite on
  the Mac (`fastrack-arm`, native arm64): **172 passed, 2 skipped**, golden
  masters included.
- **38 Taka movies vs the full `ridge-fast` analysis:** velocity +0.4 % vs
  length-matched MVEL (IQR −2.8 … +2.9 %), mean length +0.8 % (r = 0.93),
  stuck r = 0.77 (systematically higher, by design). Verdicts: 29 GOOD,
  3 MARGINAL, 6 POOR.
- **Linker vectorization:** identical velocity / length / point counts on Taka
  before and after.
- **Determinism across machines:** Mac and cloud give identical numbers for the
  same detector.
- **Timing (densest movie, E11K_300nM_1):** Mac native arm64 1/4/8/12 workers
  8.9 / 4.4 / 3.8 / 4.0 s; Rosetta (x86_64) 8 workers 5.5 s; 2-core cloud 12 s.

## Known limitations / future work

- **Floppiness is inflated by crowding** (r ≈ 0.6 with density): mislinks
  between neighbouring filaments look like wandering tracks. Read it together
  with the density metric. E11K_150nM_2 (34 % floppy, 35 % stuck, bimodal
  velocity) looks like a genuine poor-attachment case.
- **Focus thresholds come from simulated blur**; confirm on a real
  through-focus series. Strong defocus (3 px) is under-read in crowded fields
  (still rated POOR).
- Possible extras: a `--watch` mode with a persistent worker pool for use while
  acquiring; low-order background + direct stuck counting in the full pipeline.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01GHpL1AmJemrX4ED1LFE5fB
