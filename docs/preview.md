# Preview mode (`fast-preview`)

A few-seconds first look at a gliding-assay movie — meant to be run at the
microscope, while acquiring, to decide whether the imaging conditions are good
enough for the full analysis.

```bash
fast-preview /path/to/movie                 # one movie (Micro-Manager folder or TIFF stack)
fast-preview /path/to/slide                 # every movie under a folder
fast-preview /path/to/movie --latest        # newest frames of a movie still being written
fast-preview /path/to/movie --config config.taka.toml   # pixel size, stuck threshold, ridge settings
```

For each movie it prints a short report and writes (default
`outputs/preview/<name>/`) a `<movie>_preview.json`, a dashboard
`<movie>_preview.png` and a `preview_summary.csv` across movies:

```
E11K_150nM_2  [POOR]  20 frames, bin 2, ridge-fast, 9.5 s
  velocity   mean 733 nm/s, median 661, top-5% 1867   (545 tracked points; timing: metadata)
  stuck      34.6 %  (116 static vs 219 moving filaments/frame)
  length     median 1.26 um (p10 0.78, p90 2.46, mean 1.47)
  quality (worst first):
    POOR      Floppy / poorly attached tracks   34.04 %  -> check motor density / surface attachment
    POOR      Stuck filaments                   34.62 %  -> check surface / motor quality, blocking, ATP
    MARGINAL  Background / filament signal       9.52    -> reduce background (wash out free fluorophore, check TIRF angle)
    GOOD      Filament signal-to-noise          13.39
    ...
```

## What it reports

* **Gliding velocity** — mean of the moving filaments (the same `MVEL` filter as
  the full analysis), median, top-5 %.
* **Length distribution** — median / mean / p10 / p90 of the moving filaments,
  plus the median length of the stuck ones.
* **Stuck fraction** — static filaments / (static + moving per frame).
* **Image-quality metrics**, each scored 0–1 and ranked worst first, with advice:

| metric | what it measures | good → poor |
|---|---|---|
| Filament signal-to-noise | median ridge signal / pixel noise | 12 → 4 |
| Filament width (focus) | median ridge width, full-res centre crop | 300 → 380 nm |
| Illumination flatness | p5/p95 of the fitted gain field | 0.8 → 0.5 |
| Saturated pixels | pixels at the camera's max value | 0.01 → 0.5 % |
| Background / filament signal | median background / filament level | 6 → 15 |
| Filaments per 100 µm² (too few / crowded) | moving filaments per frame | 1 → 0.3 / 8 → 14 |
| Stuck filaments | see above | 15 → 40 % |
| Floppy / poorly attached tracks | tracks whose centre of mass wanders (net/total path < 0.5) | 15 → 35 % |
| Bleaching | linear decline of filament signal | 1 → 3 %/frame |
| Displacement per frame / length | median step / median length | 0.6 → 1.0 (raise the frame rate) |

The overall verdict is the worst metric's status: GOOD (score ≥ 0.7), MARGINAL
(≥ 0.35) or POOR. Anchors live in `QUALITY_SPECS` in
`src/fastrack/pipelines/preview.py`; they were calibrated on the 38-movie Taka
data set plus noise-aware blur/dim simulations (in focus ≈ 265–275 nm, a 3 px
Gaussian defocus ≈ 370 nm; cutting the signal 8× drops SNR from ~12 to ~4).
Treat them as starting points for your own setup.

## How it is fast

Defaults pick the fastest settings; all are flags:

* only the first **20 frames** (`-n`), or the newest 20 with `--latest` (the
  newest file is skipped, it may still be being written);
* **2×2 binning** (`--bin`), with ridge widths / min length scaled to match;
* the **`ridge-fast`** detector (falls back to `ridge`) with the SNR filter on
  (`min_snr 3`);
* detection in a **process pool** (`-j`, default all cores);
* no caches, no movies, a vectorized linker gate (also used by the full
  pipeline — results are bit-identical).

Focus is judged separately on a full-resolution 512 px centre crop of the middle
frame, because binning hides modest defocus.

## Two differences from the full pipeline

**Background.** The full pipeline subtracts a temporal-median background over
the whole movie. Over only 20 frames, a filament that moves less than ~2 of its
lengths covers some pixels in more than half the frames, so a median background
would contain it and subtracting it would cut moving filaments short (on a
synthetic movie: half the length and half the velocity). Preview instead uses
the per-pixel **second-smallest** value of the frames: it holds only what is
present in (nearly) every frame — the background plus stuck filaments.

**Stuck filaments.** Background subtraction removes stuck filaments from the
frames, so tracking under-counts them (this is also true of the full pipeline's
stuck %). Preview counts them directly, by ridge detection on that same
low-order background image. "Static" therefore means *moved less than about one
filament length during the preview window* — with 20 frames at 1 s and 1.5 µm
filaments that is ~75 nm/s, close to the default stuck threshold.

## Validation (Taka data, 38 movies, Oct 2026)

Against the full `fast --detector ridge-fast --preprocess` analysis of all
frames (bin 2, 20 frames, 2 cloud cores):

* velocity vs full `MVEL`: median +6.5 % (r = 0.91); against the full MVEL
  restricted to the same length range (≥ 1.3 µm), +0.4 % (IQR −2.8 to +2.9 %);
* mean length: +0.8 % (r = 0.93);
* stuck %: r = 0.77 with the full analysis, systematically higher (see above);
* time per movie: median 5.7 s, max 14 s on **2 cores** (the densest movie,
  ~530 filaments per frame). Detection is ~60 % of the time and scales with
  cores.

Known limitation: **floppiness is inflated by crowding** (r ≈ 0.6 with
density) — mislinks between neighbouring filaments look like wandering tracks.
Read it with the density metric next to it.

## Python API

```python
from fastrack.pipelines import preview
res = preview.run("/path/to/slide", preview.PreviewSettings(n_frames=20, bin=2),
                  out_dir="outputs/preview/slide")
```

`analyse_movie(descriptor, settings, pool)` analyses one movie and returns the
result dict (velocity, length, stuck, metrics, ranked quality, verdict,
per-stage seconds).
