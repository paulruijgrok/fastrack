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
WT_250nM_1  [POOR]  20 frames, bin 2, ridge-fast, 5.7 s
  velocity   mean 1307 nm/s, median 1346, top-5% 2166   (108 tracked points; timing: metadata)
  stuck      23.9 %  (11 static vs 35 moving filaments/frame)
  length     median 2.97 um (p10 1.03, p90 5.15, mean 3.00)
  attachment floppy 48.1 % (13 of 27 tracks), pinned 3.0 % (1 of 33)
  quality (worst first):
    POOR      Floppy tracks (leave their own track)       48.15 %  -> check motor density / surface attachment
    POOR      Filaments per 100 um^2 (too few)             0.54    -> add more filaments / move to a denser area
    MARGINAL  Pinned filaments (one end anchored)          3.03 %  -> check surface passivation / dead heads
    MARGINAL  Stuck filaments                             23.89 %  -> check surface / motor quality, blocking, ATP
    GOOD      Bleaching                                    1.15 %/frame
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
| Filament width (focus) | median cross-section FWHM measured from the image, full-res centre crop | 290 → 380 nm |
| Illumination flatness | p5/p95 of the fitted gain field | 0.8 → 0.5 |
| Saturated pixels | pixels at the camera's max value | 0.01 → 0.5 % |
| Background / filament signal | median background / filament level | 6 → 15 |
| Filaments per 100 µm² (too few / crowded) | moving filaments per frame | 1 → 0.3 / 8 → 14 |
| Stuck filaments | see above | 15 → 40 % |
| Floppy tracks | moving tracks that leave their own track sideways (see below) | 10 → 30 % |
| Pinned filaments | tracks with one end anchored and the other sweeping (see below) | 1 → 5 % |
| Bleaching | linear decline of filament signal | 1 → 3 %/frame |
| Displacement per frame / length | median step / median length | 0.6 → 1.0 (raise the frame rate) |

The overall verdict is the worst metric's status: GOOD (score ≥ 0.7), MARGINAL
(≥ 0.35) or POOR. Anchors live in `QUALITY_SPECS` in
`src/fastrack/pipelines/preview.py`; they were calibrated on the 38-movie Taka
data set plus noise-aware blur/dim simulations (in focus 250–290 nm, a 1 px
(80 nm) Gaussian defocus 320–370 nm, 2 px 430–600 nm; cutting the signal 8×
drops SNR from ~12 to ~4).
Treat them as starting points for your own setup.

## How it is fast

Defaults pick the fastest settings; all are flags:

* only the first **20 frames** (`-n`), or the newest 20 with `--latest` (the
  newest file is skipped, it may still be being written);
* **2×2 binning** (`--bin`), with ridge widths / min length scaled to match;
* the **`ridge-fast`** detector (falls back to `ridge`) with the SNR filter on
  (`min_snr 3`);
* detection in a **process pool** (`-j`; default: the performance cores on
  Apple Silicon, all cores elsewhere). Workers import only the detector
  (`core/detection/worker.py`), not the whole pipeline: on macOS workers are
  started fresh each run, and the full import cost ~3 s per run;
* no caches, no movies, a vectorized linker gate (also used by the full
  pipeline — results are bit-identical).

## Focus: width measured from the image

Focus is judged on a full-resolution 512 px centre crop of the middle frame,
because binning hides modest defocus. The detector only *locates* the
filaments; the width is measured from the raw pixels
(`analysis/profile_width.py`): for each of the 40 longest filaments,
intensity profiles are sampled perpendicular to the centerline, re-centred to
sub-pixel precision, combined into a median profile and fitted with a Gaussian
on a sloped background; the metric is the median FWHM. Profiles that cross
another filament are dropped, and the fit window widens to 3.5σ for blurred
filaments (a narrow window under-reads blur).

Why not the detector's own width: it depends on the detector implementation
(`ridge` reports ~60 nm wider than `ridge-fast` on the same data). The profile
FWHM agrees between the two within ~10–20 nm, is exact (±1 %) on synthetic
Gaussian lines at any angle and sub-pixel offset, and is unbiased by a
neighbour 7 px away. It runs while the workers detect the other frames.

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

On an Apple Silicon Mac (8 performance + 4 efficiency cores), the densest
movie (E11K_300nM_1) takes **3.8 s** with 8 workers in a native (arm64) conda
environment, vs 5.5 s in an Intel/Rosetta (`x86_64`) one; 12 workers is
slightly slower than 8. Check with `python -c "import platform;
print(platform.machine())"` — use a native environment (Miniforge, arm64).

## Attachment: floppy and pinned filaments

Both are computed from the tracked contours (`analysis/track_shape.py`); the
dashboard draws the flagged tracks on the image (orange / magenta) and shows a
strip of examples with the contours coloured by time.

**Floppy — leaving the track.** A filament gliding on a well-attached motor
lawn follows its own track: every part of the new contour lies on the curve of
the previous one, except the leading end that has moved past it. For each step
the distance of each new contour point to the *curve* of the old contour is
measured (ignoring points beyond the old ends). Gliding filaments sit at ~0
(median 14 nm on Taka — pixel rounding); a filament that bends, swings or flails
moves part of itself sideways.

* A step counts only if the new contour still **touches** the old one
  (≤ 160 nm somewhere): a flailing filament keeps part of itself on its track,
  a link to a *different* filament (a mislink in a crowded field) does not.
* A step is **lateral** when ≥ 10 % of the contour (90th percentile of the
  point distances) is > 320 nm off the old track (2 binned pixels; pixel
  rounding alone gives up to ~1 px).
* A moving track (≥ 5 links, ≥ 3 valid steps) is **floppy** when ≥ 30 % of its
  valid steps are lateral. The metric is the % of such tracks.

Why not FAST's own orthogonal distance (`area_score` in `Filament.sim_score`):
it compares contour point *i* in one frame with point *i* in the next, so on
curved filaments that slide several pixels per frame it measures chords across
the bend — it grows with speed (≈ 0.7 → 2.5 binned px on Taka) even for
perfect gliding.

**Pinned — one end anchored, the other sweeping.** Both tips are followed along
each track (≥ 5 links, stuck tracks included), tip identity kept consistent via
each link's `fil_direction`. A track is pinned when one tip stays put (radius of
gyration ≤ 120 nm) while the other sweeps (radius of gyration ≥ max(320 nm,
20 % of the filament length)).

On the 38 Taka movies: floppy median 7.6 % (p90 18 %, max 48 % for
WT_250nM_1); pinned 0 % in most movies, up to 3 % in the sparse, high-stuck
ones. Floppy no longer depends on crowding (r = −0.23 with density; the earlier
net/total-path straightness had r = 0.59 because mislinks looked like wandering
tracks) and correlates with the pinned fraction (r ≈ 0.5). Limitations: at
2×2 binning a pixel is ~160 nm, so subtle pinning (< 2 px sweep) is missed, and
in the most crowded or dimmest fields a few flagged tracks are still tracking
errors or jittery contours. Both cost ~0.2 s per movie.

## Python API

```python
from fastrack.pipelines import preview
res = preview.run("/path/to/slide", preview.PreviewSettings(n_frames=20, bin=2),
                  out_dir="outputs/preview/slide")
```

`analyse_movie(descriptor, settings, pool)` analyses one movie and returns the
result dict (velocity, length, stuck, metrics, ranked quality, verdict,
per-stage seconds).
