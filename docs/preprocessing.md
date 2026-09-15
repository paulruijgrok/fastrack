# Preprocessing: illumination / low-SNR correction

Opt-in correction of each movie **before detection**, for acquisitions where
filaments are low-contrast, sit at the bottom of the 16-bit range, and/or the
field has a strong illumination gradient. Available inside both pipelines
(`fast --preprocess`, `fastplus --preprocess`) and as a standalone tool that
writes corrected movies to disk (`tools/preprocess_lowsnr.py`). All three share
one implementation, `fastrack.preprocess`, so their pixels are identical.

Background and the diagnosis that motivated this: `docs/preprocessing_notes.md`.

## Why it is needed

Two independent things wreck detection on such data:

1. **The legacy frame read crushes low counts.** Frames are read as
   `cv2.IMREAD_GRAYSCALE`, which drops 16-bit TIFFs to 8-bit by `>> 8` (then
   `* 257`). Raw values spanning ~300-900 counts collapse to **two grey levels**
   before any detector runs. No threshold can recover that.
2. **Illumination gradient + static junk.** A bright-top / dark-bottom field
   (plus immobile debris) makes a global contrast threshold latch onto the
   brightest region; lowering it over-detects noise in the dark region.

The entropy detector's picture-quality gate (absolute contrast span > 1000 on the
16-bit scale) rejects such movies outright ("Bad picture quality"); the ridge
detector runs but finds only the bright band.

## What it does

Per movie, in float at native precision (the raw pages are read with
`FrameSource.read_raw`, bypassing the 16→8-bit quantisation):

1. **Temporal-median background subtraction** — per-pixel median across the
   movie (`background = "median"`; `"mean"` and `"none"` available). Removes the
   static background, the gradient's DC part and stuck junk; leaves moving
   filaments.
2. **Flat-field** — divide by a Gaussian-smoothed illumination gain field
   (`flat_sigma` px, default 50; 0 disables), normalised to mean 1, so filament
   contrast is equal top-to-bottom.
3. **Mild denoise** — `gaussian_filter(sigma = denoise_sigma)` (default 0.8; 0
   disables).
4. **Movie-wide stretch** — the `clip_hi_pct` percentile (default 99.9) of the
   corrected movie maps to 255; the result is a full-range **8-bit** frame. The
   pipeline then applies its usual `* 257`, so a preprocessed frame is exactly
   what `fast` would see when reading the standalone tool's output from disk.

Because step 4 fills the 8-bit range, the entropy quality gate passes on
corrected data without any change to its thresholds (`tests/test_preprocess.py::
test_entropy_quality_gate_passes_after_preprocessing`).

## How it is wired in (the frame-source seam)

The gliding pipeline reads every movie through a `FrameSource`
(`core/input/`). With `--preprocess` the raw source is wrapped in a
`PreprocessedSource` whose `read()` returns corrected frames; nothing downstream
(detection workers, `Motility`, the overlay movie) knows the difference.

The correction needs a **movie-wide model** (background, gain field, stretch
ceiling), which the strictly per-frame detection flow never holds. The source
therefore **fits the model once** in the parent process (`ensure_model()`), saves
it as a small `preprocess_model_<hash>.npz` in the movie's work directory (the
movie folder for frame-folder input, `outputs/…/_work/…` for stacks), and every
multiprocessing worker rebuilds the source from its descriptor and *loads* the
model instead of refitting.

For movies too long to hold in memory, `model_frames = N` fits the model on N
evenly spaced frames instead of all of them (reads only those N pages). The
background estimate is a little noisier; gain field and stretch ceiling are
essentially unchanged (test: gain within 2 %, ceiling within 10 % at N = 5 of 20).

FASTplus already holds the split filament stack in memory, so it fits and applies
the same model directly on that stack (filament channel only; the head channel is
left raw — LoG head detection is not contrast-limited the same way).

## Output trees and caches

Preprocessing changes the pixels every detector sees, so it is part of every
name that keys results:

- `fast`: the output tree gets a `__pp_<hash>` suffix (next to `__det_ridge`),
  and the per-frame filament cache tag gets `_pp<hash>`. A raw run and a
  corrected run — or two recipes — never share an output tree or a cache.
- `fastplus`: the recipe is folded into the filament detection-cache key (the
  head cache is unaffected, since heads are not preprocessed).

`<hash>` is a short MD5 of the recipe dict (`background`, `model_frames`,
`flat_sigma`, `denoise_sigma`, `clip_hi_pct`), order-independent.

## Usage

```bash
# gliding, ridge detector tuned for corrected low-SNR data
fast -d <dataset> -f --preprocess --detector ridge \
     --ridge-line-widths 3 5 --ridge-low-contrast 18 --ridge-high-contrast 50 --ridge-min-len 12

# knobs (all optional)
fast -d <dataset> --preprocess --pp-flat-sigma 40 --pp-denoise-sigma 1.0 \
     --pp-clip-hi-pct 99.8 --pp-background median --pp-model-frames 0

# long movies: fit on 30 evenly spaced frames instead of all
fast -d <dataset> --preprocess --pp-model-frames 30

# FASTplus
fastplus -d <dataset> --preprocess --detector ridge ...
```

Or in a config file:

```toml
[preprocess]
enabled = true
background = "median"     # median | mean | none
model_frames = 0          # 0 = all frames; N = N evenly spaced frames
flat_sigma = 50.0         # px; 0 disables flat-fielding
denoise_sigma = 0.8       # px; 0 disables
clip_hi_pct = 99.9
```

The standalone tool is still the way to **materialise** corrected movies — for
inspection, for the auto-crop (`--crop-frac`), or for tools that read TIFFs
directly:

```bash
python tools/preprocess_lowsnr.py -i RAW_DIR -o CORRECTED_DIR [--crop-frac 0.7]
```

## Tuning notes

Starting values (`flat_sigma 50`, `denoise 0.8`, ridge `line_widths [3,5]`,
`low/high_contrast 18/50`, `min_len 12`) were tuned on one dataset
(`examples/taka/…/slide_01`) and need per-dataset validation. `flat_sigma` should
be much larger than a filament width and smaller than the gradient's scale;
`clip_hi_pct` trades a few saturated pixels against contrast for the rest.

## Limitations / future work

- `model_frames` is a sampled fit, not a streaming one: a true rolling background
  (for movies with slow drift) is not implemented.
- The head channel in FASTplus is not preprocessed; add a switch if LoG head
  detection turns out to need it on low-SNR data.
- The batch runner's `--smoke` preflight detects frame 0 of the *raw* movie; it
  does not apply the recipe.
