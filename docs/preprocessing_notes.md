# Preprocessing for low-SNR / uneven-illumination movies (placeholder)

**Status: findings + proposed plans only. Not implemented.** Held pending a
comprehensive, cross-pipeline preprocessing strategy (gliding `fast`, directional
`fastplus`, loaded `lima`). This page records what was diagnosed on one dataset
so the eventual design starts from evidence rather than from scratch.

## Motivating dataset

`examples/taka/04302026 Motility Experiment/slide_01` — 8 movies
(E11K / WT × 300 / 500 nM, two replicates each). Each movie is a MicroManager
directory (`<movie>/Pos0/img_*_561 nm_000.tif` + `metadata.txt`), 60 frames,
1002 × 1004, uint16, single 561 nm channel.

Running `fast -d … -f --detector ridge` produced almost nothing usable: only 4 of
8 movies wrote a `paths_2D.png`, and `combined/MEAN_values.txt` / `SEM_values.txt`
contained only a header row. The entropy detector rejected the movies outright
("Bad picture quality").

## Findings

### 1. Output is empty because detection is empty, not because of a writer bug

Per-movie `*_length_velocity.{txt,png}` and the combined MEAN/SEM values are only
written when a movie yields **mobile, trackable trajectories** passing the filters
(`min_path_length`, stuck-velocity, etc.). Detection was too sparse and too
spatially biased for anything to track across frames, so there were no velocities
to write. The empty velocity files are a symptom, not a separate problem.

### 2. The images are low-contrast, with a strong illumination gradient and static junk

Filaments are present across the **entire** field of view (see
`assets/taka_lowsnr_preproc.png`, top-left), but:

- **Low contrast.** Background ≈ 318 counts; filament peaks only ≈ 120–150 counts
  above background (99.9th percentile ≈ 433–475 on a 16-bit scale). This is why
  the entropy detector's acquisition-quality gate rejects them — it requires an
  absolute contrast span > 1000 (16-bit) *and* relative contrast > 0.7
  (`core/frame.py::check_picture_quality`). The ridge detector has no such gate,
  which is why it ran at all.
- **Illumination gradient (bright top → dark bottom).** Max-projection brightness
  by vertical band (top→bottom), E11K_300nM_2: 426 → 340. Per-frame vertical-band
  std (a proxy for filament SNR) falls 21.9 → 7.5 top-to-bottom. So a single
  global contrast threshold latches onto the brighter top third and misses the
  rest of the field.
- **Static junk concentrated at the top.** The temporal median
  (`assets/taka_lowsnr_preproc.png`, top-right) shows heavy immobile
  aggregate/debris signal in the top region, which further biases detection and
  confuses tracking.

### 3. Detection is the bottleneck, and it is contrast-threshold + gradient limited

Ridge detection on a representative frame (E11K_300nM_2, mid-stack;
`dark_line=False` since filaments are bright-on-dark):

| Input / params | contours | top half | bottom half | bottom/top |
|---|---|---|---|---|
| raw, `line_widths=[3]`, `low/high=50/150` (defaults) | 255 | 227 | 28 | 0.12 |
| median-subtracted, same params | 254 | 220 | 34 | 0.15 |
| flat-field + denoise, `line_widths=[3,5]`, `low/high=18/50` | 456 | 278 | 178 | 0.64 |

The raw/default row reproduces the reported symptom (detections hug the top; see
`assets/taka_lowsnr_detect.png`, left). Simply lowering thresholds without
correcting the gradient over-detects noise in the dark regions
(`assets/taka_lowsnr_detect.png`, right). Correcting the illumination first lets a
single moderate threshold work field-wide (`assets/taka_lowsnr_flatfield.png`).

## Preprocessing recipe that worked (reference implementation to generalize)

Per movie, per frame:

1. **Temporal-median background subtraction.** Compute the per-pixel median across
   all frames (static background + illumination + stuck junk) and subtract it —
   removes immobile structure, leaves moving filaments.
2. **Illumination flat-fielding.** Estimate a smooth gain field
   (`gaussian_filter(temporal_mean, sigma≈50)`, normalized to mean 1) and divide,
   equalizing filament contrast top-to-bottom.
3. **Mild denoise.** `gaussian_filter(sigma≈0.8)` to lift SNR before ridge.

Then ridge with: `dark_line=False`, `line_widths=[3,5]`, `low_contrast≈18`,
`high_contrast≈50`, `min_len≈12`. These are starting points, not final — they
were tuned on one frame of one movie and need per-dataset validation.

## Proposed approaches (decision deferred)

**A. Bake preprocessing into the FASTrack pipeline (opt-in).**
Add a preprocessing stage at the frame-source / detection seam, selected via
`Settings` + CLI (e.g. `--preprocess flatfield`), reusable across pipelines.
Pros: first-class, tested, one command, cache-aware. Cons: more code; must thread
through `Settings`/CLI and fold preprocessing params into the detection-cache key;
touches the pipeline.

**B. Standalone preprocessing script.**
A separate tool that reads raw MicroManager movies, writes corrected TIFFs to a
new folder, then `fast --detector ridge` runs on those. Pros: fast to deliver, no
pipeline changes, raw data untouched. Cons: two-step workflow; corrected movies
duplicated on disk; parameters live outside the main config.

## Cross-pipeline considerations (for the comprehensive strategy)

- Preprocessing is generic and should apply to `fast` (gliding), `fastplus`
  (directional — after channel split), and `lima` (loaded). Placing it at the
  input/frame-source seam would let all pipelines share one implementation.
- It interacts with the **entropy quality gate**: on flat-fielded data the gate's
  absolute-contrast threshold (>1000, 16-bit) no longer makes sense and would need
  rescaling or bypassing.
- Preprocessing parameters must become part of the **detection-cache key** so that
  changing them invalidates cached detections (mirrors how detector params and the
  new registration params already key the cache).
- Temporal-median subtraction needs the **whole stack** in hand, which differs
  from the current strictly per-frame detection flow — worth accounting for in the
  frame-source abstraction (and in the FASTplus movie-parallel / cache design).
- Decide whether preprocessing is a distinct stage feeding the existing detectors,
  or whether some of it (e.g. flat-fielding) belongs inside a detector variant.

## Reference material

- Diagnostic figures: `assets/taka_lowsnr_preproc.png`,
  `assets/taka_lowsnr_detect.png`, `assets/taka_lowsnr_flatfield.png`.
- Relevant code today: `core/frame.py::check_picture_quality` (entropy gate),
  `core/detection/ridge.py` (ridge adapter, no gate), `pipelines/gliding.py`
  (quality check + output writing), `core/input/` (frame-source seam where
  preprocessing would most naturally live).
