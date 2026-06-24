# Read movies directly from multi-page TIFF stacks (no stack2tifs pre-split)

## Summary

Adds a pluggable **movie-input seam** so the analysis can read frames straight
from a multi-page TIFF stack, with the original micro-manager frame folder as one
backend — and architected so other formats (OME-TIFF, ND2, …) are just new
sources. Proven equivalent to the per-frame path: same pixels in, same numbers
out. The default frame-folder behavior is unchanged (golden-master verified).

## What changed

- **`core/input/` — the `FrameSource` seam.** `count`/`read(n)`/`frame_numbers`/
  `elapsed_times`/`identity` + a picklable `descriptor()` so multiprocessing
  workers rebuild a source in a child process. Backends: `MicroManagerDirSource`
  (frame folder, byte-identical to the legacy read) and `TiffStackSource`
  (multi-page TIFF, lazy `tifffile` page reads, `cv2` fallback). `open_movie`
  probes a path (a `.tif` file → stack; a frame folder → mm).
  `to_pipeline_image` reproduces the legacy quantization exactly (16→8 via `>>8`
  like `cv2.IMREAD_GRAYSCALE`, then `×257` like `img_as_uint`), so a stack reads
  identically to its split frames.
- **`Motility` reads via the source** (frames + timing); default builds an
  `MicroManagerDirSource`, so behavior is unchanged.
- **`pipelines/discovery.py`** unifies discovery: a movie is a frame folder *or*
  a stack file, both mapping to the same `(top_root, exp)` identity (matching
  output names and grouping).
- **`gliding.run` drives movies through the seam.** Per movie it builds a source
  descriptor + a *work dir* (cache/links/intermediate): mm = the frame folder
  (legacy); stack = `outputs/_work/<name>/` so the raw `.tif` stays read-only.
  Frame count/size, the frame-0 quality check, the two workers, and the overlay
  re-read all go through the source.
- **Timing is now an explicit, format-agnostic input.** Stacks carry no clock, so
  `--frame-rate` (or `[hardware] frame_rate_hz`) forces uniform spacing
  (`dt = 1/rate`), overriding `metadata.txt`/embedded times. Default (`None`)
  keeps mm reading `metadata.txt` — golden-master safe.
- **CLI surface:** `--input-format {auto,stack,frames}` overrides auto-detection;
  `-d` accepts a single `.tif` (not just a tree); `--frame-rate` for stack timing.
- **`fast-batch` handles stacks:** pre-flight uses the unified discovery (stack
  trees and single-`.tif` bases), `--smoke` detects frame 0 via the source, and
  the resume signature fingerprints a single file.

## Validation

- `test_input` / `test_discovery` — source reads, prober, descriptor round-trip,
  and `stack.read(n) == frames.read(n)` on the example data.
- `test_golden` — frame-folder path still reproduces the baseline bit-for-bit
  (both cache layouts).
- `test_stack_pipeline` — runs the **whole pipeline** on `examples/.../stacks`
  and `.../micromanager_tifs` with the same `--frame-rate` and asserts the
  combined statistics are **identical** (pixel/pipeline equivalence, timing held
  constant). All pass (`pytest -q` over these = green).

## Usage

```bash
fast -d <tree_of_stacks> --frame-rate 2.19      # stacks need a frame rate
fast -d <tree_of_frame_folders>                 # unchanged; uses metadata.txt
```

## Notes / follow-up

- `tifffile` is already a transitive dependency (scikit-image), so no new hard
  dependency. ND2 / other formats would be future optional sources behind the
  same seam.
- The legacy `stack2tifs` pre-split is now optional but kept for back-compat.
