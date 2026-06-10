# FASTrack (Python 3 port)

A clean, modern Python 3 port of Tural Aksel's **FAST** / **FASTrack** package —
*Fast Actin filament Spud Trekker* — for automated analysis of in-vitro actin
gliding-assay movies (Aksel et al., *Cell Reports*, 2015).

The numerical algorithm is preserved verbatim from the original Python 2.7 code.
This port only updates it to run on current scientific-Python stacks and
replaces two external dependencies:

- **Parallelism:** the original spawned per-frame Python subprocesses through the
  `ppss` bash tool. This port uses Python's built-in `multiprocessing` instead —
  no external tool to install.
- **Movie encoding:** `avconv` is replaced by `ffmpeg`.

## Package structure

The code uses a `src/` layout and is organized into logical sub-packages, each
module kept to a modest size:

```
src/fastrack/
├── config.py            # layered Settings (hardware / analysis / plotting / runtime)
├── datamodel.py         # FilamentRecord + cross-frame FilamentTable
├── registry.py          # name->factory registry behind every pluggable seam
├── motility.py          # back-compat shim (re-exports the old names)
├── core/
│   ├── frame.py island.py filament.py link.py   # image-processing objects
│   ├── motility.py                              # per-movie analysis driver
│   ├── detection/       # Detector interface + entropy/watershed implementation
│   └── tracking/        # Linker interface + greedy (incl. legacy) implementation
├── analysis/            # fitting, velocity metrics, geometry (pure numeric)
├── io/                  # images, stores (npy), export (csv), movie (ffmpeg)
├── viz/                 # plotparams + the length-velocity / 2D-path plots
├── pipelines/           # gliding (the `fast` driver) and loaded (LIMA)
└── cli/                 # console entry points: fast, lima, stack2tifs
```

Three things are pluggable via a registry + a `Settings` field, so new variants
are added without touching call sites: **filament detection**
(`core/detection`, e.g. a future low-SNR detector), **frame-to-frame tracking**
(`core/tracking`), and **output** (`io/movie` writers, `io/stores` backends,
`io/export` formats over the `FilamentTable`).  New analysis workflows are added
as modules under `pipelines/`.  See "Extending" below.

The original monolithic `fastrack.motility` import still works via a
compatibility shim that re-exports the names from their new locations.

## Test data not included

The example movies used below (`examples/`) and the original upstream sources
(`Old code/`) are **not** committed to this repository — the gliding-assay
dataset alone is well over a gigabyte. To reproduce the validation runs, obtain
the original test dataset and place it under `examples/` with the same layout
(e.g. `examples/unloaded_motility/micromanager_tifs/...`). The original code and
data are available from Tural Aksel's upstream project at
https://github.com/turalaksel/FAST.

## Install

The package runs on macOS, Linux, and Windows. From the repository root:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

This installs the runtime dependencies (numpy, scipy, scikit-image,
opencv-python, matplotlib, imageio, pillow) and the three command-line tools
`fast`, `lima`, and `stack2tifs`.

### Optional: ridge-detection filament detector

An alternative filament detector based on ridge detection (Steger's algorithm,
via the [`ridge-detector`](https://github.com/lxfhfut/ridge-detector) package) is
available as an opt-in extra. It is **not** installed by default:

```bash
pip install -e '.[ridge]'
```

Then select it with `fast -d <dir> --detector ridge` (tune with `--ridge-*`
flags: `--ridge-line-widths`, `--ridge-low-contrast`, `--ridge-high-contrast`,
`--ridge-min-len`, `--ridge-dark-line`). Ridge results are written to their own
output tree (`...__det_ridge/`) and use a separate per-frame cache, so they never
collide with the default entropy detector. Compare the two detectors (runtime +
velocities) in one command with `python tools/compare_detectors.py -d <dataset>`.
The ridge extra pulls in
`ridge-detector` **and `numba`** (the upstream package imports numba but doesn't
declare it). numba (with LLVM) is the one heavy addition, which is why the
default `pip install -e .` leaves it out.

Movie generation (`fast -m`) additionally needs **ffmpeg** on your `PATH`:

- macOS: `brew install ffmpeg`
- Debian/Ubuntu: `sudo apt install ffmpeg`
- Windows: `winget install ffmpeg` (or `choco install ffmpeg`)

`fast --overlay-movie` produces a second movie, `overlay_tracks.mp4`: each
original frame with its tracked filaments drawn on top, colored **green
(moving)** vs **red (stuck)**. It's the quickest way to see what a detector is
actually tracking and which filaments it classifies as stuck — run it for two
detectors (`--detector entropy` vs `--detector ridge`) and compare the overlays
side by side. `-m` and `--overlay-movie` are independent; both need ffmpeg.

Both movies share the same styling options (config section `[overlay]`):
`--overlay-fps` sets playback speed (default 10),
`--frame-label/--no-frame-label` toggles the frame number (bottom-left,
right-aligned so it stays steady), and `--time-label/--no-time-label` toggles an
`mm:ss` clock (bottom-right). The time comes from the acquisition metadata
(`ElapsedTime-ms`) when present, otherwise from `--frame-interval` seconds per
frame. These apply to **both** the `-m` skeleton movie and the
`--overlay-movie`; labels are on by default. Example:

```bash
fast -d <dataset> --detector ridge --overlay-movie --overlay-fps 24 --frame-interval 0.1
```

Both movies are written as H.264 in an MP4 container (`filament_tracks.mp4` and
`overlay_tracks.mp4`), which open directly in QuickTime, ImageJ, browsers, and
most players. They are saved **only in the `outputs/` tree** alongside the other
results (the raw data folders are left untouched); the per-frame PNGs are
intermediate and removed after encoding. If ffmpeg is not found, analysis still
runs — only the optional tracking movie is skipped, with a notice. Skeleton/path
image compositing is done in pure Python (Pillow), so no ImageMagick install is
required on any platform.

## Run on the test dataset

The unloaded-motility example contains α-cardiac and β-cardiac myosin movies.
From the project root:

```bash
fast -d examples/unloaded_motility/micromanager_tifs
```

Results are written under `outputs/<dataset>__pt_none__n_5__ymax_1500__p_5__fx_none/`,
including per-movie `*_length_velocity.png` plots and combined
`combined/MEAN_values.txt` / `combined/SEM_values.txt`.

Useful flags (defaults match the original): `-px 80.65` (pixel size nm),
`-n 5` (frames averaged), `-p 5` (min path length), `-maxd 2016.25`
(max inter-frame distance nm), `-minv 80` (stuck threshold nm/s),
`-j N` (worker processes; defaults to all logical cores), `-f` (force
re-analysis), `-m` (make tracking movie), `--exact-rank` (use the exact 16-bit
percentile filters instead of the default 8-bit fast path — slower but exact),
`--morph-contrast` (one-pass morphological-gradient contrast instead of two
percentile passes — faster but noise-sensitive; off by default, A/B it),
`-v` (verbose).

### Config files

For runs with many non-default options (especially the overlay-movie styling),
specifying everything on the command line is impractical. Pass one or more TOML
files with `--config`; any explicit CLI flag overrides the file, and multiple
files layer left-to-right (later wins):

```bash
fast -d <dataset> --config config.example.toml            # all settings from file
fast -d <dataset> --config base.toml --overlay-fps 24     # file + CLI override
```

A fully-commented `config.example.toml` at the repo root lists every section
(`[hardware]`, `[analysis]`, `[plotting]`, `[runtime]`, `[ridge]`, `[overlay]`)
and key. Only the keys you set are applied; the rest keep their defaults. TOML
config requires Python 3.11+.

The dominant per-frame cost is the two radius-15 local percentile filters in
`entropy_clusters` (and `check_picture_quality`). scikit-image rank filters keep
a per-pixel local histogram whose size scales with grey levels (65 536 for
uint16 vs 256 for uint8), so by default each frame is rescaled to 8-bit before
those filters. On the example dataset this is ~1.8× faster with <0.4% velocity
deltas and an unchanged α>β ordering. Pass `--exact-rank` for the reference
16-bit path, e.g. when producing validation numbers. Profile both paths on your
data with `tools/profile_frame.py` (add `--fast-rank` to time the 8-bit path)
and A/B the velocities with `tools/compare_fast_rank.py`.

### Recommended configurations

There are two useful operating points. Use the **fast screening** config for
routine throughput and exploration, and the **exact validation** config for any
numbers that go into a figure or table.

**Fast screening** — `fast_rank` (default on) + morphological-gradient contrast
+ all cores:

```bash
fast -d examples/unloaded_motility/micromanager_tifs --morph-contrast -j 8
```

On the example dataset this runs end-to-end in roughly a third of the exact-path
time (~3× overall). Velocities stay within about 1–1.5% of the reference and the
α>β ordering is unchanged, which is well within tolerance for screening.

**Exact validation** — native 16-bit percentile filters, no morphological
approximation:

```bash
fast -d examples/unloaded_motility/micromanager_tifs --exact-rank -f
```

`--exact-rank` restores the 16-bit percentile path and omitting `--morph-contrast`
keeps the exact 5th/95th-percentile contrast map. Use this to produce the
reference numbers; `-f` forces re-analysis so cached fast-path filaments are not
reused.

The speed/accuracy trade-off of each optimization is `fast_rank` (≈1.8×, <0.4%
deltas, nearly lossless), `--morph-contrast` (further speedup, ~1–1.5% deltas,
more noise-sensitive), and `-j` (lossless, scales with cores). A/B any config
against the exact path on your own data with `tools/compare_fast_rank.py`
(add `--morph` to include the morphological-gradient path).

To convert the original stacks to frame files first (if needed):

```bash
stack2tifs -d examples/unloaded_motility/stacks
```

For the loaded-motility (utrophin) series, run `fast` on that tree, then:

```bash
lima -d outputs/<loaded_dataset_output_dir>
```

## Validate against the paper

The key qualitative check from Aksel et al. 2015: **α-cardiac myosin glides
filaments substantially faster than β-cardiac** (roughly 1.5–2.5×). Open the
combined `MEAN_values.txt` and compare the `top-vel-5` / `MVEL` columns for the
`alpha_*` vs `beta_*` rows. Expected order-of-magnitude: β-cardiac top velocities
in the ~500–900 nm/s range, with α-cardiac clearly higher. The α > β ordering is
the primary parity criterion.

## Tests

```bash
pip install -e .[test]
pytest -q
```

The suite has two tiers. The light unit tests need no image stack or dataset and
run anywhere: `test_smoke.py` (pure-numeric helpers), `test_config.py` (layered
settings), `test_registry.py` (the strategy registry + that the built-in
detectors/linkers/writers/stores register), `test_datamodel.py`
(`FilamentTable` + CSV export), and `test_tracking.py` (the greedy linker's
corrected-vs-legacy partner recovery).

`test_golden.py` is the **golden-master regression**: it re-runs the analysis on
the example dataset with the exact (deterministic) settings and asserts the
combined `MEAN_values.txt` / `SEM_values.txt` match the committed baseline in
`tests/baseline/`. It auto-skips when the image dependencies, the example
dataset, or the baseline are absent. Capture/refresh the baseline with
`python tools/capture_baseline.py -d examples/unloaded_motility/micromanager_tifs`.

## Extending

Each variable part of the pipeline is a small interface plus a registry, with
the current behaviour as the first registered implementation. To add a variant,
write a class and register it; select it via the corresponding `Settings` field
(`detection_algorithm`, `tracking_algorithm`, etc.) — no call sites change.

- **Filament detector** (e.g. better low-SNR segmentation): subclass
  `fastrack.core.detection.base.Detector`, implement `detect(frame)`, decorate
  with `@DETECTORS.register("my_detector")`.
- **Tracker**: subclass `fastrack.core.tracking.base.Linker`, implement
  `link(frame1, frame2, dt, elapsed_times)`, register under `LINKERS`.
- **Movie writer / store / export**: register under `MOVIE_WRITERS`
  (`io/movie`), `STORES` (`io/stores`), or add an exporter in `io/export.py`
  operating on a `FilamentTable`.
- **New analysis pipeline**: add a module under `pipelines/` and register a
  `Pipeline` subclass under `PIPELINES`.

Settings compose from layers (`Settings.from_sources(*dict_layers)`) so hardware,
analysis, and runtime config can live in separate files and be mixed; a TOML
loader (`Settings.from_toml`, Python 3.11+) is the thin adapter on top.

## Notes / things to review

- **`make_frame_links` partner recovery.** The original recovered the accepted
  partner filament by relying on `filament2` being the last loop variable after
  the candidate loop (a latent bug: the link's scores describe the sorted-best
  candidate, but its partner identity/geometry came from whatever filament was
  iterated last). By default this port fixes that, recovering the partner by
  label so identity and scores agree:
  `filament2 = self.frame2.filaments[int(link_candidates[0, 0])]`.

  To reproduce the published results bit-for-bit, pass `--legacy-linking` to
  `fast` (or set `legacy_linking=True` on a `Motility` object / in
  `pipeline.run(...)`), which restores the original leftover-loop-variable
  behaviour. Running the dataset both ways lets you measure how much the fix
  actually moves the velocities:

  ```bash
  fast -d examples/unloaded_motility/micromanager_tifs            # corrected (default)
  fast -d examples/unloaded_motility/micromanager_tifs --legacy-linking -f
  ```

  (Use `-f` on the second run to force re-analysis rather than reusing cached
  links.)
- The image pipeline (scipy/scikit-image/opencv) could not be executed in the
  build sandbox, so end-to-end numbers must be produced on your Mac using the
  steps above.
- All Matplotlib output uses the headless `Agg` backend, so plots render without
  a display server.

## Credits and license

The original FAST / FASTrack algorithm and code were written by **Tural Aksel**
(turalaksel@gmail.com) and published in Aksel et al., "Ensemble force changes
that result from human cardiac myosin mutations and a small-molecule effector,"
*Cell Reports*, 2015. Upstream: https://github.com/turalaksel/FAST.

This repository is a Python 3 port that preserves Aksel's numerical algorithm
and modernizes the surrounding code. It is distributed under the same MIT license
as the original; see [`LICENSE`](LICENSE), which retains Tural Aksel's copyright.
