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

## Quickstart

New to Python? Install [Miniforge](https://github.com/conda-forge/miniforge),
open a fresh terminal, then:

```bash
mamba create -n fastrack python=3.11      # one-time: make an environment
conda activate fastrack                   # do this in each new terminal
git clone https://github.com/paulruijgrok/fastrack.git
cd fastrack
pip install -e .                          # install FASTrack + the `fast` command
fast --help                               # check it works
```

Then run the analysis on a folder of movies:

```bash
fast -d /path/to/your/movies
```

That's the standard single-colour workflow. See [Install](#install) for the full
details (and the pip/venv alternative), and
[FASTplus](#fastplus--directional-polarity-aware-analysis-optional) for the
two-colour, polarity-aware mode.

## Package structure

The code uses a `src/` layout and is organized into logical sub-packages, each
module kept to a modest size:

```
src/fastrack/
├── config.py            # layered Settings (hardware / analysis / plotting / runtime / directional)
├── datamodel.py         # FilamentRecord + cross-frame FilamentTable
├── registry.py          # name->factory registry behind every pluggable seam
├── motility.py          # back-compat shim (re-exports the old names)
├── core/
│   ├── frame.py island.py filament.py link.py   # image-processing objects
│   ├── motility.py                              # per-movie analysis driver
│   ├── detection/       # Detector interface + entropy/watershed + ridge detectors
│   ├── tracking/        # Linker interface + greedy (incl. legacy) linker
│   └── input/           # movie input adapters (micro-manager dirs, TIFF stacks)
├── analysis/            # fitting, velocity metrics, geometry (pure numeric)
├── io/                  # images, stores (npy/npz), export (csv), movie (ffmpeg)
├── viz/                 # plotparams + the length-velocity / 2D-path plots
├── pipelines/           # gliding (the `fast` driver), loaded (LIMA), batch
└── cli/                 # console entry points: fast, lima, stack2tifs, fast-batch
```

Three things are pluggable via a registry + a `Settings` field, so new variants
are added without touching call sites: **filament detection**
(`core/detection`, e.g. the optional ridge detector), **frame-to-frame tracking**
(`core/tracking`), and **output** (`io/movie` writers, `io/stores` backends,
`io/export` formats over the `FilamentTable`).  New analysis workflows are added
as modules under `pipelines/`.  See "Extending" below.

The original monolithic `fastrack.motility` import still works via a
compatibility shim that re-exports the names from their new locations.

**FASTplus directional add-on.** An opt-in mode for two-colour, polarity-labelled
movies (signed plus-/minus-end velocity + kinetics) adds the `polarity/`
sub-package and a few modules alongside the core (`core/detection/heads.py`,
`core/tracking/head_tracker.py`, `io/dual_channel.py`,
`analysis/{frame_average,kinetics,perturbation}.py`,
`pipelines/directional.py`) plus the `fastplus` command. These ship with the
package but are only used in directional mode — see
[FASTplus directional analysis](#fastplus--directional-polarity-aware-analysis-optional)
and **[docs/fastplus.md](docs/fastplus.md)**.

## Test data not included

The example movies used below (`examples/`) and the original upstream sources
(`Old code/`) are **not** committed to this repository — the gliding-assay
dataset alone is well over a gigabyte. To reproduce the validation runs, obtain
the original test dataset and place it under `examples/` with the same layout
(e.g. `examples/unloaded_motility/micromanager_tifs/...`). The original code and
data are available from Tural Aksel's upstream project at
https://github.com/turalaksel/FAST.

## Install

FASTrack runs on macOS, Linux, and Windows. **If you are new to Python, use the
conda route below** — it installs the scientific stack (and `ffmpeg`) as
prebuilt binaries, which avoids the compiler/build errors that trip people up
with a bare `pip` install.

### Recommended: Miniforge + conda environment

**1. Install Miniforge** (a small, free conda installer that defaults to the
`conda-forge` package channel and ships the fast `mamba` solver). Download the
installer for your OS from <https://github.com/conda-forge/miniforge> and run it,
accepting the defaults. After installing, **open a new terminal** so the `conda`
command is available. (On Windows, use the "Miniforge Prompt" that was added to
the Start menu.)

You can use `mamba` (faster) or `conda` interchangeably in the commands below —
Miniforge provides both.

**2. Create and activate an environment** for FASTrack (Python 3.11):

```bash
mamba create -n fastrack python=3.11
conda activate fastrack
```

You will run FASTrack from inside this `fastrack` environment; activate it
(`conda activate fastrack`) each time you open a new terminal. Keeping FASTrack
in its own environment means it can't clash with other Python tools.

**3. Get the code and install it.** If you have `git`:

```bash
git clone https://github.com/paulruijgrok/fastrack.git
cd fastrack
pip install -e .
```

(No git? Download the repository ZIP from GitHub, unzip it, then `cd` into the
folder and run `pip install -e .`.) The `-e` makes it an *editable* install, so
`git pull`ing updates takes effect without reinstalling.

This installs the runtime dependencies (numpy, scipy, scikit-image,
opencv-python, matplotlib, imageio, pillow) and the command-line tools
`fast`, `lima`, `stack2tifs`, and `fast-batch`.

**4. (Optional) Install ffmpeg** — only needed to render tracking movies
(`fast -m` / `--overlay-movie`). With conda this is one cross-platform command:

```bash
mamba install -c conda-forge ffmpeg
```

**5. Check it worked:**

```bash
fast --help            # should print the usage message
```

### Advanced alternative: pip + venv (no conda)

If you already manage Python yourself and have a working build toolchain, a plain
virtual environment works too. Note you may need to install `ffmpeg` separately
(see below), and on some systems compiling a dependency wheel can fail — in which
case the conda route above is the easy fix.

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

### Optional extras

FASTrack keeps the default install lean; extra features are opt-in:

```bash
pip install -e '.[ridge-fast]'   # faster ridge-detection filament detector (below)
pip install -e '.[plus]'         # FASTplus directional analysis (see docs/fastplus.md)
```

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
collide with the default entropy detector. The ridge extra pulls in
`ridge-detector` **and `numba`** (the upstream package imports numba but doesn't
declare it). numba (with LLVM) is the one heavy addition, which is why the
default `pip install -e .` leaves it out.

**Faster variant (`ridge-fast`).** A performance-optimized, numerically-identical
drop-in ([`ridge-detector-fast`](https://github.com/paulruijgrok/ridge-detector),
~4× faster via analytical 2×2 eigendecomposition, float32, OpenCV separable
filters, and a numba-compiled contour tracer) is available as a separate extra:

```bash
pip install -e '.[ridge-fast]'      # installs ridge-detector-fast from git
fast -d <dir> --detector ridge-fast
```

It takes the same `--ridge-*` parameters and gets its own output tree
(`...__det_ridge-fast/`) and cache. Because the two produce identical contours,
`ridge-fast` is the one to use for real runs; plain `ridge` is kept as the
reference. Compare any set of detectors (runtime + velocities) in one command,
e.g. confirm the identical-results-and-4× story with
`python tools/compare_detectors.py -d <dataset> --detectors ridge ridge-fast`, or
add `entropy` to the list for a three-way comparison. Pass `--time-detection` to
also report the **detection stage in isolation** (in-process, numba warm-up
excluded), which shows the ~4× directly — the full-pipeline wall-clock dilutes it
with the unchanged linking/plotting/I/O work. Use `--detection-only` to run just
that timer (skipping the full runs).

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

## FASTplus — directional (polarity-aware) analysis (optional)

**FASTplus** extends FASTrack to **two-colour, polarity-labelled** gliding
assays. One channel holds the filaments (as in standard FASTrack); the other
labels one polar end of each filament with a point-like "head" (e.g. gelsolin on
actin barbed ends). FASTplus detects and tracks the heads, decides which
filaments are *unambiguously* labelled (exactly one head, on one end), and scores
motion as **signed velocity** — positive when the motors stroke toward the
(+)/barbed end, negative toward the (−) end. It can also average velocity
per-frame across replicate movies and fit the kinetics of an external
perturbation (e.g. an optogenetic light pulse) with a continuous
exponential-rise/decay model.

Quick start:

```bash
pip install -e '.[plus]'         # adds optomerge (channel registration) + tifffile
fastplus -d <DIR_OF_RGB_TIFFS> --mode head-centric --head-channel red \
    --head-quality 8 --spf 0.1356 --kinetic-model exp_rise_decay -v
```

The directional code ships with the package; the `[plus]` extra only adds the
two-colour **channel-registration** dependency (`optomerge`). On already-aligned
data you can run with `--no-register` and skip the extra entirely.

**Full documentation — concept, installation, CLI/config reference,
perturbation-timing input, outputs, and the parallel benchmark — is in
[docs/fastplus.md](docs/fastplus.md).** Design and verification notes:
[docs/PR_fastplus_directional.md](docs/PR_fastplus_directional.md),
[docs/fastplus_parallel_verification.md](docs/fastplus_parallel_verification.md).

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

### Intermediate cache layout

Detection writes its per-frame filaments (the `filXYs` intermediate consumed by
the tracker) to a cache in each movie folder. Two layouts are available via
`--cache-layout` (or `[runtime] cache_layout` in a config file):

- `per-frame` (default) — one `filXYs<tag>NNN.npy` per frame, as in the original
  pipeline.
- `per-movie` — a single `filXYs<tag>.npz` per movie. Detection still runs in
  parallel across frames; the workers return their results and the parent process
  is the single writer, so memory stays bounded (one frame in flight) and the
  movie folder isn't littered with hundreds of tiny files.

Both layouts store identical content and produce identical results — the
golden-master regression runs under both — so the choice only affects how the
intermediate cache is laid out on disk, e.g.:

```bash
fast -d <dataset> --cache-layout per-movie
```

### Exporting trajectories for downstream analysis

The intermediate `.npy`/`.npz` caches are fast for re-runs but need the program
(and numpy) to open. To get the rich result — the full tracked trajectories of
every filament — in a form any tool can read, use `--export-trajectories`:

```bash
fast -d <dataset> --export-trajectories            # tidy trajectory CSV per movie
fast -d <dataset> --export-trajectories --export-contours   # also the skeleton geometry
```

This writes one **tidy CSV per movie** to the `outputs/` tree,
`<movie>_trajectories.csv`, with one row per filament per frame in **physical
units** (nm, seconds):

| column | meaning |
|---|---|
| `movie` | movie identifier (folder) |
| `path_id` | trajectory id within the movie — **group by this** to get one filament's full track |
| `frame`, `time_s` | frame index and acquisition time (s) |
| `length_nm` | filament length (nm) |
| `x_nm`, `y_nm` | midpoint position (nm; `x` = column, `y` = row) |
| `cm_x_nm`, `cm_y_nm` | centre-of-mass position (nm) |
| `velocity_nm_s` | instantaneous speed (nm/s); blank on each trajectory's last frame |
| `stuck` | 1 if the path is classified stuck, else 0 |
| `n_points` | number of skeleton points (links to the contour file) |

It's plain CSV — open it in Excel, pandas (`df.groupby(["movie", "path_id"])`), R,
or Julia and build whatever analysis isn't in FASTrack. `--export-contours`
additionally writes `<movie>_contours.csv`, the long-format skeleton geometry
(one row per contour point: `movie, path_id, frame, point, x_nm, y_nm`), joinable
to the trajectory table on `(movie, path_id, frame)`. Both are written for every
processed movie, independent of the velocity-statistics plots.

For convenience, every movie's rows are also concatenated into a single
dataset-wide file under `outputs/<...>/combined/`: **`all_trajectories.csv`** (and
`all_contours.csv` with `--export-contours`), so you can load the whole
experiment at once — the `movie` column keeps the movies distinct. (Use `-f` for
a complete combined file; top-level folders skipped as already-analysed aren't
re-exported.)

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

### Input: frame folders or TIFF stacks

`fast` reads movies in two layouts, auto-detected per movie:

- **micro-manager frame folders** — a directory of `img_******NNN_*_000.tif`
  frames plus a `metadata.txt` (the original layout; timing comes from
  `metadata.txt`).
- **multi-page TIFF stacks** — one `.tif` file per movie, read directly (no
  pre-split). Point `-d` at a tree of stacks *or* at a single `.tif`.

```bash
fast -d examples/unloaded_motility/stacks --frame-rate 2.19      # a tree of stacks
fast -d .../alpha_0.04mg_ml/_2.tif --frame-rate 2.19             # one stack file
```

**Stacks need a frame rate.** A TIFF stack carries no acquisition clock, so pass
`--frame-rate <Hz>` (or `[hardware] frame_rate_hz`); it forces uniform timing.
Without it, velocities default to 1 s/frame. `--frame-rate` also works on frame
folders, where it overrides `metadata.txt` (otherwise `metadata.txt` is used, so
existing results are unchanged). Use `--input-format {auto,stack,frames}` to
override the auto-detection. Reading a stack and its pre-split frames produces
identical results under the same timing (covered by `tests/test_stack_pipeline.py`).

The legacy `stack2tifs` pre-split is therefore optional now, but still available:

```bash
stack2tifs -d examples/unloaded_motility/stacks
```

For the loaded-motility (utrophin) series, run `fast` on that tree, then:

```bash
lima -d outputs/<loaded_dataset_output_dir>
```

## Batch processing many datasets (unattended)

`fast-batch` runs the analysis over a whole list of datasets without supervision
(e.g. overnight). Give it a manifest — a `.csv`/`.tsv`/`.xlsx` table with one row
per dataset:

```csv
name,base_dir,config
alpha_overnight,data/2024-05-01/alpha,configs/alpha.toml
beta_default,data/2024-05-01/beta,
```

The base-directory column is required (aliases: `base_dir`, `directory`,
`dataset`, …); `config` (a TOML, see `config.example.toml`) and `name` are
optional. Relative paths resolve against the manifest's location. Then:

```bash
fast-batch datasets.csv                 # process the whole list
fast-batch datasets.csv --preflight-only   # just check everything first
fast-batch datasets.csv --smoke         # pre-flight + detect frame 0 of each
```

What it does:

- **Pre-flight check** first, over the entire list: base dir exists, has movie
  folders with `.tif` (or cached `filXYs`), the config parses, the chosen
  detector's optional package is installed, and `./outputs` is writable. With
  `--smoke` it also detects the first frame of each dataset to catch
  detector/dependency errors before the long run.
- **Never stops on one failure:** each dataset runs in isolation; any error
  (including a hard exit from the pipeline) is logged with a full traceback and
  the run moves on to the next dataset. Use `--stop-on-error` to opt out.
- **Resumable:** a state file (`<logdir>/batch_state.json`) records each
  dataset's outcome and an input/config signature, saved after every dataset. A
  re-run **skips** datasets already completed successfully and unchanged; pass
  `--force` to redo everything or `--retry-failed` to retry only the failures.
  Editing a dataset's data or config changes its signature, so it re-runs
  automatically. Skipping is decided here, at the batch level — so whenever the
  batch *does* run a dataset it fully (re)generates its results (it forces the
  analysis), rather than letting a leftover `outputs/` tree from an earlier run
  silently short-circuit the work.
- **Detailed logs:** a timestamped run log plus a per-dataset log (capturing the
  pipeline's frame-by-frame output) under `--logdir` (default
  `fastrack_batch_logs/`).

`.xlsx` manifests need the optional extra (`pip install '.[batch]'`, which adds
openpyxl); `.csv`/`.tsv` need nothing beyond the base install. Per-dataset
worker count is `-j` (default: all cores).

### On a SLURM cluster

`hpc/` has ready-to-edit `sbatch` wrappers: `fast_single.sbatch` (one dataset)
and `fast_batch_array.sbatch` (a chunked **job array** — each task processes a
shard of the manifest with its own state file, so datasets run concurrently
across nodes and restarts resume cleanly). See [`hpc/README.md`](hpc/README.md).
For why the workload runs on CPU arrays rather than GPUs — and what a GPU port
would actually involve — see [`docs/gpu_feasibility.md`](docs/gpu_feasibility.md).

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
