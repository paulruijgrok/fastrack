# FASTplus — directional (polarity-aware) gliding-motility analysis

FASTplus is an opt-in mode of FASTrack for **two-colour, polarity-labelled**
gliding assays. It scores filament motion as **signed velocity** (plus- vs
minus-end directed) and fits the kinetics of external perturbations such as
optogenetic light pulses.

This page is the complete guide. For a one-paragraph summary see the
[main README](../README.md#fastplus--directional-polarity-aware-analysis-optional).

---

## Contents

- [Concept](#concept)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Choosing a mode](#choosing-a-mode)
- [The velocity sign convention](#the-velocity-sign-convention)
- [Perturbation (LED) timing](#perturbation-led-timing)
- [Outputs](#outputs)
- [Configuration reference](#configuration-reference)
- [Performance / parallelism](#performance--parallelism)
- [Troubleshooting](#troubleshooting)

---

## Concept

A filament is a polar polymer: its two ends are chemically distinct (the
(+)/"barbed" end and the (−)/"pointed" end). In a polarity experiment a second
fluorescent label marks **one** of those ends as a small, approximately
point-like "head" (e.g. gelsolin on actin barbed ends). The movie therefore has
two channels:

- a **filament** channel (identical in nature to standard FASTrack), and
- a **head** channel — bright point sources marking one polar end.

FASTplus offers two analysis strategies:

- **Head-centric** (`--mode head-centric`, the validated path) — detect and
  track the heads across frames, then in each frame associate every head with a
  detected (but not frame-to-frame tracked) filament. This suits the high
  filament density and frequent crossings of typical polarity data, where
  tracking whole filaments frame-to-frame is unreliable.
- **Filament-centric** (`--mode filament-centric`) — track the filaments, then
  attach a sign from the head sitting on one tip. Closer to standard FASTrack but
  currently uses a lightweight tracker (see Notes in the
  [PR description](PR_fastplus_directional.md)).

In both modes, a filament is included in the directional statistics only if it is
**unambiguously labelled**: exactly one head, on exactly one end. Filaments with
heads on both ends, in the middle, or with no head are excluded and reported in
the per-class tallies.

The detection/tracking of the heads mirrors the ImageJ/TrackMate pipeline used in
the original study (Ruijgrok et al., *Nat. Chem. Biol.* 2021), reimplemented in
pure Python: a single-scale Laplacian-of-Gaussian spot detector (≈ TrackMate LoG)
and a constant-velocity Kalman + LAP tracker with gap closing (≈ TrackMate
LinearMotionLAP).

## Installation

FASTplus is part of the FASTrack package — first install FASTrack itself
following the [README install instructions](../README.md#install) (Miniforge +
a conda environment is recommended). Then add the FASTplus extra:

```bash
conda activate fastrack          # the environment you created for FASTrack
pip install -e '.[plus]'
```

The `[plus]` extra adds:

- **`optomerge`** — registers (aligns) the two fluorescence channels;
- **`tifffile`** — robust multi-page TIFF reading for the RGB movies.

You do **not** need the extra if your channels are already aligned: the
directional code itself ships with the base package, so

```bash
pip install -e .                 # base install
fastplus -d <DIR> ... --no-register
```

works on pre-registered data. `--no-register` skips the `optomerge` step
entirely. (If `optomerge` is missing and registration is requested, FASTplus
warns and proceeds on the raw channels rather than crashing.)

Verify:

```bash
fastplus --help
```

## Quick start

```bash
fastplus -d <DIR_OF_RGB_TIFFS> \
    --mode head-centric --head-channel red \
    --head-quality 8 --spf 0.1356 \
    --kinetic-model exp_rise_decay -v
```

- `-d` points at a directory; FASTplus finds every file whose name ends in
  `RGB.tif` (case-insensitive), recursively.
- `--head-channel red` selects the channel holding the heads.
- `--spf 0.1356` is seconds-per-frame (here 135.6 ms); set it to get velocities
  in nm/s (otherwise they come out per-frame).
- `-v` prints progress, the resolved perturbation schedule, and worker count.

Config-file equivalent (see [`config.plus.example.toml`](../config.plus.example.toml)):

```bash
fastplus -d <DIR> --config config.plus.example.toml
```

CLI flags override config-file values, which override built-in defaults.

## Choosing a mode

| | head-centric | filament-centric |
|---|---|---|
| tracks | the head spots | the filaments |
| best for | dense fields, frequent crossings (typical polarity data) | sparser fields |
| status | validated | lightweight tracker; see PR notes |

## The velocity sign convention

Velocity is **positive when the (invisible) motors are stroking toward the
(+)/barbed end** of the filament, and negative toward the (−) end. Because
surface motors propel a filament (−)-end-first, "stroking toward (+)" means the
(+) end trails the motion. In terms of where the fluorescent label sits relative
to the direction of travel:

| label marks | head lagging (at the back) | head leading (at the front) |
|---|---|---|
| **(+) end** (default, e.g. gelsolin/actin) | **+** | **−** |
| **(−) end** | **−** | **+** |

Set which end the label marks with `--head-marks plus` (default) or
`--head-marks minus`. If your average comes out with the opposite sign to what
you expect, this flag is almost always what you want to change.

## Perturbation (LED) timing

To fit kinetics, FASTplus needs to know when the perturbation switched on/off.
The schedule is resolved **per movie**, with `--switch-source auto` (default)
trying these in order:

1. **Per-movie sidecar** `<movie>.perturb.toml` (also `.json` / `.yaml`) — the
   clearest option. See [`config.perturb.example.toml`](../config.perturb.example.toml):

   ```toml
   [perturbation]
   switch_frames = [98, 298]     # or switch_times_s = [13.3, 40.4]
   states        = [1, 0]        # state AFTER each switch; >0 = ON
   ```

   The trailing `" RGB"` in a movie name is ignored when matching, so
   `movie 01 RGB.tif` matches `movie 01.perturb.toml`.

2. **Legacy LED files** (the original v0.1 format) — a per-frame times file
   `<base>.csv` (one acquisition time in ms per row) plus `<base> led.csv`
   (row 0 = switch times, row 1 = LED voltages). FASTplus reproduces the original
   switch-frame logic, including trimming to whole on/off cycles.

3. **Config / CLI** — the same schedule for every movie, via
   `--switch-frames 98 298` (or `--perturb 13.3 40.4` in seconds) and optional
   `--perturb-states 1 0`.

Any number of on/off cycles is supported; each rise (light on) and decay (light
off) becomes a fitted segment. For pooled runs the first movie's schedule is used
for the combined fit, and a warning is issued if a later movie's switch frames
differ.

Choose the kinetic model with `--kinetic-model {none,exp_rise,exp_decay,exp_rise_decay}`.
The fit is **piecewise-continuous**: a single dark baseline `A0` is fixed from the
pre-illumination data, and each segment starts exactly where the previous one
ended.

## Outputs

Written under `<-d dir>/fastplus_out/` (or `--output DIR`):

Per movie (in a subfolder named after the movie):

- `directional_paths.csv` — one row per tracked object per frame-step, with the
  signed velocity (nm/s).
- `qc_overlay.png` — montage of frames with detected filament contours (blue) and
  heads coloured by polarity class (green = plus-end / red = both-ends /
  orange = middle / grey = none). The QC view to confirm detection + association.
- `qc_overlay.mp4` — the same as a movie (needs `ffmpeg`; add `--overlay`).

Per dataset (at the top of `fastplus_out/`):

- `frame_average.csv` — per-frame mean signed velocity, SEM, and N across movies.
- `frame_average.png` — mean velocity vs time with central-percentile bands,
  switch lines, shaded light-ON intervals, and the fitted kinetic curve.
- `kinetics.txt` — the fit (dark baseline `A0`, and per cycle: τ, start and
  target levels) plus the polarity-class tally.

Pass `--overlay` to also write the QC overlay PNG/MP4 per movie.

## Configuration reference

All keys live in the `[directional]` section (plus a few shared `[hardware]` /
`[analysis]` keys); each has a CLI equivalent.

| Setting (`[directional]`) | CLI flag | Default | Meaning |
|---|---|---|---|
| `mode` | `--mode` | `head-centric` | analysis strategy |
| `head_channel` | `--head-channel` | `red` | channel with the heads |
| `filament_channel` | `--filament-channel` | `green` | channel with the filaments |
| `head_sigma` | `--head-sigma` | 1.5 | Gaussian pre-blur (px) |
| `head_radius` | `--head-radius` | 5.0 | estimated head radius (px) |
| `head_quality` | `--head-quality` | 5.0 | LoG quality threshold (tune per dataset) |
| `head_tracking_algorithm` | `--head-tracker` | `kalman-lap` | head tracker |
| `initial_search_radius` | `--initial-search` | 20 | first-step linking gate (px) |
| `kalman_search_radius` | `--kalman-search` | 15 | gate once a velocity is known (px) |
| `max_frame_gap` | `--max-gap` | 4 | gap-closing tolerance (frames) |
| `end_fraction` | `--end-fraction` | 0.15 | fraction of length counted as an "end" |
| `max_end_distance_nm` | `--max-end-distance` | 500 | head→tip association distance (nm) |
| `head_marks_end` | `--head-marks` | `plus` | which end the label marks (sign convention) |
| `register_channels` | `--register/--no-register` | on | optomerge channel registration |
| `channel_map` | `--channel-map` | — | e.g. `"red=heads,green=filaments"` |
| `perturbation_source` | `--switch-source` | `auto` | sidecar / led-csv / config / none |
| `switch_frames` | `--switch-frames` | — | explicit switch frames (config source) |
| `perturbation_times_s` | `--perturb` | — | explicit switch times in s (config source) |
| `perturbation_states` | `--perturb-states` | — | state after each switch; >0 = ON |
| `kinetic_model` | `--kinetic-model` | `none` | exp rise/decay model to fit |
| `percentiles` | `--percentiles` | `14 86 2 98` | central-percentile bands (inner→outer) |
| `detection_cache_layout` | `--cache-layout` | `per-movie` | detection cache: per-movie `.npz` or per-frame `.npy` |
| `export_detections` | `--export-detections` | off | write per-movie minimal detection CSV + heads CSV |
| `export_detection_contours` | `--export-contours` | off | also write the full long-format contour CSV |
| `parallel_movies` | `--movie-workers` | 1 | analyse N movies concurrently (alternative to `-j`) |
| (runtime) `force_analysis` | `-f` | off | re-detect, ignoring/refreshing the cache |
| (runtime) `recalculate` | `-r` | off | reuse cached detections, recompute scoring |

Shared knobs used by FASTplus: `[hardware] pixel_size_nm`, `[hardware]
frame_rate_hz` (or the CLI sugar `--spf` seconds-per-frame), `[analysis]
detection_algorithm` (the filament detector — `entropy`, `ridge`, `ridge-fast`),
and `[analysis] stuck_velocity_nm_s`.

Quick-test flags: `--limit N` (first N movies), `--max-frames N`, `--frame-step K`.

## Performance / parallelism

There are two, mutually-exclusive ways to use multiple cores:

- **Within-movie (default)** — per-frame filament detection (the dominant cost)
  runs across worker processes; control with `-j` (`-j 1` serial, omit for all
  cores). Best for a few large movies. Output is independent of worker count
  (verified byte-for-byte); measured ~3.2× on 8 workers for a 2-movie /
  800-frame set. See
  [docs/fastplus_parallel_verification.md](fastplus_parallel_verification.md)
  and `tools/benchmark_fastplus_parallel.py`.
- **Across-movie** — `--movie-workers N` analyses N movies concurrently, one
  process each. Best for **many small movies**. Because a worker process can't
  spawn its own pool, per-frame detection runs serially inside each worker, so
  `--movie-workers` and `-j` are alternatives, not combined. **Peak memory scales
  with N** (N movies in flight) — the main thing to weigh. A failed movie is
  logged and skipped rather than aborting the run.

### Detection cache (fast reruns)

Detection results are cached per movie, so reruns — parameter sweeps on
association, scoring, or fitting — skip the expensive detection (and even the
movie load) entirely. The cache reuses the FASTrack `STORES` machinery:

- `--cache-layout per-movie` (default) — one `filXYs_fp_*.npz` per movie;
  `--cache-layout per-frame` — one `.npy` per frame (legacy layout).
- The cache is written into each movie's output folder and is keyed by movie +
  **detection parameters** (detector, channel, registration, frame subsetting):
  change any of them and the cache is recomputed automatically.
- `-f` / `--force` re-detects and refreshes the cache; `-r` reuses cached
  detections and recomputes scoring (this is also the default whenever a valid
  cache exists — `-r` is kept for parity with `fast`).

Optional per-movie CSV exports of the cached detections (modern-standard
outputs): `--export-detections` writes `filaments_minimal.csv` (one row per
filament) + `heads.csv`; `--export-contours` adds the full long-format
`filaments_contours.csv` (one row per contour point).

## Troubleshooting

- **"0 RGB movies found"** — `-d` must point at (a parent of) the `*RGB.tif`
  files; the path is resolved from your current directory. Pass an absolute path
  if unsure.
- **Velocities look ~10× off** — set `--spf` (seconds per frame). Filenames like
  `…1356mspf…` mean 135.6 ms/frame → `--spf 0.1356`.
- **Average has the wrong sign** — flip `--head-marks` (plus ↔ minus).
- **Too many / too few heads** — tune `--head-quality` (higher = fewer). The
  threshold is on the raw LoG response, so it is dataset-dependent; use
  `--overlay` and inspect `qc_overlay.png`.
- **`optomerge` install failed** — it is only needed for channel registration;
  run with `--no-register` on pre-aligned data, or install it once its packaging
  is finalized.
- **No `qc_overlay.mp4`** — install `ffmpeg` (`mamba install -c conda-forge
  ffmpeg`); the PNG montage is always written.
