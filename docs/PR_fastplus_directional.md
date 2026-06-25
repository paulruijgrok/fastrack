# FASTplus: directional (polarity-aware) two-channel gliding-motility analysis

Adds an opt-in **directional** analysis mode to FASTrack: using a second
fluorescence channel that labels one polar end of each filament (the "head"),
the package scores gliding motion as **signed** velocity — plus-end- vs
minus-end-directed — and fits the kinetics of optogenetic (LED) perturbations.

Branch: `fastplus-directional` → `main`.
8 commits · 28 files · ~3.9k lines added · default install and existing `fast`
behaviour unchanged.

---

## Why

FASTrack analyzes single-channel gliding assays and reports **unsigned** speeds.
For polarity experiments (e.g. actin barbed ends labelled with gelsolin, driven
by optogenetic myosins) we need to know the *direction* of motion relative to
the filament's intrinsic polarity, and how that velocity responds to light
pulses. This PR is the Python successor to the ImageJ/TrackMate-based v0.1
pipeline behind Ruijgrok et al., *Nat. Chem. Biol.* 2021 — now pure-Python,
automated, and integrated into FASTrack.

## Design approach

Built entirely on FASTrack's existing extension seams — **no edits to existing
call sites**:

- New strategies register in the existing registries (`DETECTORS`, a new
  `HEAD_TRACKERS`, `PIPELINES`); selected by name via `Settings`.
- A new `DirectionalSettings` section plugs into the layered `Settings`; a
  `fastplus` console entry point mirrors the `fast` CLI conventions.
- Niche dependencies (`optomerge` for two-channel registration, `tifffile` for
  robust multi-page TIFF IO) are isolated behind a `pip install 'fastrack[plus]'`
  extra, exactly like the existing `ridge` extra. The numpy-only core needs
  nothing new.

## What's included (maps to the original design requirements)

1. **Filament-centric directional tracking** (`gliding-directional`) — track
   filaments frame-to-frame, then attach a sign from the head sitting on one tip.
   (Currently uses a lightweight tracker — see Notes / limitations.)
2. **Head-centric tracking** (`polarity-head-centric`) — track the point-like
   heads; associate each head track, per frame, with an unambiguously labelled
   filament; and score its directionality. This is the approach suited to the
   high filament density and frequent crossings of the original data, where
   frame-to-frame filament tracking is unreliable.
   - Head detection: single-scale LoG ≈ TrackMate LoG detector
     (`core/detection/heads.py`, `heads-log`).
   - Head tracking: constant-velocity Kalman + Hungarian LAP + gap closing ≈
     TrackMate LinearMotionLAP (`core/tracking/head_tracker.py`, `kalman-lap`).
   - Association + **one-head-on-one-end** disambiguation gate; filaments with
     heads on both ends / in the middle / none are excluded
     (`polarity/association.py`, `polarity/disambiguation.py`).
3. **Per-frame directional averaging** across replicate movies, with central-
   percentile bands (`analysis/frame_average.py`).
4. **Kinetic-model fitting** keyed to external-perturbation timing — a global,
   **continuous** piecewise exponential rise/decay fit (shared dark baseline;
   each segment starts exactly where the previous ended), for N on/off cycles
   (`analysis/kinetics.py`, `analysis/perturbation.py`).
5. **Two-channel ingestion + registration** via the standalone `optomerge`
   package (optional; graceful fallback if absent) (`io/dual_channel.py`).
6. **Pure-Python, automated, memory-aware** throughout; movies processed one at
   a time, channels held as uint8, per-frame detection parallelized.

### Sign convention

Velocity is **positive when the motors stroke toward the (+)/barbed end**. For a
`head_marks_end="plus"` label (default; gelsolin/actin), a **lagging** head is
positive and a **leading** head negative; `"minus"` reverses it. Documented in
`polarity/scoring.py`, the `--head-marks` help, and the example configs.

### Perturbation (LED) schedule — three input routes

Resolved per movie with precedence `auto` = **sidecar → legacy `led.csv` →
config/CLI** (`analysis/perturbation.py`):

- per-movie sidecar `<movie>.perturb.toml` (also .json/.yaml) — easiest;
- legacy v0.1 `<base>.csv` + `<base> led.csv` (reproduces the original
  `find_switch_signal_files` / `get_switch_frames` logic; validated to switch
  frames [98, 298] on real data);
- explicit `switch_frames` / `perturbation_times_s` in config or CLI.

Handles ≥2 on/off cycles; LED-on intervals are shaded on the plot.

### QC outputs

Per movie: `directional_paths.csv`; `qc_overlay.png/.mp4` (filament contours +
heads colour-coded by polarity class). Per dataset: `frame_average.csv`,
`frame_average.png` (mean ± percentile bands, switch lines, lit shading,
continuous fit), `kinetics.txt`.

## Parallelism

Per-frame filament detection (the dominant cost) is mapped over `nprocs` worker
processes (`-j`) via a `multiprocessing.Pool`, mirroring `pipelines/gliding`;
head tracking and per-frame averaging stay on the parent. Serial vs all-core
outputs are **byte-for-byte identical** (CSVs, kinetics, plot pixels). Measured
**3.2× on 8 workers** on a 2-movie/800-frame set, `verify=OK` at every worker
count. See `docs/fastplus_parallel_verification.md` (+ benchmark script
`tools/benchmark_fastplus_parallel.py` and speedup plot).

## Testing

- `tests/test_fastplus_directional.py`: **30 tests** covering registry wiring,
  LoG detection, Kalman-LAP tracking (gap closing, crossing tracks), the
  disambiguation gate (all four cases), the signed-velocity convention,
  per-frame averaging + percentile bands, led.csv parsing → real switch frames,
  N-cycle segments, continuous-fit continuity (decay starts where rise ends),
  the parallel Pool plumbing, and discovery / error handling.
- Determinism verified on real data (serial vs parallel; see doc above).

## Try it

```bash
pip install -e .                      # base; [plus] enables optomerge registration
fastplus -d <DIR_OF_RGB_TIFFS> --mode head-centric --head-channel red \
    --head-quality 8 --spf 0.1356 --kinetic-model exp_rise_decay -v
# config-driven:
fastplus -d <DIR> --config config.plus.example.toml
```

## Notes / limitations

- The **filament-centric** path currently links filaments with a lightweight
  nearest-CM tracker rather than the package's greedy `Linker`; reconcile when
  that path is exercised in earnest. Head-centric is the validated path.
- The `optomerge` registration adapter (`io/dual_channel._apply_optomerge`) is
  written against an assumed API and needs validation against optomerge's real
  entry points; until then run with `--no-register` on pre-aligned data.
- Head detection and the across-movie loop remain serial (both are cheap
  relative to filament detection, and this keeps peak memory bounded).

---

## Future development

### 1. Caching for fast reruns (high priority)

The directional pipeline currently **does no caching** — every run repeats the
heavy detection from scratch, and `-f` / `force_analysis` is therefore a no-op.
Add a cache that **mirrors the FASTrack package's existing mechanism** (the
`io/stores` `STORES` registry + `io/export`), so reruns skip detection and `-f`
becomes meaningful:

- **Legacy layout** — one `.npy` per frame (per-frame `filXYs`-style records),
  matching FASTrack's default `cache_layout = "per-frame"`.
- **Modern standard** — one **per-movie** store:
  - a per-movie `.npy`/`.npz` of the `FilamentRecord`s (and head `SpotRecord`s /
    tracks), mirroring `cache_layout = "per-movie"`;
  - per-movie **CSV** exports: a *minimal* trajectory/contour CSV and a *full*
    long-format contour CSV, mirroring `--export-trajectories` /
    `--export-contours`.
- Cache keyed by movie identity + detector parameters (invalidate on change);
  wire `force_analysis` (`-f`) to bypass/refresh, and `recalculate` (`-r`) to
  re-derive scoring from cached detections without re-detecting.
- Reuse the `STORES` registry and `FilamentRecord`/`SpotRecord` datamodels so
  the formats are shared with `fast` rather than reinvented.

This is the single biggest workflow win: parameter sweeps on scoring /
association / fitting would no longer pay the filament-detection cost each time.

### 2. Other follow-ups

- Across-movie parallelism (one worker per movie) for large multi-movie runs,
  weighed against per-movie memory (`DESIGN_NOTES.md` §6).
- Reconcile the filament-centric tracker with the greedy `Linker`.
- Validate / finalize the `optomerge` registration adapter against its real API.
- Optional `laptrack` head tracker (exact Jaqaman LAP) behind `[plus-laptrack]`.
- Non-uniform frame timing from the acquisition times file (vs uniform `--spf`).
