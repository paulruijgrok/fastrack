"""FASTplus directional analysis driver (head-centric and filament-centric).

Two pipelines, registered under :data:`PIPELINES`:

* ``polarity-head-centric`` -- track the polarity *heads*, associate each head
  track (per frame) with an unambiguously labelled filament, and score signed
  velocity along the filament's polar axis.  This is the high-density / frequent-
  crossing regime of the Ruijgrok et al. data.
* ``gliding-directional`` -- run the normal filament detection + linking, then
  attach a sign to each filament path from the head sitting on one tip.

The reusable, numpy-only scoring core lives in :func:`analyze_head_centric` /
:func:`analyze_filament_centric`; the pipeline classes and :func:`run` wrap it
with movie discovery, two-channel IO, filament detection, per-frame averaging
across movies, and kinetic fitting.  Heavy image-processing imports (``Frame``)
are deferred so the scoring core can be unit-tested without scipy/scikit-image.
"""
from __future__ import annotations

import os
import warnings
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import PIPELINES, Pipeline
from ..analysis.frame_average import FrameVelocityAggregator
from ..analysis.kinetics import KineticModelFitter
from ..polarity import (DirectionalScorer, HeadFilamentAssociator,
                        PolarityClassifier, SpotTable)
from ..polarity.datamodel import PolarFilament


# --------------------------------------------------------------------------- #
# Reusable numpy-only scoring cores
# --------------------------------------------------------------------------- #
def analyze_head_centric(
    head_frames: Sequence[Sequence],
    filament_frames: Sequence[Sequence],
    *,
    scorer: DirectionalScorer,
    associator: HeadFilamentAssociator,
    classifier: PolarityClassifier,
    head_tracker,
    elapsed_times: Optional[Sequence[float]] = None,
    polar_by_frame: Optional[dict] = None,
) -> Tuple[List, Counter]:
    """Head-centric directional scoring for one movie.

    ``head_frames[f]``      : list of detected (untracked) head ``SpotRecord`` in frame f.
    ``filament_frames[f]``  : sequence of filament-like objects (``.contour`` etc.) in frame f.

    Returns ``(directional_paths, qc_counts)`` where ``qc_counts`` tallies the
    polarity classifications (plus_end / both_ends / middle / none).  If a
    ``polar_by_frame`` dict is given, it is filled with frame -> list of
    classified ``PolarFilament`` (for QC overlays).
    """
    # 1. track the heads across the whole movie
    all_spots = [s for fr in head_frames for s in fr]
    tracked = head_tracker.track(all_spots)            # sets s.track_id
    spots_by_frame: Dict[int, list] = defaultdict(list)
    for s in tracked:
        spots_by_frame[s.frame].append(s)

    # 2. per-frame association + disambiguation -> polar axis per head track
    axis_by_track: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
    qc: Counter = Counter()
    for f, fils in enumerate(filament_frames):
        heads = spots_by_frame.get(f, [])
        polar = associator.associate_frame(fils, heads, f)
        classifier.classify_all(polar)
        if polar_by_frame is not None:
            polar_by_frame[f] = polar
        for pf in polar:
            qc[pf.classification] += 1
            if pf.is_unambiguous and pf.head_ids:
                axis = pf.polarity_vector
                if axis is not None:
                    axis_by_track[pf.head_ids[0]][f] = axis

    # 3. signed-velocity scoring for head tracks that were ever unambiguous
    paths = []
    for tid, spots in SpotTable(tracked).tracks().items():
        axes = axis_by_track.get(tid)
        if not axes:
            continue
        dp = scorer.score_head_track(tid, spots, axes, elapsed_times)
        if dp.n_steps() > 0:
            paths.append(dp)
    return paths, qc


def analyze_filament_centric(
    filament_paths: Sequence[dict],
    head_frames: Sequence[Sequence],
    *,
    scorer: DirectionalScorer,
    associator: HeadFilamentAssociator,
    classifier: PolarityClassifier,
    head_tracker,
    elapsed_times: Optional[Sequence[float]] = None,
    polar_by_frame: Optional[dict] = None,
) -> Tuple[List, Counter]:
    """Filament-centric directional scoring for one movie.

    ``filament_paths`` is the output of the normal filament tracker: each item is
    ``{"path_id", "frames": [...], "positions": [(x,y),...],
       "filaments": [filament-like,...]}``.  Polarity is taken from the head that
    sits on one tip (classified over the path's frames); the path's signed
    velocity uses that fixed polar axis.
    """
    all_spots = [s for fr in head_frames for s in fr]
    tracked = head_tracker.track(all_spots)
    spots_by_frame: Dict[int, list] = defaultdict(list)
    for s in tracked:
        spots_by_frame[s.frame].append(s)

    paths, qc = [], Counter()
    for path in filament_paths:
        frames = list(path["frames"])
        fils = path.get("filaments", [])
        # classify polarity on each frame of the path; adopt the majority call
        axis_votes = []
        for fr, fil in zip(frames, fils):
            polar = associator.associate_frame([fil], spots_by_frame.get(fr, []), fr)
            classifier.classify_all(polar)
            pf = polar[0] if polar else None
            if pf is not None:
                qc[pf.classification] += 1
                if polar_by_frame is not None:
                    polar_by_frame.setdefault(fr, []).append(pf)
                if pf.is_unambiguous and pf.polarity_vector is not None:
                    axis_votes.append(pf.polarity_vector)
        if not axis_votes:
            continue
        axis = np.mean(axis_votes, axis=0)
        axis = axis / (np.linalg.norm(axis) or 1.0)
        dp = scorer.score_filament_path(path["path_id"], path["positions"],
                                        frames, axis, elapsed_times)
        if dp.n_steps() > 0:
            paths.append(dp)
    return paths, qc


# --------------------------------------------------------------------------- #
# Parallelism helpers
# --------------------------------------------------------------------------- #
def resolve_workers(nprocs) -> int:
    """Number of worker processes: ``None`` -> all cores, else the given count."""
    if nprocs is None:
        return max(1, cpu_count())
    return max(1, int(nprocs))


def _pmap(func, tasks, nprocs=None, chunksize=None):
    """Map ``func`` over ``tasks``; serial if 1 worker, else a process Pool.

    ``func`` must be a top-level (picklable) function.  Results preserve the
    order of ``tasks`` (``Pool.imap``).
    """
    tasks = list(tasks)
    workers = resolve_workers(nprocs)
    if workers <= 1 or len(tasks) <= 1:
        return [func(t) for t in tasks]
    if chunksize is None:
        chunksize = max(1, len(tasks) // (workers * 4))
    with Pool(processes=min(workers, len(tasks))) as pool:
        return list(pool.imap(func, tasks, chunksize=chunksize))


def _selftest_double(x):
    """Trivial picklable worker used to unit-test the Pool plumbing."""
    return x * 2


# --------------------------------------------------------------------------- #
# Heavy helpers (deferred imports)
# --------------------------------------------------------------------------- #
def _detect_one_frame(task):
    """Per-frame filament-detection worker (picklable; runs in a child process).

    ``task`` = (img, frame_idx, algorithm, params, fast_rank, morph_contrast).
    Returns (frame_idx, list[FilamentRecord]) -- lightweight, picklable records
    rather than live Frame/Filament objects, so they cross the process boundary
    cheaply and carry exactly what association/scoring need (contour, cm, length).
    """
    img, idx, algo, params, fast_rank, morph_contrast = task
    from ..core.frame import Frame
    from ..core.detection import DETECTORS
    from ..datamodel import FilamentRecord
    try:
        from skimage.util import img_as_uint
    except Exception:
        img_as_uint = lambda x: x  # noqa: E731
    det = _make_detector(DETECTORS, algo, params, fast_rank, morph_contrast)
    frame = Frame()
    frame.frame_no = idx
    frame.img = img_as_uint(img) if img.dtype != np.uint16 else img
    frame.width, frame.height = frame.img.shape
    try:
        det.detect(frame)
        recs = [FilamentRecord.from_filament(f) for f in frame.filaments]
    except Exception as exc:                      # one bad frame shouldn't abort
        warnings.warn("frame %d filament detection failed: %s" % (idx, exc))
        recs = []
    return idx, recs


def detect_filaments_in_stack(stack: np.ndarray, detection_algorithm: str = "entropy",
                              detection_params: Optional[dict] = None,
                              fast_rank: bool = True, morph_contrast: bool = False,
                              nprocs=None) -> List[list]:
    """Detect filaments on every frame of an in-memory stack, in parallel.

    Per-frame detection (the dominant cost) is mapped over ``nprocs`` worker
    processes (``None`` -> all cores).  Returns a list (per frame) of picklable
    ``FilamentRecord`` objects; ordering matches the stack.
    """
    n = int(stack.shape[0])
    tasks = [(stack[t], t, detection_algorithm, detection_params,
              fast_rank, morph_contrast) for t in range(n)]
    out: List[list] = [[] for _ in range(n)]
    for idx, recs in _pmap(_detect_one_frame, tasks, nprocs):
        out[idx] = recs
    return out


def _make_detector(DETECTORS, name, params, fast_rank, morph_contrast):
    params = params or {}
    if name in ("ridge", "ridge-fast"):
        return DETECTORS.create(name, **params)
    return DETECTORS.create(name, fast_rank=fast_rank, morph_contrast=morph_contrast)


def detect_heads_in_stack(stack: np.ndarray, *, gaussian_sigma=1.5, radius=5.0,
                          quality_threshold=5.0, subpixel=True) -> List[list]:
    """Detect head spots per frame; returns a list (per frame) of ``SpotRecord``."""
    from ..core.detection.heads import detect_spots
    return [detect_spots(stack[t], frame=t, gaussian_sigma=gaussian_sigma,
                         radius=radius, quality_threshold=quality_threshold,
                         subpixel=subpixel) for t in range(stack.shape[0])]


# --------------------------------------------------------------------------- #
# Per-movie worker (picklable; used serially or one-per-process across movies)
# --------------------------------------------------------------------------- #
def _process_one_movie(task: dict) -> dict:
    """Run the full per-movie pipeline and return aggregatable results.

    Writes this movie's own outputs (directional_paths.csv, overlays, detection
    CSVs, cache) and returns ``{name, dpaths, qc, pert, error}`` for the parent
    to pool into the combined averages.  Catches per-movie errors so one bad
    movie cannot abort a multi-movie run.
    """
    from ..core.tracking import HEAD_TRACKERS
    from ..io.dual_channel import TwoChannelMovie
    from ..io.detection_cache import DetectionCache
    from ..io.export import write_rows_csv
    from ..analysis import perturbation as _pert

    t = task
    name = os.path.basename(t["path"])
    try:
        scorer = DirectionalScorer(pixel_size_nm=t["pixel_size_nm"], dt_s=t["dt_s"],
                                   stuck_velocity_nm_s=t["stuck_velocity_nm_s"],
                                   head_marks_end=t["head_marks_end"])
        associator = HeadFilamentAssociator(
            max_end_distance_px=t["max_end_distance_nm"] / t["pixel_size_nm"],
            end_fraction=t["end_fraction"])
        classifier = PolarityClassifier()
        tracker = HEAD_TRACKERS.create(
            t["head_tracking_algorithm"],
            initial_search_radius=t["initial_search_radius"],
            kalman_search_radius=t["kalman_search_radius"],
            max_frame_gap=t["max_frame_gap"])

        pert = _pert.resolve(
            t["path"], source=t["perturbation_source"],
            config_switch_frames=t["switch_frames"] or None,
            config_times_s=t["perturbation_times_s"] or None,
            config_states=t["perturbation_states"] or None,
            frame_interval_s=t["base_dt"], verbose=t["verbose"])

        mdir = os.path.join(t["out_root"], _safe(t["path"], t["main_dir"]))
        os.makedirs(mdir, exist_ok=True)

        # --- detection cache (reuses STORES; keyed by movie + detection params) #
        common = {"register": t["register_channels"], "channel_map": t["channel_map"],
                  "max_frames": t["max_frames"], "frame_step": t["frame_step"]}
        fil_params = dict(common, detector=t["detection_algorithm"],
                          params=t["detection_params"], channel=t["filament_channel"])
        head_params = dict(common, channel=t["head_channel"], sigma=t["head_sigma"],
                           radius=t["head_radius"], quality=t["head_quality"],
                           subpixel=t["head_subpixel"])
        layout = t["detection_cache_layout"]
        fil_cache = DetectionCache(mdir, "fil", fil_params, layout=layout)
        head_cache = DetectionCache(mdir, "head", head_params, layout=layout)

        force = t["force_analysis"]
        overlay = t["overlay"]
        head_stack = None
        cache_hit = (not force and fil_cache.count() > 0
                     and fil_cache.count() == head_cache.count())
        if cache_hit and not overlay:
            filament_frames = fil_cache.load()
            head_frames = head_cache.load()
        else:
            movie = TwoChannelMovie(t["path"], t["head_channel"], t["filament_channel"],
                                    t["channel_map"], register=t["register_channels"])
            head_stack, fil_stack = movie.split()
            movie.release()
            if t["max_frames"] or t["frame_step"] > 1:
                sl = slice(0, t["max_frames"], t["frame_step"] if t["frame_step"] > 1 else None)
                head_stack, fil_stack = head_stack[sl], fil_stack[sl]
            n = fil_stack.shape[0]
            if not force and fil_cache.has_all(n):
                filament_frames = fil_cache.load()
            else:
                filament_frames = detect_filaments_in_stack(
                    fil_stack, detection_algorithm=t["detection_algorithm"],
                    detection_params=t["detection_params"], nprocs=t["nprocs"])
                fil_cache.save(filament_frames)
            if not force and head_cache.has_all(n):
                head_frames = head_cache.load()
            else:
                head_frames = detect_heads_in_stack(
                    head_stack, gaussian_sigma=t["head_sigma"], radius=t["head_radius"],
                    quality_threshold=t["head_quality"], subpixel=t["head_subpixel"])
                head_cache.save(head_frames)

        if t["export_detections"] or t["export_detection_contours"]:
            _export_detection_csvs(mdir, filament_frames, head_frames,
                                   contours=t["export_detection_contours"])

        polar_by_frame = {} if overlay else None
        if t["mode"] == "filament-centric":
            dpaths, qc = _filament_centric_movie(
                filament_frames, head_frames, scorer, associator, classifier,
                tracker, t["max_inter_frame_distance_nm"] / t["pixel_size_nm"],
                polar_by_frame=polar_by_frame)
        else:
            dpaths, qc = analyze_head_centric(
                head_frames, filament_frames, scorer=scorer, associator=associator,
                classifier=classifier, head_tracker=tracker,
                polar_by_frame=polar_by_frame)

        rows = [r for dp in dpaths for r in dp.to_rows()]
        write_rows_csv(rows, os.path.join(mdir, "directional_paths.csv"),
                       ["path_id", "source", "frame", "time_s", "signed_velocity_nm_s"])

        if overlay:
            from ..polarity.overlay import (save_classification_montage,
                                            save_overlay_movie)
            fil_by_frame = {i: fl for i, fl in enumerate(filament_frames)}
            save_classification_montage(
                head_stack, polar_by_frame, os.path.join(mdir, "qc_overlay.png"),
                max_frames=t["montage_frames"], filament_by_frame=fil_by_frame)
            save_overlay_movie(
                head_stack, polar_by_frame, os.path.join(mdir, "qc_overlay.mp4"),
                fps=t["overlay_fps"], filament_by_frame=fil_by_frame)

        return {"name": name, "dpaths": dpaths, "qc": dict(qc),
                "pert": pert if pert else None, "error": None}
    except Exception as exc:                          # never let one movie abort the run
        return {"name": name, "dpaths": [], "qc": {}, "pert": None,
                "error": "%s: %s" % (type(exc).__name__, exc)}


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
@PIPELINES.register("polarity-head-centric")
class HeadCentricPipeline(Pipeline):
    def run(self, main_dir, settings):
        return run(main_dir, **settings.to_directional_kwargs())


@PIPELINES.register("gliding-directional")
class FilamentCentricDirectionalPipeline(Pipeline):
    def run(self, main_dir, settings):
        kw = settings.to_directional_kwargs()
        kw["mode"] = "filament-centric"
        return run(main_dir, **kw)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def find_rgb_movies(root: str, suffix: str = "rgb.tif") -> List[str]:
    hits = []
    for dp, _d, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(suffix):
                hits.append(os.path.join(dp, fn))
    return sorted(hits)


def run(main_dir, *, mode="head-centric", head_channel="red", filament_channel="green",
        channel_map="", register_channels=True,
        head_sigma=1.5, head_radius=5.0, head_quality=5.0, head_subpixel=True,
        head_tracking_algorithm="kalman-lap", initial_search_radius=20.0,
        kalman_search_radius=15.0, max_frame_gap=4,
        end_fraction=0.15, max_end_distance_nm=500.0, head_marks_end="plus",
        pixel_size_nm=80.65, frame_rate_hz=None, max_inter_frame_distance_nm=2016.25,
        min_path_length=5, stuck_velocity_nm_s=80.0, num_frames_ave=5,
        detection_algorithm="entropy", detection_params=None,
        perturbation_source="auto", switch_frames=(), perturbation_times_s=(),
        perturbation_states=(), kinetic_model="none",
        percentiles=(14, 86, 2, 98),
        detection_cache_layout="per-movie", export_detections=False,
        export_detection_contours=False,
        force_analysis=False, recalculate=False, nprocs=None, verbose=False,
        parallel_movies=1, limit=None, max_frames=None, frame_step=1,
        overlay=False, overlay_fps=10, montage_frames=12,
        output_dir=None, **_ignored):
    """Directional analysis over every ``*RGB.tif`` movie under ``main_dir``.

    Collects per-frame signed velocities across all movies (identical format),
    writes per-movie and combined outputs, and -- if perturbation times and a
    kinetic model are given -- fits exp rise/decay to the frame-averaged trace.
    """
    from ..core.tracking import HEAD_TRACKERS
    from ..io.dual_channel import TwoChannelMovie
    from ..io.export import write_rows_csv  # tidy CSV writer (see io.export)
    from ..io.detection_cache import DetectionCache
    from ..analysis import perturbation as _pert

    if not main_dir or not os.path.isdir(main_dir):
        raise NotADirectoryError(
            "fastplus -d: directory not found: %r\n"
            "  (paths are resolved from the current working directory; pass an "
            "absolute path if unsure)" % (main_dir,))

    movies = find_rgb_movies(main_dir)
    if limit:
        movies = movies[:limit]
    if not movies:
        print("[fastplus] no *RGB.tif movies found under %s" % main_dir)
        print("[fastplus] (discovery matches files whose name ends in 'RGB.tif', "
              "case-insensitive, searched recursively)")
        return {"movies": 0, "qc": {}, "frame_average": [], "kinetics": None,
                "output_dir": None}
    if verbose:
        print("[fastplus] %d RGB movie(s) under %s" % (len(movies), main_dir))

    out_root = output_dir or os.path.join(main_dir, "fastplus_out")
    os.makedirs(out_root, exist_ok=True)

    # time between *analysed* frames (folds in temporal subsampling)
    base_dt = (1.0 / frame_rate_hz) if frame_rate_hz else 1.0
    dt_s = base_dt * max(1, int(frame_step))

    aggregator = FrameVelocityAggregator(dt_s=dt_s)
    total_qc: Counter = Counter()
    combined_pert = None          # representative switch schedule for the pooled fit

    # Parallelism — two mutually-exclusive ways to spend cores:
    #   * within-movie (default): movies run serially here and per-frame filament
    #     detection is mapped over `nprocs` workers inside detect_filaments_in_stack.
    #   * across-movie (`parallel_movies` > 1): each movie runs in its own worker
    #     process (detecting serially, since a Pool worker is daemonic and cannot
    #     spawn its own Pool). Better for many small movies; peak memory scales
    #     with `parallel_movies` x the per-movie footprint.
    # Head tracking, per-frame averaging, and the combined fit always run on the
    # parent (cross-movie state); each worker writes only its own per-movie outputs.
    movie_workers = max(1, int(parallel_movies))
    per_frame_nprocs = 1 if movie_workers > 1 else nprocs

    def _make_task(path):
        return {
            "path": path, "main_dir": main_dir, "out_root": out_root,
            "mode": mode, "head_channel": head_channel,
            "filament_channel": filament_channel, "channel_map": channel_map,
            "register_channels": register_channels,
            "head_sigma": head_sigma, "head_radius": head_radius,
            "head_quality": head_quality, "head_subpixel": head_subpixel,
            "head_tracking_algorithm": head_tracking_algorithm,
            "initial_search_radius": initial_search_radius,
            "kalman_search_radius": kalman_search_radius,
            "max_frame_gap": max_frame_gap, "end_fraction": end_fraction,
            "max_end_distance_nm": max_end_distance_nm,
            "head_marks_end": head_marks_end, "pixel_size_nm": pixel_size_nm,
            "max_inter_frame_distance_nm": max_inter_frame_distance_nm,
            "stuck_velocity_nm_s": stuck_velocity_nm_s,
            "detection_algorithm": detection_algorithm,
            "detection_params": detection_params,
            "detection_cache_layout": detection_cache_layout,
            "export_detections": export_detections,
            "export_detection_contours": export_detection_contours,
            "perturbation_source": perturbation_source,
            "switch_frames": switch_frames,
            "perturbation_times_s": perturbation_times_s,
            "perturbation_states": perturbation_states,
            "base_dt": base_dt, "dt_s": dt_s,
            "force_analysis": force_analysis, "nprocs": per_frame_nprocs,
            "max_frames": max_frames, "frame_step": frame_step,
            "overlay": overlay, "overlay_fps": overlay_fps,
            "montage_frames": montage_frames, "verbose": verbose,
        }

    tasks = [_make_task(p) for p in movies]

    def _consume(res):
        nonlocal combined_pert
        if verbose:
            print("[fastplus]   %s%s" % (res["name"],
                  "" if not res.get("error") else "  ERROR: " + res["error"]))
        if res.get("error"):
            warnings.warn("movie failed: %s (%s)" % (res["name"], res["error"]))
            return
        total_qc.update(res["qc"])
        aggregator.add_movie(res["dpaths"])
        pert = res.get("pert")
        if pert:
            if combined_pert is None:
                combined_pert = pert
            elif list(pert.switch_frames) != list(combined_pert.switch_frames):
                warnings.warn(
                    "perturbation switch frames differ across movies (%s vs %s); "
                    "using the first for the pooled kinetic fit"
                    % (combined_pert.switch_frames, pert.switch_frames))

    if verbose:
        if movie_workers > 1:
            print("[fastplus] across-movie workers: %d (per-frame detection serial)"
                  % min(movie_workers, len(tasks)))
        else:
            print("[fastplus] filament detection workers: %d" % resolve_workers(nprocs))

    if movie_workers > 1 and len(tasks) > 1:
        with Pool(processes=min(movie_workers, len(tasks))) as pool:
            for res in pool.imap(_process_one_movie, tasks):
                _consume(res)
    else:
        for t in tasks:
            _consume(_process_one_movie(t))

    # combined per-frame averages across all movies
    fa_rows = aggregator.to_rows()
    write_rows_csv(fa_rows, os.path.join(out_root, "frame_average.csv"),
                   ["frame", "time_s", "mean_signed_velocity_nm_s", "sem_nm_s", "n"])
    st = aggregator.frame_means() if fa_rows else None

    # switch schedule -> seconds for fitting / plotting (frame-step independent)
    switch_times_plot, lit_intervals = [], []
    segments = []
    if combined_pert and st is not None:
        n_frames_seg = int(round(float(st["time_s"].max()) / base_dt)) + 1
        segments = combined_pert.segments(n_frames_seg, base_dt)
        switch_times_plot = combined_pert.switch_times_s(base_dt)
        lit_intervals = [(a * base_dt, b * base_dt)
                         for (a, b) in combined_pert.on_off_frames()]

    # kinetic fit
    kinetics = None          # legacy per-segment list (fallback)
    cont = None              # continuous piecewise fit (preferred)
    fit_curve = fit_label = None
    if kinetic_model != "none" and st is not None:
        if segments:
            from ..analysis.kinetics import fit_continuous
            cont = fit_continuous(st["time_s"], st["mean"], segments)
            if cont:
                fit_curve = (cont["curve_t"], cont["curve_v"])
                taus = ", ".join("τ_%s=%.2g s" % (c["kind"], c["tau"])
                                 for c in cont["cycles"])
                fit_label = "continuous fit (%s)" % taus
                _write_kinetics_continuous(
                    os.path.join(out_root, "kinetics.txt"), cont, total_qc)
        else:
            fitter = KineticModelFitter(perturbation_times_s)
            kinetics = [fitter.fit(st["time_s"], st["mean"], model=kinetic_model)]
            _write_kinetics(os.path.join(out_root, "kinetics.txt"), kinetics, total_qc)

    # central-percentile bands (flat lower,upper pairs -> per-frame lo/hi)
    bands = None
    if st is not None and percentiles:
        flat = list(percentiles)
        pairs = list(zip(flat[0::2], flat[1::2]))
        raw = aggregator.frame_percentile_bands(pairs)
        bands = [(lo, hi, "%g–%g%%" % pairs[i]) for i, (lo, hi) in enumerate(raw)]

    # velocity-vs-time plot (always; overlays switches, lit regions, fit if present)
    if st is not None:
        from ..polarity.overlay import save_frame_average_plot
        plot = save_frame_average_plot(
            st, os.path.join(out_root, "frame_average.png"),
            perturbation_times_s=switch_times_plot, kinetics=kinetics,
            lit_intervals_s=lit_intervals, fit_curve=fit_curve, fit_label=fit_label,
            bands=bands)
        if verbose and plot:
            print("[fastplus] plot ->", plot)

    if verbose:
        print("[fastplus] classifications:", dict(total_qc))
        print("[fastplus] outputs ->", out_root)
    return {"movies": len(movies), "qc": dict(total_qc),
            "frame_average": fa_rows, "kinetics": cont or kinetics,
            "output_dir": out_root}


def _filament_centric_movie(filament_frames, head_frames, scorer, associator,
                            classifier, tracker, max_velocity_px,
                            polar_by_frame=None):
    """Track filaments with the greedy linker, then score signed velocity."""
    # Building minimal frame-like carriers for the existing linker is heavyweight;
    # for the directional add-on we link filament centres of mass greedily here.
    paths = _greedy_cm_paths(filament_frames, max_velocity_px)
    return analyze_filament_centric(paths, head_frames, scorer=scorer,
                                    associator=associator, classifier=classifier,
                                    head_tracker=tracker,
                                    polar_by_frame=polar_by_frame)


def _greedy_cm_paths(filament_frames, max_velocity_px):
    """Lightweight nearest-CM filament linking -> path dicts (numpy only)."""
    paths, open_paths, next_id = [], [], 0
    for f, fils in enumerate(filament_frames):
        cms = [np.asarray(getattr(fl, "cm", None), float)[::-1] if getattr(fl, "cm", None) is not None
               else None for fl in fils]
        used = set()
        for op in open_paths:
            best, bestd = None, max_velocity_px
            for i, cm in enumerate(cms):
                if cm is None or i in used:
                    continue
                d = np.linalg.norm(cm - op["positions"][-1])
                if d < bestd:
                    best, bestd = i, d
            if best is not None:
                used.add(best)
                op["frames"].append(f); op["positions"].append(cms[best])
                op["filaments"].append(fils[best]); op["miss"] = 0
            else:
                op["miss"] = op.get("miss", 0) + 1
        open_paths = [op for op in open_paths if op.get("miss", 0) <= 2]
        for i, cm in enumerate(cms):
            if cm is None or i in used:
                continue
            open_paths.append({"path_id": next_id, "frames": [f], "positions": [cm],
                               "filaments": [fils[i]], "miss": 0})
            next_id += 1
        paths = [p for p in paths]  # keep
    # collect all (open_paths still hold the data we appended in place)
    all_paths = {}
    for op in open_paths:
        all_paths[op["path_id"]] = op
    return list(all_paths.values())


def _export_detection_csvs(mdir, filament_frames, head_frames, contours=False):
    """Per-movie CSV exports of the cached detections (modern-standard outputs).

    ``filaments_minimal.csv`` (one row per filament) + ``heads.csv`` always; the
    full long-format ``filaments_contours.csv`` (one row per contour point) when
    ``contours`` is set. Mirrors fast's --export-trajectories / --export-contours.
    """
    from ..io.export import write_rows_csv

    frows = []
    for f, recs in enumerate(filament_frames):
        for label, r in enumerate(recs):
            row = dict(r.to_row()) if hasattr(r, "to_row") else {}
            row.setdefault("frame", f)
            row.setdefault("label", label)
            frows.append(row)
    write_rows_csv(frows, os.path.join(mdir, "filaments_minimal.csv"),
                   ["frame", "label", "n_points", "length", "density", "width",
                    "area", "end2end", "midpoint_x", "midpoint_y", "path_id"])

    hrows = [s.to_row() for fr in head_frames for s in fr]
    write_rows_csv(hrows, os.path.join(mdir, "heads.csv"),
                   ["frame", "track_id", "x", "y", "quality", "radius"])

    if contours:
        crows = []
        for f, recs in enumerate(filament_frames):
            for label, r in enumerate(recs):
                cont = getattr(r, "contour", None)
                if cont is None:
                    continue
                for p, rc in enumerate(np.asarray(cont)):
                    crows.append({"frame": f, "label": label, "point": p,
                                  "x": float(rc[1]), "y": float(rc[0])})
        write_rows_csv(crows, os.path.join(mdir, "filaments_contours.csv"),
                       ["frame", "label", "point", "x", "y"])


def _safe(path, root):
    rel = os.path.relpath(path, root)
    return os.path.splitext(rel)[0].replace(os.sep, "__").replace(" ", "_")


def _write_kinetics(path, kinetics, qc):
    with open(path, "w") as fh:
        fh.write("FASTplus kinetic fit\n====================\n")
        fh.write("classifications: %s\n\n" % dict(qc))
        for k in kinetics:
            fh.write("model=%s  t0=%.3g  v0=%.4g  amp=%.4g  tau=%.4g  R2=%.3f  n=%d\n"
                     % (k["model"], k["t0"], k["v0"], k["amp"], k["tau"], k["r2"], k["n"]))


def _write_kinetics_continuous(path, cont, qc):
    """Write the continuous piecewise fit (one shared dark baseline A0)."""
    with open(path, "w") as fh:
        fh.write("FASTplus continuous piecewise kinetic fit\n")
        fh.write("=========================================\n")
        fh.write("classifications: %s\n" % dict(qc))
        fh.write("dark baseline A0 = %.4g nm/s   overall R2 = %.3f\n\n"
                 % (cont["A0"], cont["r2"]))
        fh.write("%-6s %-6s %10s %12s %12s %10s\n" %
                 ("cycle", "kind", "tau(s)", "start(nm/s)", "level(nm/s)", "t0(s)"))
        for i, c in enumerate(cont["cycles"]):
            fh.write("%-6d %-6s %10.3g %12.1f %12.1f %10.2f\n" %
                     (i, c["kind"], c["tau"], c["start_level"], c["level"], c["t0_s"]))
