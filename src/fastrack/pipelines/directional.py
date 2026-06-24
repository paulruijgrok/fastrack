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
from collections import Counter, defaultdict
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
) -> Tuple[List, Counter]:
    """Head-centric directional scoring for one movie.

    ``head_frames[f]``      : list of detected (untracked) head ``SpotRecord`` in frame f.
    ``filament_frames[f]``  : sequence of filament-like objects (``.contour`` etc.) in frame f.

    Returns ``(directional_paths, qc_counts)`` where ``qc_counts`` tallies the
    polarity classifications (plus_end / both_ends / middle / none).
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
# Heavy helpers (deferred imports)
# --------------------------------------------------------------------------- #
def detect_filaments_in_stack(stack: np.ndarray, detection_algorithm: str = "entropy",
                              detection_params: Optional[dict] = None,
                              fast_rank: bool = True, morph_contrast: bool = False) -> List[list]:
    """Run the configured filament detector on each frame of an in-memory stack.

    Returns a list (per frame) of the live ``Filament`` objects.  Imports the
    heavy ``Frame`` / detector machinery lazily.
    """
    from ..core.frame import Frame
    from ..core.detection import DETECTORS
    from skimage.util import img_as_uint

    det = _make_detector(DETECTORS, detection_algorithm, detection_params,
                         fast_rank, morph_contrast)
    out = []
    for t in range(stack.shape[0]):
        frame = Frame()
        frame.frame_no = t
        img = stack[t]
        frame.img = img_as_uint(img) if img.dtype != np.uint16 else img
        frame.width, frame.height = frame.img.shape
        det.detect(frame)
        out.append(list(frame.filaments))
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
        end_fraction=0.15, max_end_distance_nm=500.0,
        pixel_size_nm=80.65, frame_rate_hz=None, max_inter_frame_distance_nm=2016.25,
        min_path_length=5, stuck_velocity_nm_s=80.0, num_frames_ave=5,
        detection_algorithm="entropy", detection_params=None,
        perturbation_times_s=(), kinetic_model="none",
        force_analysis=False, nprocs=None, verbose=False,
        output_dir=None, **_ignored):
    """Directional analysis over every ``*RGB.tif`` movie under ``main_dir``.

    Collects per-frame signed velocities across all movies (identical format),
    writes per-movie and combined outputs, and -- if perturbation times and a
    kinetic model are given -- fits exp rise/decay to the frame-averaged trace.
    """
    from ..core.tracking import HEAD_TRACKERS
    from ..io.dual_channel import TwoChannelMovie
    from ..io.export import write_rows_csv  # tidy CSV writer (see io.export)

    movies = find_rgb_movies(main_dir)
    if verbose:
        print("[fastplus] %d RGB movies under %s" % (len(movies), main_dir))
    out_root = output_dir or os.path.join(main_dir, "fastplus_out")
    os.makedirs(out_root, exist_ok=True)

    dt_s = (1.0 / frame_rate_hz) if frame_rate_hz else 1.0
    scorer = DirectionalScorer(pixel_size_nm=pixel_size_nm, dt_s=dt_s,
                               stuck_velocity_nm_s=stuck_velocity_nm_s)
    associator = HeadFilamentAssociator(
        max_end_distance_px=max_end_distance_nm / pixel_size_nm,
        end_fraction=end_fraction)
    classifier = PolarityClassifier()

    aggregator = FrameVelocityAggregator(dt_s=dt_s)
    total_qc: Counter = Counter()

    for path in movies:
        tracker = HEAD_TRACKERS.create(
            head_tracking_algorithm, initial_search_radius=initial_search_radius,
            kalman_search_radius=kalman_search_radius, max_frame_gap=max_frame_gap)
        if verbose:
            print("[fastplus]   %s" % os.path.basename(path))

        movie = TwoChannelMovie(path, head_channel, filament_channel,
                                channel_map, register=register_channels)
        head_stack, fil_stack = movie.split()
        movie.release()

        head_frames = detect_heads_in_stack(
            head_stack, gaussian_sigma=head_sigma, radius=head_radius,
            quality_threshold=head_quality, subpixel=head_subpixel)
        filament_frames = detect_filaments_in_stack(
            fil_stack, detection_algorithm=detection_algorithm,
            detection_params=detection_params)

        if mode == "filament-centric":
            paths = _filament_centric_movie(
                filament_frames, head_frames, scorer, associator, classifier,
                tracker, max_inter_frame_distance_nm / pixel_size_nm)
            dpaths, qc = paths
        else:
            dpaths, qc = analyze_head_centric(
                head_frames, filament_frames, scorer=scorer, associator=associator,
                classifier=classifier, head_tracker=tracker)

        total_qc.update(qc)
        aggregator.add_movie(dpaths)

        mdir = os.path.join(out_root, _safe(path, main_dir))
        os.makedirs(mdir, exist_ok=True)
        rows = [r for dp in dpaths for r in dp.to_rows()]
        write_rows_csv(rows, os.path.join(mdir, "directional_paths.csv"),
                       ["path_id", "source", "frame", "time_s", "signed_velocity_nm_s"])

    # combined per-frame averages across all movies
    fa_rows = aggregator.to_rows()
    write_rows_csv(fa_rows, os.path.join(out_root, "frame_average.csv"),
                   ["frame", "time_s", "mean_signed_velocity_nm_s", "sem_nm_s", "n"])

    # kinetic fit
    kinetics = None
    if kinetic_model != "none" and fa_rows:
        st = aggregator.frame_means()
        fitter = KineticModelFitter(perturbation_times_s)
        if kinetic_model == "exp_rise_decay":
            kinetics = fitter.fit_segments(st["time_s"], st["mean"])
        else:
            kinetics = [fitter.fit(st["time_s"], st["mean"], model=kinetic_model)]
        _write_kinetics(os.path.join(out_root, "kinetics.txt"), kinetics, total_qc)

    if verbose:
        print("[fastplus] classifications:", dict(total_qc))
        print("[fastplus] outputs ->", out_root)
    return {"movies": len(movies), "qc": dict(total_qc),
            "frame_average": fa_rows, "kinetics": kinetics, "output_dir": out_root}


def _filament_centric_movie(filament_frames, head_frames, scorer, associator,
                            classifier, tracker, max_velocity_px):
    """Track filaments with the greedy linker, then score signed velocity."""
    from ..core.tracking import LINKERS
    # Build minimal frame-like carriers for the existing linker is heavyweight;
    # for the directional add-on we link filament centres of mass greedily here.
    paths = _greedy_cm_paths(filament_frames, max_velocity_px)
    return analyze_filament_centric(paths, head_frames, scorer=scorer,
                                    associator=associator, classifier=classifier,
                                    head_tracker=tracker)


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
