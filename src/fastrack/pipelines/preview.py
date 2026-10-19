"""Preview mode: a fast first look at a gliding-assay movie.

Meant for the microscope: point it at a movie (or a folder of movies, possibly
still being acquired) and within seconds get

* the fraction of **stuck** filaments,
* the filament **length** distribution,
* a **gliding velocity** estimate, and
* an **image-quality report** that ranks what limits the data (signal-to-noise,
  focus, illumination flatness, background, saturation, filament density,
  floppiness / poor attachment, bleaching, frame rate) with a suggested action.

Speed comes from doing less, not from a different algorithm: only ``n_frames``
frames are read (default 20, the first or -- with ``latest`` -- the most recent
ones), frames are 2x2-binned before anything else, preprocessing and the ridge
detector (``ridge-fast`` when installed) run on the small frames, per-frame
detection is spread over all cores, nothing is cached or written except the
small report, and linking reuses the standard FAST greedy linker and path code.

Stuck filaments are **counted, not tracked**: the temporal-median background
that preprocessing subtracts contains exactly the objects that do not move, so
the static filaments are detected on that image.  (Tracking cannot see them --
the background subtraction removes them -- which is also why the full pipeline's
``p-stuck`` under-counts filaments that are stuck for the whole movie.)

All numbers are estimates for deciding "is this worth recording / analysing";
the full ``fast`` pipeline remains the reference.  See ``docs/preview.md``.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from .. import preprocess as _pp
from ..core.input import open_source
from ..core.input.base import to_pipeline_image

#: Ridge settings (full-resolution pixels) validated on the taka data set.
DEFAULT_RIDGE = dict(line_widths=[3, 5], low_contrast=60.0, high_contrast=120.0,
                     min_len=8.0, min_snr=3.0)


@dataclass
class PreviewSettings:
    n_frames: int = 20                 # frames analysed (consecutive)
    latest: bool = False               # use the most recent frames (live acquisition)
    bin: int = 2                       # 2x2 binning before analysis (1 = off)
    detector: str = "ridge-fast"       # falls back to "ridge" if not installed
    ridge: Dict = field(default_factory=lambda: dict(DEFAULT_RIDGE))
    preprocess: Dict = field(default_factory=lambda: dict(_pp.DEFAULT_PARAMS))
    pixel_size_nm: float = 80.65
    max_inter_frame_distance_nm: float = 2016.25
    stuck_velocity_nm_s: float = 80.0
    num_frames_ave: int = 5
    frame_rate_hz: Optional[float] = None   # needed for stacks without timestamps
    nprocs: Optional[int] = None       # None -> all cores


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bin(img, b):
    if b <= 1:
        return np.asarray(img, dtype=np.float32)
    h, w = img.shape
    h, w = h - h % b, w - w % b
    a = np.asarray(img[:h, :w], dtype=np.float32)
    return a.reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def _ridge_params(ridge, b):
    """Full-resolution ridge settings -> settings for b-binned frames."""
    p = dict(ridge)
    if b > 1:
        p["line_widths"] = sorted({max(1, int(round(w / b))) for w in ridge["line_widths"]})
        p["min_len"] = float(ridge["min_len"]) / b
    return p


FOCUS_CROP = 512      # full-resolution centre crop used for the focus metric


def _center_crop(img, size):
    h, w = img.shape
    if h <= size and w <= size:
        return np.asarray(img)
    r0, c0 = max(0, (h - size) // 2), max(0, (w - size) // 2)
    return np.asarray(img[r0:r0 + size, c0:c0 + size])


def _focus_task(crops, s, algo):
    """Detection task for the full-resolution focus crop (middle frame)."""
    stack = np.stack([c.astype(np.float32) for c in crops])
    pp = dict(s.preprocess)
    pp["model_frames"] = 0
    m = _pp.fit_model(stack, **pp)
    img = to_pipeline_image(m.apply(stack[len(stack) // 2]))
    return (img, -1, m.output_noise_sd(8) * 257.0, algo, _ridge_params(s.ridge, 1))


def _detector_name(name):
    if name == "ridge-fast":
        try:
            import ridge_detector_fast  # noqa: F401
        except ImportError:
            return "ridge"
    return name


def _contour_signal(img_f, contour):
    """Median of the (1-px smoothed) image along an integer [row, col] contour."""
    r = np.clip(contour[:, 0], 0, img_f.shape[0] - 1)
    c = np.clip(contour[:, 1], 0, img_f.shape[1] - 1)
    return float(np.median(img_f[r, c]))


def _detect(task):
    """Worker: detect one frame; return filXYs + per-ridge signal (img units)."""
    from ..core.detection import DETECTORS
    from ..core.frame import Frame
    img, no, noise, algo, params = task
    fr = Frame()
    fr.img = img
    fr.width, fr.height = img.shape
    fr.noise_sd = noise
    fr.frame_no = no
    DETECTORS.create(algo, **params).detect(fr)
    sm = gaussian_filter(img.astype(np.float32), 1.0)
    sig = [_contour_signal(sm, np.asarray(e[0])) for e in fr.filXYs]
    return no, fr.filXYs, sig


def _pool(nprocs):
    from multiprocessing import Pool, cpu_count
    n = nprocs or cpu_count()
    return Pool(processes=max(1, n)) if n > 1 else None


# --------------------------------------------------------------------------- #
# quality scoring
# --------------------------------------------------------------------------- #
#: name -> (label, unit, good, poor, advice).  Score 1 at ``good``, 0 at
# ``poor``, linear in between (either direction; see docs/preview.md).  Anchors were calibrated on the 38-movie Taka set (bin 2, 20
# frames; Oct 2026) plus noise-aware blur/dim simulations: e.g. full-res focus
# width is ~265-275 nm in focus and ~370 nm at a 3 px Gaussian defocus; filament
# SNR drops from ~12 to ~4 when the signal is cut 8x.  Treat them as starting
# points to tune per setup.
QUALITY_SPECS = {
    "snr": ("Filament signal-to-noise", "", 12.0, 4.0,
            "increase exposure / laser power, or reduce background fluorescence"),
    "focus_width_nm": ("Filament width (focus)", "nm", 300.0, 380.0,
                       "refocus (filaments look wide/blurred)"),
    "flatness": ("Illumination flatness (p5/p95 gain)", "", 0.80, 0.50,
                 "re-center / flatten the illumination (TIRF angle, beam alignment)"),
    "saturation_pct": ("Saturated pixels", "%", 0.01, 0.5,
                       "reduce exposure / laser power (pixels clip)"),
    "background_to_signal": ("Background / filament signal", "", 6.0, 15.0,
                             "reduce background (wash out free fluorophore, check TIRF angle)"),
    "density_low": ("Filaments per 100 um^2 (too few)", "", 1.0, 0.3,
                    "add more filaments / move to a denser area (too few for statistics)"),
    "density_high": ("Filaments per 100 um^2 (crowded)", "", 8.0, 14.0,
                     "dilute filaments (crowding breaks tracking)"),
    "stuck_pct": ("Stuck filaments", "%", 15.0, 40.0,
                  "check surface / motor quality, blocking, ATP (many filaments stuck)"),
    "floppy_pct": ("Floppy / poorly attached tracks", "%", 15.0, 35.0,
                   "check motor density / surface attachment (filaments flail)"),
    "bleach_pct_per_frame": ("Bleaching", "%/frame", 1.0, 3.0,
                             "reduce laser power / exposure or add oxygen scavenger"),
    "step_over_length": ("Displacement per frame / filament length", "", 0.6, 1.0,
                         "increase the frame rate (filaments move too far per frame)"),
}


def _score(value, good, poor):
    if value is None or not np.isfinite(value):
        return None
    if good == poor:
        return 1.0
    return float(np.clip((value - poor) / (good - poor), 0.0, 1.0))


def score_quality(metrics):
    """Rank the quality metrics; worst first.  Returns a list of dicts."""
    rows = []
    for key, (label, unit, good, poor, advice) in QUALITY_SPECS.items():
        src = "density" if key.startswith("density") else key
        v = metrics.get(src)
        s = _score(v, good, poor)
        if s is None:
            continue
        rows.append(dict(metric=key, label=label, value=v, unit=unit, score=s,
                         status="good" if s >= 0.7 else ("marginal" if s >= 0.35 else "poor"),
                         advice=advice))
    rows.sort(key=lambda r: r["score"])
    return rows


def verdict(rows):
    if not rows:
        return "unknown"
    worst = rows[0]["score"]
    return "GOOD" if worst >= 0.7 else ("MARGINAL" if worst >= 0.35 else "POOR")


# --------------------------------------------------------------------------- #
# per-movie analysis
# --------------------------------------------------------------------------- #
def analyse_movie(descriptor, settings: PreviewSettings = None, pool=None, keep_images=False):
    """Preview one movie (a frame-source descriptor).  Returns a result dict."""
    from ..core.detection import DETECTORS
    from ..core.frame import Frame
    from ..core.motility import Motility

    s = settings or PreviewSettings()
    T = {}
    t0 = time.perf_counter()
    b = max(1, int(s.bin))
    dx = s.pixel_size_nm * b

    # ----- read (+ saturation on the raw pixels) + bin ---------------------- #
    src = open_source(descriptor)
    nos = list(src.frame_numbers())
    if s.latest:
        # newest n frames, skipping the newest file (it may still be being written)
        sel = nos[-(s.n_frames + 1):-1] if len(nos) > 1 else nos
    else:
        sel = nos[:s.n_frames]
    frames_raw, used, sat, crops = [], [], [], []
    for no in sel:
        try:
            r = src.read_raw(no)
        except Exception:
            continue
        top = {np.uint8: 255, np.uint16: 65535}.get(r.dtype.type)
        mx = int(r.max())
        if top and mx in (255, 4095, 16383, 65535):
            sat.append(float(np.count_nonzero(r == mx)) / r.size)
        else:
            sat.append(0.0)
        frames_raw.append(_bin(r, b))
        crops.append(_center_crop(r, FOCUS_CROP))
        used.append(no)
    if len(used) < 3:
        raise ValueError("need at least 3 readable frames, found %d" % len(used))
    T["read"] = time.perf_counter() - t0

    # ----- preprocessing (fit on the preview frames) ------------------------ #
    t = time.perf_counter()
    pp = dict(s.preprocess)
    pp["flat_sigma"] = float(pp.get("flat_sigma", 50.0)) / b
    pp["model_frames"] = 0
    raw = np.stack(frames_raw)
    model = _pp.fit_model(raw, **pp)
    # Background = per-pixel 2nd-smallest value of the preview frames, not the
    # temporal median: over only ~20 frames a filament that moves less than
    # ~2 lengths covers some pixels in over half the frames, so the median
    # would contain it and subtracting it would cut moving filaments short.
    # A low order statistic holds only what is present in (all but one of)
    # the frames: background + stuck filaments, which are counted separately
    # below.  Its downward noise bias is a uniform pedestal the ridge detector
    # ignores.  It is taken on the unsmoothed frames because apply() denoises
    # *after* subtracting -- a pre-smoothed background would be blurred twice
    # and leave ridges behind.  Gain field and stretch come from the median fit.
    k = 1 if len(raw) > 3 else 0
    mn = np.partition(raw, k, axis=0)[k]
    model.background = mn.astype(np.float32)
    imgs = [to_pipeline_image(model.apply(r)) for r in frames_raw]
    noise_img = model.output_noise_sd(8) * 257.0
    T["preprocess"] = time.perf_counter() - t

    # ----- detection (parallel) -------------------------------------------- #
    t = time.perf_counter()
    algo = _detector_name(s.detector)
    params = _ridge_params(s.ridge, b)
    tasks = [(im, no, noise_img, algo, params) for im, no in zip(imgs, used)]
    # Focus is judged at full resolution (binning hides modest defocus): the
    # centre crop of the middle frame, corrected with its own small model.
    ftask = _focus_task(crops, s, algo)
    if pool is not None:
        fres = pool.apply_async(_detect, (ftask,))
        res = pool.map(_detect, tasks)
        focus_fx = fres.get()[1]
    else:
        res = [_detect(x) for x in tasks]
        focus_fx = _detect(ftask)[1]
    filxys = {no: f for no, f, _ in res}
    signal = {no: sg for no, _, sg in res}
    T["detect"] = time.perf_counter() - t

    # ----- static (stuck) filaments: detect on the low-order background ----- #
    # Background subtraction removes stuck filaments from the frames, so they
    # are counted on that background itself, which keeps only structure present
    # in (all but one of) the preview frames: a filament
    # that moves by more than its own length within the window leaves no trace,
    # however slowly it moves (a median would keep anything covering a pixel in
    # over half the frames).  So "static" here means slower than about
    # length / window duration -- with 20 frames at 1 s and 1.5 um filaments,
    # ~75 nm/s, close to the default stuck threshold.
    t = time.perf_counter()
    mn = gaussian_filter(mn, model.denoise_sigma)
    flat = mn - gaussian_filter(mn, 15.0 / b)
    if model.gain is not None:
        flat = flat / model.gain
    # noise of this image, measured on itself (filaments are sparse)
    static_noise = 1.4826 * float(np.median(np.abs(flat - np.median(flat))))
    sframe = Frame()
    sframe.img = to_pipeline_image(np.clip(flat / model.hi * 255.0, 0, 255).astype(np.uint8))
    sframe.width, sframe.height = sframe.img.shape
    sframe.noise_sd = static_noise / model.hi * 255.0 * 257.0
    DETECTORS.create(algo, **params).detect(sframe)
    static = sframe.filXYs
    T["static"] = time.perf_counter() - t

    # ----- linking + paths (standard FAST code) ----------------------------- #
    t = time.perf_counter()
    frames = []
    for no in used:
        fr = Frame()
        fr.frame_no = no
        fr.filXYs = filxys[no]
        fr.filXY2filaments()
        frames.append(fr)
    m = Motility()
    m.dx = dx
    m.max_velocity = s.max_inter_frame_distance_nm / dx
    m.max_length_dif = 5.0 / b
    m.min_velocity = s.stuck_velocity_nm_s
    et = src.elapsed_times()
    if s.frame_rate_hz:
        m.elapsed_times, m.dt = [], 1.0 / s.frame_rate_hz
        timing = "frame_rate"
    elif et is not None and len(et) >= max(used) + 1:
        m.elapsed_times = list(et)
        timing = "metadata"
    else:
        m.elapsed_times, m.dt = [], 1.0
        timing = "assumed 1 s/frame"
    m.frame1 = frames[0]
    for fr in frames[1:]:
        m.frame2 = fr
        m.make_frame_links()
        m.frame1 = fr
    m.process_frame_links(s.num_frames_ave)
    T["link"] = time.perf_counter() - t

    # ----- results ---------------------------------------------------------- #
    t = time.perf_counter()
    stats = None
    if len(m.full_len_vel):
        stats = m.plot_length_velocity(print_plot=False, fit_f="none", min_points=5,
                                       min_path_length=s.num_frames_ave)
        if stats[0] == -1:
            stats = None
    flv = m.full_len_vel if len(m.full_len_vel) else np.zeros((0, 4))
    moving_v = flv[flv[:, 1] > 0, 1] if len(flv) else np.zeros(0)

    det_len_um = np.array([fl.fil_length for fr in frames for fl in fr.filaments]) * dx * 1e-3
    static_len_um = np.array([len(np.asarray(e[0])) for e in static], float) * dx * 1e-3
    widths_nm = np.array([e[1] for e in focus_fx], float) * s.pixel_size_nm
    sig_all = np.concatenate([np.asarray(signal[no], float) for no in used]) if used else np.zeros(0)
    snr_all = sig_all / max(noise_img, 1e-9)

    n_moving = float(np.mean([len(filxys[no]) for no in used]))
    n_static = len(static)
    area_100um2 = (frames_raw[0].size * dx * dx * 1e-6) / 100.0

    # floppiness: moving paths that wander instead of gliding (net / total
    # centre-of-mass travel).  The cm, not the contour midpoint: the midpoint
    # is one quantised contour pixel that jumps when the contour's length or
    # ordering changes, which made slow and crowded movies look floppy.
    straight = []
    vel_cv = []
    for p in m.paths:
        if getattr(p, "stuck", False) or len(p.links) < s.num_frames_ave:
            continue
        pts = [np.asarray(p.links[-1].filament1_cm, float)]
        for lk in p.links[::-1]:
            pts.append(np.asarray(lk.filament2_cm, float))
        pts = np.array(pts)
        steps = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
        tot = steps.sum()
        if tot > 0:
            straight.append(np.sqrt(((pts[-1] - pts[0]) ** 2).sum()) / tot)
        iv = np.abs([lk.instant_velocity for lk in p.links])
        if iv.mean() > 0:
            vel_cv.append(iv.std() / iv.mean())
    straight = np.array(straight)
    floppy_pct = 100.0 * float(np.mean(straight < 0.5)) if len(straight) else None

    # bleaching: per-frame median ridge signal, linear trend relative to start
    per_frame = np.array([np.median(signal[no]) if len(signal[no]) else np.nan for no in used])
    ok = np.isfinite(per_frame)
    bleach = None
    if ok.sum() >= 5 and per_frame[ok][0] > 0:
        slope = np.polyfit(np.arange(len(per_frame))[ok], per_frame[ok], 1)[0]
        bleach = float(-100.0 * slope / np.median(per_frame[ok][:3]))

    gain = model.gain
    flatness = (float(np.percentile(gain, 5) / np.percentile(gain, 95))
                if gain is not None else None)
    step_nm = []
    for p in m.paths:
        if getattr(p, "stuck", False):
            continue
        for lk in p.links:
            step_nm.append(abs(lk.distance_score) * dx)
    med_len_nm = float(np.median(det_len_um) * 1e3) if len(det_len_um) else np.nan

    metrics = dict(
        snr=float(np.median(snr_all)) if len(snr_all) else None,
        focus_width_nm=float(np.median(widths_nm)) if len(widths_nm) else None,
        flatness=flatness,
        saturation_pct=100.0 * float(np.mean(sat)),
        background_to_signal=float(np.median(model.background) / model.hi),
        density=n_moving / area_100um2,
        stuck_pct=100.0 * n_static / (n_static + n_moving) if (n_static + n_moving) else None,
        floppy_pct=floppy_pct,
        bleach_pct_per_frame=bleach,
        step_over_length=(float(np.median(step_nm)) / med_len_nm
                          if step_nm and np.isfinite(med_len_nm) and med_len_nm > 0 else None),
    )
    ranking = score_quality(metrics)

    result = dict(
        n_frames=len(used), frames=[int(x) for x in used], bin=b, detector=algo,
        timing_source=timing, pixel_size_nm=s.pixel_size_nm,
        moving_per_frame=n_moving, static_filaments=n_static,
        stuck_pct=metrics["stuck_pct"],
        tracked_points=int(len(flv)),
        velocity_nm_s=dict(
            mean_moving=float(stats[3]) if stats else (float(moving_v.mean()) if len(moving_v) else None),
            median_moving=float(np.median(moving_v)) if len(moving_v) else None,
            top5=float(stats[0]) if stats else None,
        ),
        length_um=dict(
            median=float(np.median(det_len_um)) if len(det_len_um) else None,
            mean=float(np.mean(det_len_um)) if len(det_len_um) else None,
            p10=float(np.percentile(det_len_um, 10)) if len(det_len_um) else None,
            p90=float(np.percentile(det_len_um, 90)) if len(det_len_um) else None,
            static_median=float(np.median(static_len_um)) if len(static_len_um) else None,
        ),
        metrics=metrics, quality=ranking, verdict=verdict(ranking),
        velocity_cv_median=float(np.median(vel_cv)) if vel_cv else None,
    )
    T["stats"] = time.perf_counter() - t
    T["total"] = time.perf_counter() - t0
    result["seconds"] = {k: round(v, 3) for k, v in T.items()}
    if keep_images:
        result["_plot"] = dict(img=model.apply(frames_raw[len(frames_raw) // 2]),
                               moving=filxys[used[len(used) // 2]], static=static,
                               lengths_um=det_len_um, static_lengths_um=static_len_um,
                               velocities=moving_v)
    return result


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _fmt(v, nd=0):
    return "-" if v is None else ("%.*f" % (nd, v))


def format_report(name, r):
    v, L = r["velocity_nm_s"], r["length_um"]
    lines = [
        "%s  [%s]  %d frames, bin %d, %s, %.1f s"
        % (name, r["verdict"], r["n_frames"], r["bin"], r["detector"], r["seconds"]["total"]),
        "  velocity   mean %s nm/s, median %s, top-5%% %s   (%d tracked points; timing: %s)"
        % (_fmt(v["mean_moving"]), _fmt(v["median_moving"]), _fmt(v["top5"]),
           r["tracked_points"], r["timing_source"]),
        "  stuck      %s %%  (%d static vs %.0f moving filaments/frame)"
        % (_fmt(r["stuck_pct"], 1), r["static_filaments"], r["moving_per_frame"]),
        "  length     median %s um (p10 %s, p90 %s, mean %s)"
        % (_fmt(L["median"], 2), _fmt(L["p10"], 2), _fmt(L["p90"], 2), _fmt(L["mean"], 2)),
        "  quality (worst first):",
    ]
    for q in r["quality"]:
        lines.append("    %-9s %-40s %8s %-8s%s"
                     % (q["status"].upper(), q["label"], _fmt(q["value"], 2), q["unit"],
                        ("-> " + q["advice"]) if q["status"] != "good" else ""))
    return "\n".join(lines)


def plot_dashboard(name, r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        _plot_dashboard(plt, name, r, path)


def _plot_dashboard(plt, name, r, path):
    # rcParamsDefault: other FASTrack plotting modules set large global fonts.
    P = r["_plot"]
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.7, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    img = P["img"]
    ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.8))
    for e in P["moving"]:
        c = np.asarray(e[0])
        ax.plot(c[:, 1], c[:, 0], "-", color="lime", lw=0.8)
    for e in P["static"]:
        c = np.asarray(e[0])
        ax.plot(c[:, 1], c[:, 0], "-", color="red", lw=1.0)
    ax.set_title("%s  [%s]\nmoving (green) / stuck (red)" % (name, r["verdict"]), fontsize=10)
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, max(6.0, np.percentile(P["lengths_um"], 99) if len(P["lengths_um"]) else 6), 30)
    ax.hist(P["lengths_um"], bins=bins, color="tab:green", alpha=0.7, label="moving")
    if len(P["static_lengths_um"]):
        ax.hist(P["static_lengths_um"], bins=bins, color="tab:red", alpha=0.7, label="stuck")
    ax.set_xlabel("filament length (um)")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[0, 2])
    if len(P["velocities"]):
        ax.hist(P["velocities"], bins=30, color="tab:blue", alpha=0.8)
        if r["velocity_nm_s"]["mean_moving"]:
            ax.axvline(r["velocity_nm_s"]["mean_moving"], color="k", ls="--")
    ax.set_xlabel("velocity (nm/s)")
    ax.set_title("mean %s nm/s, stuck %s %%" % (_fmt(r["velocity_nm_s"]["mean_moving"]),
                                               _fmt(r["stuck_pct"], 1)), fontsize=10)
    ax = fig.add_subplot(gs[1, 1:])
    q = r["quality"][::-1]
    colors = ["tab:green" if x["status"] == "good" else ("orange" if x["status"] == "marginal"
              else "tab:red") for x in q]
    ax.barh([x["label"] for x in q], [x["score"] for x in q], color=colors)
    for i, x in enumerate(q):
        ax.text(0.01, i, "%s %s" % (_fmt(x["value"], 2), x["unit"]), va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("quality score (1 = good)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _descriptors(path, frame_rate):
    from .discovery import discover_movies
    out = []
    for mv in discover_movies(path):
        if mv["kind"] == "mm":
            d = {"kind": "mm_dir", "directory": os.path.abspath(mv["input"])}
        else:
            d = {"kind": "tiff_stack", "path": os.path.abspath(mv["input"]),
                 "frame_rate": frame_rate or 1.0}
        full = os.path.abspath(mv["input"])
        if os.path.basename(full).lower().startswith("pos") and mv["kind"] == "mm":
            full = os.path.dirname(full)                   # name the movie, not Pos0
        base = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path))
        name = os.path.relpath(full, os.path.dirname(base))
        out.append((name, d))
    return out


def run(path, settings: PreviewSettings = None, out_dir=None, plot=True, quiet=False):
    """Preview every movie under ``path``; print reports, write JSON/CSV/PNG.

    Returns ``{name: result}``.  Outputs (when ``out_dir`` is given): one
    ``<movie>_preview.json`` and (``plot``) ``<movie>_preview.png`` per movie,
    plus ``preview_summary.csv``.
    """
    s = settings or PreviewSettings()
    movies = _descriptors(path, s.frame_rate_hz)
    if not movies:
        raise SystemExit("no movies found under %s" % path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results = {}
    pool = _pool(s.nprocs)
    try:
        for name, d in movies:
            try:
                r = analyse_movie(d, s, pool=pool, keep_images=bool(plot and out_dir))
            except Exception as exc:                       # fail-isolated per movie
                if not quiet:
                    print("%s  FAILED: %s" % (name, exc))
                results[name] = {"error": str(exc)}
                continue
            if not quiet:
                print(format_report(name, r))
            safe = name.replace(os.sep, "__")
            if out_dir:
                if plot:
                    t = time.perf_counter()
                    plot_dashboard(name, r, os.path.join(out_dir, safe + "_preview.png"))
                    r["seconds"]["plot"] = round(time.perf_counter() - t, 3)
                r.pop("_plot", None)
                with open(os.path.join(out_dir, safe + "_preview.json"), "w") as f:
                    json.dump(dict(movie=name, settings=asdict(s), **r), f, indent=1, default=float)
            r.pop("_plot", None)
            results[name] = r
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if out_dir:
        _write_summary(results, os.path.join(out_dir, "preview_summary.csv"))
    return results


def _write_summary(results, path):
    import csv
    keys = list(QUALITY_SPECS)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movie", "verdict", "seconds", "frames", "mean_velocity_nm_s",
                    "median_velocity_nm_s", "top5_velocity_nm_s", "stuck_pct",
                    "median_length_um", "moving_per_frame", "worst_metric"] +
                   ["%s" % k for k in keys])
        for name, r in results.items():
            if "error" in r:
                w.writerow([name, "FAILED: " + r["error"]])
                continue
            q = {x["metric"]: x["value"] for x in r["quality"]}
            w.writerow([name, r["verdict"], r["seconds"]["total"], r["n_frames"],
                        r["velocity_nm_s"]["mean_moving"], r["velocity_nm_s"]["median_moving"],
                        r["velocity_nm_s"]["top5"], r["stuck_pct"], r["length_um"]["median"],
                        round(r["moving_per_frame"], 1),
                        r["quality"][0]["metric"] if r["quality"] else ""] +
                       [q.get(k) for k in keys])


__all__ = ["DEFAULT_RIDGE", "PreviewSettings", "QUALITY_SPECS", "analyse_movie",
           "format_report", "plot_dashboard", "run", "score_quality", "verdict"]
