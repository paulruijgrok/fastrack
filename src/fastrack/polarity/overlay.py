"""QC overlay rendering for directional analysis (FASTplus).

Draws, on sampled frames of the head channel, every detected head colour-coded
by its polarity classification, so the head<->filament association and the
one-head-on-one-end gate can be eyeballed:

    plus_end  -> green   (included: exactly one head on one tip)
    both_ends -> red     (excluded)
    middle    -> orange  (excluded)
    none      -> grey    (filament with no head; markers drawn on the tips)

Two outputs:
* :func:`save_classification_montage` -- a multi-frame PNG grid (matplotlib).
* :func:`save_overlay_movie`          -- an mp4 of the head channel with markers
  (imageio + opencv), one frame per analysed frame.

Both degrade gracefully: if matplotlib / imageio / cv2 is unavailable the
function returns ``None`` instead of raising.  Inputs are plain numpy arrays and
the :class:`~fastrack.polarity.datamodel.PolarFilament` records the pipeline
already computes, so this adds no new dependencies to the core.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .datamodel import BOTH_ENDS, MIDDLE, NONE, PLUS_END

# RGB 0-1 colours per classification
_COLORS = {
    PLUS_END: (0.15, 0.85, 0.15),
    BOTH_ENDS: (0.90, 0.15, 0.15),
    MIDDLE: (1.00, 0.60, 0.00),
    NONE: (0.55, 0.55, 0.55),
}
# BGR 0-255 for opencv
_COLORS_BGR = {k: (int(b * 255), int(g * 255), int(r * 255))
               for k, (r, g, b) in _COLORS.items()}


def _norm8(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, [1, 99.5])
    z = np.clip((img.astype(float) - lo) / max(hi - lo, 1e-9), 0, 1)
    return (z * 255).astype(np.uint8)


def _marked_points(pf) -> List[tuple]:
    """(x, y, classification) markers for a PolarFilament."""
    cls = pf.classification
    out = []
    if pf.plus_end_xy is not None:
        out.append((float(pf.plus_end_xy[0]), float(pf.plus_end_xy[1]), cls))
    # for excluded/none, mark the filament centre so it is still visible
    if not out and pf.cm is not None:
        out.append((float(pf.cm[0]), float(pf.cm[1]), cls))
    return out


def _contour_xy(fil):
    """Return a filament's contour as (N, 2) (x, y) or None.

    Filament contours are stored (row, col); convert to (x, y) for plotting.
    """
    c = getattr(fil, "contour", None)
    if c is None:
        return None
    c = np.asarray(c)
    if c.ndim != 2 or len(c) < 2:
        return None
    return c[:, ::-1]


_FIL_COLOR = (0.20, 0.55, 1.00)        # blue contour (matplotlib RGB)
_FIL_COLOR_BGR = (255, 140, 50)        # same in BGR for opencv


def save_classification_montage(
    head_stack: np.ndarray,
    polar_by_frame: Dict[int, Sequence],
    out_path: str,
    max_frames: int = 12,
    cols: int = 4,
    filament_by_frame: Optional[Dict[int, Sequence]] = None,
) -> Optional[str]:
    """Grid of frames with detected filament contours + heads coloured by class.

    ``filament_by_frame`` (optional) maps frame -> sequence of filament-like
    objects (``.contour``); their skeletons are drawn in blue beneath the head
    markers so filament-detection quality can be checked too.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return None

    n = min(max_frames, head_stack.shape[0])
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        ax = axes[i]
        ax.imshow(_norm8(head_stack[i]), cmap="gray")
        if filament_by_frame:
            for fil in filament_by_frame.get(i, []):
                cxy = _contour_xy(fil)
                if cxy is not None:
                    ax.plot(cxy[:, 0], cxy[:, 1], "-", color=_FIL_COLOR,
                            lw=0.8, alpha=0.7)
        for pf in polar_by_frame.get(i, []):
            for (x, y, cls) in _marked_points(pf):
                ax.plot(x, y, "o", mfc="none", mec=_COLORS[cls], ms=9, mew=1.4)
        ax.set_title("frame %d" % i, fontsize=8)
        ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                     markeredgecolor=c, label=k, markersize=8)
              for k, c in _COLORS.items()]
    legend.append(Line2D([0], [0], color=_FIL_COLOR, lw=1.5, label="filament"))
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def save_frame_average_plot(
    stats: Dict[str, np.ndarray],
    out_path: str,
    perturbation_times_s: Sequence[float] = (),
    kinetics: Optional[Sequence[dict]] = None,
) -> Optional[str]:
    """Plot mean signed velocity vs time with SEM band, perturbations, and fit.

    ``stats`` is the dict from :meth:`FrameVelocityAggregator.frame_means`
    (keys: time_s, mean, sem). ``kinetics`` (optional) is the list of fit-result
    dicts from :class:`~fastrack.analysis.kinetics.KineticModelFitter`; if given,
    each fitted curve is overlaid.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    t = np.asarray(stats["time_s"], float)
    m = np.asarray(stats["mean"], float)
    sem = np.asarray(stats["sem"], float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.fill_between(t, m - sem, m + sem, color="0.80", label="± SEM")
    ax.plot(t, m, "-", color="0.15", lw=1.5, label="mean signed velocity")

    for k, pt in enumerate(perturbation_times_s or ()):
        ax.axvline(pt, color="tab:purple", ls="--", lw=1.0,
                   label="perturbation" if k == 0 else None)

    if kinetics:
        from ..analysis.kinetics import KineticModelFitter
        for k, res in enumerate(kinetics):
            tt = np.linspace(res["t0"], t.max(), 200)
            yy = KineticModelFitter.predict(res["model"], tt, res)
            ax.plot(tt, yy, "-", color="tab:red", lw=1.8,
                    label=("fit: %s τ=%.2g s" % (res["model"], res["tau"]))
                    if k == 0 else "%s τ=%.2g s" % (res["model"], res["tau"]))

    ax.set_xlabel("time (s)")
    ax.set_ylabel("signed velocity (nm/s)\n(+ plus-end  /  − minus-end)")
    ax.set_title("Per-frame mean directional velocity")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_overlay_movie(
    head_stack: np.ndarray,
    polar_by_frame: Dict[int, Sequence],
    out_path: str,
    fps: int = 10,
    filament_by_frame: Optional[Dict[int, Sequence]] = None,
) -> Optional[str]:
    """mp4 of the head channel with filament contours + polarity markers.

    Returns the path on success, or ``None`` if no mp4 writer is available
    (e.g. imageio-ffmpeg not installed); any partially-written file is removed.
    """
    try:
        import imageio.v2 as imageio
    except Exception:
        return None
    try:
        import cv2
    except Exception:
        cv2 = None

    frames = []
    for t in range(head_stack.shape[0]):
        g = _norm8(head_stack[t])
        rgb = np.stack([g, g, g], axis=-1)
        if cv2 is not None:
            if filament_by_frame:
                for fil in filament_by_frame.get(t, []):
                    cxy = _contour_xy(fil)
                    if cxy is not None:
                        pts = np.round(cxy).astype(np.int32).reshape(-1, 1, 2)
                        cv2.polylines(rgb, [pts], False, _FIL_COLOR_BGR[::-1], 1)
            for pf in polar_by_frame.get(t, []):
                for (x, y, cls) in _marked_points(pf):
                    cv2.circle(rgb, (int(round(x)), int(round(y))), 6,
                               _COLORS_BGR[cls][::-1], 1)  # ::-1 -> RGB for imageio
        frames.append(rgb)
    try:
        imageio.mimsave(out_path, frames, fps=fps)
        return out_path
    except Exception:
        # clean up the partial/empty file imageio may have created
        import os
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        return None
