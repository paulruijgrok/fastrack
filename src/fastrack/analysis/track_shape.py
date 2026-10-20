"""Per-track attachment metrics from contour trajectories: floppy and pinned.

Both work on a tracked :class:`~fastrack.core.link.Path` (its links carry the
filament contours of consecutive frames) and are used by ``fast-preview``.

**Floppy -- leaving the track.**  A filament gliding on a well-attached motor
lawn follows its own track ("follow the leader"): each part of the new contour
lies on the curve of the previous contour, except the leading end that has
moved past it.  For each step, the distance from every new contour point to the
*curve* of the old contour is measured (nearest point on the densely resampled
polyline), ignoring new points that project beyond the old contour's ends.
Gliding filaments sit at ~0 (pixel rounding); a filament that bends, swings or
flails moves part of itself sideways off the track.

This is deliberately *not* FAST's index-matched orthogonal distance
(``Filament.sim_score``'s ``area_score``): that compares contour point *i* in
one frame with point *i* in the next, so on curved filaments that slide several
pixels along their track it measures chords across the bend and grows with
speed (~0.7 -> 2.5 binned px on Taka data) even for perfect gliding.

A step only counts if the new contour still *touches* the old one somewhere
(min distance <= ``touch_nm``): a flailing filament keeps part of itself on its
track, whereas a link to a different filament (a mislink in a crowded field)
or a step past the whole old contour does not.  A step is *lateral* when the
90th percentile of the point distances exceeds ``lateral_nm`` (>= 10 % of the
contour left the track).  A track is floppy when >= ``frac`` of its valid
steps are lateral.

**Pinned -- one end anchored, the other sweeping.**  Both tips are followed
along the track, with tip identity kept consistent through each link's
``fil_direction`` (contour orientation relative to the previous frame).  A track
is pinned when one tip stays put (radius of gyration <= ``fixed_rg_nm``) while
the other sweeps (radius of gyration >= max(``free_rg_nm``, ``free_frac`` x
filament length)).
"""
import numpy as np
from scipy.spatial import cKDTree


def _dense(contour, step=0.25):
    """Polyline resampled every ``step`` px, and the arc length of each sample."""
    c = np.asarray(contour, float).reshape(-1, 2)
    if len(c) < 2:
        return c, np.zeros(len(c))
    seg = np.sqrt(((c[1:] - c[:-1]) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return c[:1], np.zeros(1)
    t = np.arange(0.0, s[-1] + 1e-9, step)
    return np.column_stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])]), t


def off_track(c_old, c_new, touch_px=1.0, end_px=1.0):
    """``(valid, p90)`` for one step: does the new contour still touch the old
    one, and the 90th-percentile distance (px) of its points from the old
    contour's curve (excluding points beyond the old ends)."""
    pts, s = _dense(c_old)
    if len(pts) < 3:
        return False, np.nan
    d, j = cKDTree(pts).query(np.asarray(c_new, float).reshape(-1, 2))
    keep = (s[j] > end_px) & (s[j] < s[-1] - end_px)
    if keep.sum() < 3:
        return False, np.nan
    dk = d[keep]
    return bool(dk.min() <= touch_px), float(np.percentile(dk, 90))


def floppy_track(path, dx_nm, lateral_nm=320.0, touch_nm=160.0, min_valid=3):
    """``(n_valid_steps, lateral_fraction)`` of one track (fraction is NaN when
    fewer than ``min_valid`` steps are valid).  ``dx_nm``: pixel size of the
    contours."""
    vals = [off_track(lk.filament1_contour, lk.filament2_contour, touch_px=touch_nm / dx_nm)
            for lk in path.links]
    good = np.array([p for ok, p in vals if ok], float)
    if len(good) < min_valid:
        return len(good), np.nan
    return len(good), float(np.mean(good > lateral_nm / dx_nm))


def tip_tracks(path):
    """Positions of the two tips along a path, earliest frame first: two
    ``(n_frames, 2)`` arrays, tip identity kept by ``fil_direction``."""
    links = path.links[::-1]                  # paths are stored latest link first
    c = np.asarray(links[0].filament1_contour, float).reshape(-1, 2)
    a, b = [c[0]], [c[-1]]
    orient = 1
    for lk in links:
        orient *= 1 if lk.fil_direction >= 0 else -1
        c = np.asarray(lk.filament2_contour, float).reshape(-1, 2)
        first, last = (c[0], c[-1]) if orient > 0 else (c[-1], c[0])
        a.append(first)
        b.append(last)
    return np.array(a), np.array(b)


def _rg(x):
    return float(np.sqrt(((x - x.mean(axis=0)) ** 2).sum(axis=1).mean()))


def pinned_track(path, dx_nm, fixed_rg_nm=120.0, free_rg_nm=320.0, free_frac=0.2):
    """``(is_pinned, rg_fixed_nm, rg_free_nm)`` for one track."""
    a, b = tip_tracks(path)
    ra, rb = _rg(a) * dx_nm, _rg(b) * dx_nm
    fixed, free = min(ra, rb), max(ra, rb)
    length_nm = float(np.mean([lk.filament1_length for lk in path.links])) * dx_nm
    pinned = fixed <= fixed_rg_nm and free >= max(free_rg_nm, free_frac * length_nm)
    return bool(pinned), fixed, free


def attachment_metrics(paths, dx_nm, min_links=5, frac=0.3, **kw):
    """Floppy and pinned fractions over the tracks of one movie.

    Returns ``dict(floppy_pct, pinned_pct, n_floppy_tracks, n_pinned_tracks,
    floppy_paths, pinned_paths)``.  Floppy: % of moving tracks (with >=
    ``min_valid`` valid steps) whose lateral-step fraction is >= ``frac``.
    Pinned: % of all tracks with >= ``min_links`` links (stuck ones included --
    a pinned filament's centre barely moves).
    """
    fkw = {k: kw[k] for k in ("lateral_nm", "touch_nm", "min_valid") if k in kw}
    pkw = {k: kw[k] for k in ("fixed_rg_nm", "free_rg_nm", "free_frac") if k in kw}
    floppy, n_fl = [], 0
    pinned, n_pin = [], 0
    for p in paths:
        if len(p.links) < min_links:
            continue
        n_pin += 1
        if pinned_track(p, dx_nm, **pkw)[0]:
            pinned.append(p)
        if getattr(p, "stuck", False):
            continue
        _, f = floppy_track(p, dx_nm, **fkw)
        if np.isfinite(f):
            n_fl += 1
            if f >= frac:
                floppy.append(p)
    return dict(
        floppy_pct=100.0 * len(floppy) / n_fl if n_fl else None,
        pinned_pct=100.0 * len(pinned) / n_pin if n_pin else None,
        n_floppy_tracks=n_fl, n_pinned_tracks=n_pin,
        floppy_paths=floppy, pinned_paths=pinned,
    )
