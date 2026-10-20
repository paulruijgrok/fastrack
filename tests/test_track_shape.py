"""Floppy / pinned attachment metrics from contour trajectories (analysis/track_shape.py)."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from fastrack.analysis import track_shape as TS  # noqa: E402
from fastrack.core.link import Link, Path  # noqa: E402

DX = 160.0          # nm per (binned) pixel


def _arc(centre, radius, a0, a1, n=40):
    t = np.linspace(a0, a1, n)
    return np.rint(np.column_stack([centre[0] + radius * np.sin(t),
                                    centre[1] + radius * np.cos(t)])).astype(int)


def _path(contours, fil_dirs=None, stuck=False):
    """Path from per-frame contours (earliest first); links stored latest first
    like Motility.create_paths."""
    links = []
    for i in range(len(contours) - 1):
        lk = Link()
        lk.frame1_no, lk.frame2_no = i, i + 1
        lk.filament1_contour, lk.filament2_contour = contours[i], contours[i + 1]
        lk.filament1_length = lk.filament2_length = float(len(contours[i]))
        lk.fil_direction = 1 if fil_dirs is None else fil_dirs[i]
        links.append(lk)
    p = Path()
    p.links = links[::-1]
    p.stuck = stuck
    return p


def _gliding_on_arc(n_frames=10, step=0.08):
    """A curved filament (radius 60 px, ~40 px long) sliding along its own
    circle by ~5 px per frame -- perfect follow-the-leader gliding."""
    return [_arc((100, 100), 60, 0.3 + k * step, 0.95 + k * step) for k in range(n_frames)]


def _swivelling(n_frames=10):
    """One end fixed at (100, 100), the filament swinging +-35 deg about it."""
    out = []
    for k in range(n_frames):
        ang = np.deg2rad(35 * np.sin(k * 0.9))
        r = np.arange(0, 30)
        out.append(np.rint(np.column_stack([100 + r * np.sin(ang), 100 + r * np.cos(ang)])).astype(int))
    return out


def test_gliding_along_a_curved_track_stays_on_track():
    p = _path(_gliding_on_arc())
    for lk in p.links:
        ok, d90 = TS.off_track(lk.filament1_contour, lk.filament2_contour)
        assert ok and d90 < 1.0            # pixel rounding only
    n, frac = TS.floppy_track(p, DX)
    assert n == 9 and frac == 0.0
    assert not TS.pinned_track(p, DX)[0]


def test_fast_index_matched_distance_is_not_used_for_this():
    """Sanity check of the motivation: index-matched point distance on the same
    gliding arc is several px, the curve distance ~0."""
    c = _gliding_on_arc()
    chord = np.mean(np.sqrt(((c[1].astype(float) - c[0]) ** 2).sum(1)))
    assert chord > 3.0
    assert TS.off_track(c[0], c[1])[1] < 1.0


def test_swivelling_filament_is_pinned_and_floppy():
    p = _path(_swivelling())
    pinned, rg_fixed, rg_free = TS.pinned_track(p, DX)
    assert pinned and rg_fixed < 50 and rg_free > 1000
    n, frac = TS.floppy_track(p, DX)
    assert n >= 3 and frac >= 0.5


def test_jump_to_another_filament_is_not_a_valid_step():
    a = _arc((100, 100), 60, 0.3, 0.95)
    far = a + np.array([25, 0])            # a different filament 25 px away
    assert TS.off_track(a, far)[0] is False
    # a track made only of such jumps has no valid steps -> no floppy verdict
    p = _path([a, far, a, far, a])
    assert np.isnan(TS.floppy_track(p, DX)[1])


def test_tip_identity_follows_fil_direction_when_contours_flip():
    """Contour order reversed every other frame (fil_direction = -1 on those
    links): tips must still be tracked, so a gliding filament isn't pinned."""
    conts = _gliding_on_arc()
    flipped = [c if k % 2 == 0 else c[::-1] for k, c in enumerate(conts)]
    dirs = [-1] * (len(conts) - 1)
    a, b = TS.tip_tracks(_path(flipped, dirs))
    # tip A is always the same physical end: it advances smoothly along the arc
    steps = np.sqrt((np.diff(a, axis=0) ** 2).sum(1))
    assert steps.max() < 8
    assert not TS.pinned_track(_path(flipped, dirs), DX)[0]


def test_attachment_metrics_percentages():
    paths = ([_path(_gliding_on_arc()) for _ in range(3)] + [_path(_swivelling())]
             + [_path(_gliding_on_arc(n_frames=3))])          # too short: ignored
    m = TS.attachment_metrics(paths, DX)
    assert m["n_pinned_tracks"] == 4 and m["pinned_pct"] == pytest.approx(25.0)
    assert m["n_floppy_tracks"] == 4 and m["floppy_pct"] == pytest.approx(25.0)
    assert len(m["pinned_paths"]) == 1 and m["pinned_paths"][0] is paths[3]
    stuck = [_path(_swivelling(), stuck=True)]
    m = TS.attachment_metrics(stuck, DX)
    assert m["pinned_pct"] == 100.0 and m["floppy_pct"] is None   # floppy: moving tracks only
