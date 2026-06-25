"""Whole-sequence head tracking (FASTplus).

Python equivalent of TrackMate's ``LinearMotionLAP`` tracker: each track carries
a constant-velocity Kalman state ``[x, y, vx, vy]``; frame-to-frame linking is a
Linear Assignment Problem (Jaqaman 2008) on predicted-to-observed distances,
solved with the Hungarian method; unmatched tracks coast forward for up to
``max_frame_gap`` frames (gap closing) before retiring.

Unlike the filament :class:`~fastrack.core.tracking.base.Linker` (a *pairwise*,
mutate-the-frames operation), head tracking is naturally a *whole-movie* problem,
so it gets its own tiny registry, :data:`HEAD_TRACKERS`, rather than being forced
into the pairwise interface.

The Hungarian solver uses ``scipy.optimize.linear_sum_assignment`` when available
and otherwise an internal pure-python implementation.

TrackMate parameter mapping (FASTplus v0.1 defaults):
  initial_search_radius  first-step linking gate            (TM 20)
  kalman_search_radius   gate once a velocity is established (TM 15)
  max_frame_gap          frames a track may coast a gap      (TM 4)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ...registry import Registry

#: Registry of whole-sequence head trackers (populated on import).
HEAD_TRACKERS = Registry("head_tracker")

try:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _HAVE_SCIPY_LSA = True
except Exception:
    _HAVE_SCIPY_LSA = False


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #
def solve_assignment(cost: np.ndarray):
    """Return (row_idx, col_idx) minimizing total cost (scipy or internal)."""
    if _HAVE_SCIPY_LSA:
        return _scipy_lsa(cost)
    return _hungarian_numpy(cost)


def _hungarian_numpy(cost: np.ndarray):
    """Compact O(n^3) Hungarian (Munkres) on a square-padded copy of ``cost``."""
    cost = np.asarray(cost, dtype=np.float64)
    n_r, n_c = cost.shape
    n = max(n_r, n_c)
    big = (cost.max() * 10 + 1) if cost.size else 1.0
    C = np.full((n, n), big, dtype=np.float64)
    C[:n_r, :n_c] = cost

    u = np.zeros(n + 1); v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int); way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minv = np.full(n + 1, np.inf); used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True; i0 = p[j0]; delta = np.inf; j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
    rows, cols = [], []
    for j in range(1, n + 1):
        i = p[j]
        if i <= n_r and j <= n_c:
            rows.append(i - 1); cols.append(j - 1)
    order = np.argsort(rows)
    return np.array(rows)[order], np.array(cols)[order]


# --------------------------------------------------------------------------- #
# Track state
# --------------------------------------------------------------------------- #
@dataclass
class _Track:
    track_id: int
    state: np.ndarray     # [x, y, vx, vy]
    cov: np.ndarray
    last_frame: int
    spots: list = field(default_factory=list)
    misses: int = 0
    established: bool = False

    def predicted_xy(self):
        return self.state[0], self.state[1]


@HEAD_TRACKERS.register("kalman-lap")
class KalmanLAPTracker:
    """Constant-velocity Kalman + LAP linking + gap closing."""

    def __init__(self, initial_search_radius=20.0, kalman_search_radius=15.0,
                 max_frame_gap=4, process_noise=1.0, measurement_noise=1.0, dt=1.0):
        self.r0 = float(initial_search_radius)
        self.rk = float(kalman_search_radius)
        self.max_gap = int(max_frame_gap)
        self.dt = float(dt)
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        q = process_noise
        self.Q = q * np.array([[dt**3/3, 0, dt**2/2, 0],
                               [0, dt**3/3, 0, dt**2/2],
                               [dt**2/2, 0, dt, 0],
                               [0, dt**2/2, 0, dt]], dtype=float)
        self.R = measurement_noise * np.eye(2)
        self._next_id = 0

    def _predict(self, trk: _Track):
        trk.state = self.F @ trk.state
        trk.cov = self.F @ trk.cov @ self.F.T + self.Q

    def _update(self, trk: _Track, z: np.ndarray):
        S = self.H @ trk.cov @ self.H.T + self.R
        K = trk.cov @ self.H.T @ np.linalg.inv(S)
        trk.state = trk.state + K @ (z - self.H @ trk.state)
        trk.cov = (np.eye(4) - K @ self.H) @ trk.cov

    def _new_track(self, spot) -> _Track:
        trk = _Track(track_id=self._next_id,
                     state=np.array([spot.x, spot.y, 0.0, 0.0]),
                     cov=np.diag([1.0, 1.0, 100.0, 100.0]),
                     last_frame=spot.frame, spots=[spot])
        self._next_id += 1
        return trk

    def track(self, spots: List["object"]) -> List["object"]:
        """Link ``SpotRecord`` detections into tracks.

        Sets ``spot.track_id`` in place and returns the (same) spots that were
        assigned to a track, sorted by ``(track_id, frame)``.
        """
        by_frame: Dict[int, list] = {}
        for s in spots:
            by_frame.setdefault(s.frame, []).append(s)
        frames = sorted(by_frame)
        if not frames:
            return []

        active: List[_Track] = [self._new_track(s) for s in by_frame[frames[0]]]
        finished: List[_Track] = []

        for fi in range(1, len(frames)):
            f = frames[fi]
            dets = by_frame[f]
            for trk in active:
                self._predict(trk)

            assigned = set()
            if active and dets:
                INF = 1e6
                cost = np.full((len(active), len(dets)), INF)
                for ti, trk in enumerate(active):
                    px, py = trk.predicted_xy()
                    gate = self.rk if trk.established else self.r0
                    for di, d in enumerate(dets):
                        dist = np.hypot(d.x - px, d.y - py)
                        if dist <= gate:
                            cost[ti, di] = dist
                rows, cols = solve_assignment(cost)
                for ti, di in zip(rows, cols):
                    if cost[ti, di] >= INF:
                        continue
                    trk, d = active[ti], dets[di]
                    self._update(trk, np.array([d.x, d.y]))
                    trk.spots.append(d)
                    trk.last_frame = f
                    trk.misses = 0
                    if len(trk.spots) >= 2:
                        trk.established = True
                    assigned.add(di)

            still: List[_Track] = []
            for trk in active:
                if trk.last_frame == f:
                    still.append(trk)
                else:
                    trk.misses += 1
                    (still if trk.misses <= self.max_gap else finished).append(trk)
            active = still

            for di, d in enumerate(dets):
                if di not in assigned:
                    active.append(self._new_track(d))

        finished.extend(active)

        out = []
        for trk in finished:
            for s in trk.spots:
                s.track_id = trk.track_id
                out.append(s)
        out.sort(key=lambda s: (s.track_id, s.frame))
        return out

    def backend(self) -> str:
        return "scipy-LAP=%s" % _HAVE_SCIPY_LSA
