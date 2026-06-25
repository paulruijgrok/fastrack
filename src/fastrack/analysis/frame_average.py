"""Per-frame directional velocity averaging across movies (FASTplus, req. 3).

The base FASTrack analysis reports one velocity distribution per movie.  For
optogenetic / perturbation experiments we instead need the *signed* mean
velocity **as a function of frame (time)**, pooled over many movies of identical
format, so that a kinetic model can later be fitted to the time course.

:class:`FrameVelocityAggregator` accumulates per-step signed velocities keyed by
frame index across any number of movies and returns, for each frame, the mean,
SEM and count.  numpy only.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


class FrameVelocityAggregator:
    """Accumulate signed velocities by frame index across movies."""

    def __init__(self, dt_s: float = 1.0):
        self.dt_s = float(dt_s)
        self._by_frame: Dict[int, List[float]] = {}
        self.n_movies = 0

    # ------------------------------------------------------------------ #
    def add_path(self, directional_path) -> None:
        """Add one :class:`~fastrack.polarity.datamodel.DirectionalPath`.

        Velocity ``i`` (between frame ``frames[i]`` and the next) is filed under
        ``frames[i]``.
        """
        dp = directional_path
        for i, v in enumerate(dp.signed_velocity_nm_s):
            f = dp.frames[i] if i < len(dp.frames) else i
            self._by_frame.setdefault(int(f), []).append(float(v))

    def add_movie(self, directional_paths: Iterable) -> None:
        """Add all paths from one movie (counts as a single movie for n_movies)."""
        for dp in directional_paths:
            self.add_path(dp)
        self.n_movies += 1

    # ------------------------------------------------------------------ #
    def frames(self) -> List[int]:
        return sorted(self._by_frame)

    def frame_means(self, times_s: Optional[Dict[int, float]] = None) -> Dict[str, np.ndarray]:
        """Per-frame signed-velocity statistics.

        Returns arrays aligned by frame: ``frame``, ``time_s``, ``mean``,
        ``sem``, ``n`` (and ``std``).
        """
        frames = self.frames()
        mean, sem, std, n, t = [], [], [], [], []
        for f in frames:
            vals = np.asarray(self._by_frame[f], dtype=float)
            m = float(vals.mean()) if vals.size else np.nan
            s = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            mean.append(m); std.append(s)
            sem.append(s / np.sqrt(vals.size) if vals.size else np.nan)
            n.append(vals.size)
            t.append(times_s[f] if times_s and f in times_s else f * self.dt_s)
        return {
            "frame": np.array(frames, dtype=int),
            "time_s": np.array(t, dtype=float),
            "mean": np.array(mean, dtype=float),
            "sem": np.array(sem, dtype=float),
            "std": np.array(std, dtype=float),
            "n": np.array(n, dtype=int),
        }

    def frame_percentile_bands(self, pairs):
        """Per-frame central-percentile bands, aligned to :meth:`frames`.

        ``pairs`` is a list of (lower_pct, upper_pct), e.g. ``[(14, 86), (2, 98)]``.
        Returns a list of ``(lo_array, hi_array)`` (one per pair), each aligned
        frame-by-frame with :meth:`frame_means`.
        """
        frames = self.frames()
        bands = []
        for lo_p, hi_p in pairs:
            lo_arr, hi_arr = [], []
            for f in frames:
                vals = np.asarray(self._by_frame[f], dtype=float)
                if vals.size:
                    lo_arr.append(float(np.nanpercentile(vals, lo_p)))
                    hi_arr.append(float(np.nanpercentile(vals, hi_p)))
                else:
                    lo_arr.append(np.nan); hi_arr.append(np.nan)
            bands.append((np.array(lo_arr), np.array(hi_arr)))
        return bands

    def to_rows(self, times_s: Optional[Dict[int, float]] = None) -> List[dict]:
        st = self.frame_means(times_s)
        return [
            {"frame": int(st["frame"][i]), "time_s": float(st["time_s"][i]),
             "mean_signed_velocity_nm_s": float(st["mean"][i]),
             "sem_nm_s": float(st["sem"][i]), "n": int(st["n"][i])}
            for i in range(len(st["frame"]))
        ]
