"""Signed (polarity-aware) velocity scoring (FASTplus).

Given a tracked object and a fixed polar axis (minus-end -> plus-end unit
vector), each frame-to-frame displacement is projected onto that axis: motion
toward the plus-end is positive, motion toward the minus-end is negative.  This
turns the unsigned speeds of the base FASTrack analysis into *directional*
velocities.

* :meth:`DirectionalScorer.score_head_track` -- head-centric: the head *is* the
  marked (plus) end, and the polar axis is taken from the per-frame association
  with an unambiguously labelled filament.
* :meth:`DirectionalScorer.score_filament_path` -- filament-centric: a normal
  filament path, with polarity attached from the head sitting on one tip.

numpy only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .datamodel import DirectionalPath
from .spot import SpotRecord


class DirectionalScorer:
    """Project displacements onto the filament polar axis to get signed velocity."""

    def __init__(self, pixel_size_nm: float = 80.65, dt_s: float = 1.0,
                 stuck_velocity_nm_s: float = 80.0):
        self.pixel_size_nm = float(pixel_size_nm)
        self.dt_s = float(dt_s)
        self.stuck_velocity_nm_s = float(stuck_velocity_nm_s)

    # ------------------------------------------------------------------ #
    def _signed_step(self, disp_px: np.ndarray, axis_unit: np.ndarray, dt: float) -> float:
        """Signed velocity (nm/s) = projection of displacement onto polar axis."""
        proj_px = float(np.dot(disp_px, axis_unit))
        return proj_px * self.pixel_size_nm / dt if dt > 0 else 0.0

    def _times(self, frames: Sequence[int], elapsed_times: Optional[Sequence[float]]):
        if elapsed_times is not None and len(elapsed_times):
            return [float(elapsed_times[f]) for f in frames]
        return [f * self.dt_s for f in frames]

    # ------------------------------------------------------------------ #
    def score_head_track(
        self,
        path_id: int,
        spots: Sequence[SpotRecord],
        axis_by_frame: Dict[int, np.ndarray],
        elapsed_times: Optional[Sequence[float]] = None,
    ) -> DirectionalPath:
        """Score a head track using a per-frame polar-axis (minus->plus unit vec).

        ``axis_by_frame`` maps frame number to the unit polarity vector of the
        filament the head was associated with in that frame.  Steps lacking an
        axis (head not on an unambiguous filament that frame) are skipped.
        """
        spots = sorted(spots, key=lambda s: s.frame)
        frames = [s.frame for s in spots]
        times = self._times(frames, elapsed_times)
        dp = DirectionalPath(path_id=path_id, source="head",
                             frames=frames, times_s=times,
                             positions=[s.xy for s in spots])
        for i in range(len(spots) - 1):
            f0 = spots[i].frame
            axis = axis_by_frame.get(f0)
            if axis is None:
                continue
            disp = spots[i + 1].xy - spots[i].xy
            dt = times[i + 1] - times[i]
            dp.signed_velocity_nm_s.append(self._signed_step(disp, axis, dt))
        dp.plus_end_directed = (dp.mean_signed_velocity() >= 0)
        return dp

    def score_filament_path(
        self,
        path_id: int,
        positions_px: Sequence[np.ndarray],
        frames: Sequence[int],
        axis_unit: np.ndarray,
        elapsed_times: Optional[Sequence[float]] = None,
    ) -> DirectionalPath:
        """Score a filament path with a single (track-constant) polar axis."""
        times = self._times(list(frames), elapsed_times)
        dp = DirectionalPath(path_id=path_id, source="filament",
                             frames=list(frames), times_s=times,
                             positions=[np.asarray(p, float) for p in positions_px])
        axis_unit = np.asarray(axis_unit, float)
        for i in range(len(positions_px) - 1):
            disp = np.asarray(positions_px[i + 1], float) - np.asarray(positions_px[i], float)
            dt = times[i + 1] - times[i]
            dp.signed_velocity_nm_s.append(self._signed_step(disp, axis_unit, dt))
        dp.plus_end_directed = (dp.mean_signed_velocity() >= 0)
        return dp

    def is_stuck(self, dp: DirectionalPath) -> bool:
        return abs(dp.mean_signed_velocity()) < self.stuck_velocity_nm_s
