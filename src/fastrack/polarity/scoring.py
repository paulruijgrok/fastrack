"""Signed (polarity-aware) velocity scoring (FASTplus).

SIGN CONVENTION
---------------
The velocity is **positive when the (invisible) motors driving the gliding are
stroking toward the (+)-end (barbed end) of the filament**, and negative when
they stroke toward the (-)-end (pointed end).  Because surface motors propel a
filament (-)-end-first, "motors stroking toward (+)" means the (+)-end of the
filament trails the motion.  Expressed in terms of the fluorescent end-label
("head") relative to the direction of motion:

    label marks the (+)-end (e.g. gelsolin on actin barbed ends):
        head LAGGING (at the back of the moving filament)  -> POSITIVE
        head LEADING (at the front)                        -> NEGATIVE
    label marks the (-)-end:
        head LEADING                                       -> POSITIVE
        head LAGGING                                       -> NEGATIVE

Which end the label marks is set by ``head_marks_end`` ("plus" by default).

IMPLEMENTATION
--------------
Each frame-to-frame displacement is projected onto the filament's polar axis
(the unit vector pointing from the opposite tip toward the *labelled* tip, i.e.
toward the head).  A projection > 0 means the head is leading.  The result is
then multiplied by a sign factor so that the output follows the convention
above: ``-1`` when the label marks the (+)-end, ``+1`` when it marks the (-)-end.

* :meth:`DirectionalScorer.score_head_track` -- head-centric: per-frame polar
  axis from the association with an unambiguously labelled filament.
* :meth:`DirectionalScorer.score_filament_path` -- filament-centric: a tracked
  filament path with a single (track-constant) polar axis.

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
                 stuck_velocity_nm_s: float = 80.0, head_marks_end: str = "plus"):
        self.pixel_size_nm = float(pixel_size_nm)
        self.dt_s = float(dt_s)
        self.stuck_velocity_nm_s = float(stuck_velocity_nm_s)
        if head_marks_end not in ("plus", "minus"):
            raise ValueError("head_marks_end must be 'plus' or 'minus'")
        self.head_marks_end = head_marks_end
        #: polar axis points toward the labelled tip (head). Projection > 0 means
        #: head leading. For a (+)-end label, head-leading is NEGATIVE, so flip.
        self.sign = -1.0 if head_marks_end == "plus" else 1.0

    # ------------------------------------------------------------------ #
    def _signed_step(self, disp_px: np.ndarray, axis_unit: np.ndarray, dt: float) -> float:
        """Signed velocity (nm/s); sign follows the module's convention."""
        proj_px = float(np.dot(disp_px, axis_unit))
        if dt <= 0:
            return 0.0
        return self.sign * proj_px * self.pixel_size_nm / dt

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
