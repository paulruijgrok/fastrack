"""Associate head tracks with detected filaments and assign a polar end (FASTplus).

In head-centric analysis the filaments are detected per frame but *not* tracked
frame-to-frame (filament density is high and crossings are frequent).  Identity
across time comes from the tracked *heads*; this module attaches, in each frame,
each detected filament to the head(s) sitting on it and records which filament
tip the head marks.

The geometric primitives only need a filament's two contour tips and centre of
mass, so this works against either a live ``Filament`` (``.contour``, ``.cm``)
or a :class:`~fastrack.datamodel.FilamentRecord`.  Depends only on numpy.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .datamodel import PolarFilament
from .spot import SpotRecord


def _contour_of(filament) -> Optional[np.ndarray]:
    c = getattr(filament, "contour", None)
    if c is None:
        return None
    c = np.asarray(c, dtype=float)
    return c if c.ndim == 2 and len(c) >= 2 else None


def _tips(contour: np.ndarray):
    """Return the two filament tips as (x, y).  Contour is stored (row, col)."""
    p0 = contour[0][::-1]    # (col, row) -> (x, y)
    p1 = contour[-1][::-1]
    return p0, p1


def _point_to_contour_distance(pt_xy: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Distance from point (x, y) to every contour vertex (stored row, col)."""
    pts_xy = contour[:, ::-1]
    return np.sqrt(np.sum((pts_xy - pt_xy) ** 2, axis=1))


class HeadFilamentAssociator:
    """Attach tracked heads to per-frame filaments and tag the marked tip."""

    def __init__(self, max_end_distance_px: float = 6.0, end_fraction: float = 0.15):
        #: how close a head must be to a tip to count as "on" the filament.
        self.max_end_distance_px = float(max_end_distance_px)
        #: fraction of contour length from each tip that still counts as an "end".
        self.end_fraction = float(end_fraction)

    # ------------------------------------------------------------------ #
    def _region_of(self, contour: np.ndarray, spot: SpotRecord) -> Optional[str]:
        """Where on the filament the head sits: 'tip0', 'tip1', 'middle', or None.

        None means the head is not on this filament (too far from the skeleton).
        """
        d = _point_to_contour_distance(spot.xy, contour)
        i = int(np.argmin(d))
        if d[i] > self.max_end_distance_px:
            return None
        n = len(contour)
        end_n = max(1, int(round(self.end_fraction * n)))
        if i < end_n:
            return "tip0"
        if i >= n - end_n:
            return "tip1"
        return "middle"

    def associate_frame(
        self,
        filaments: Sequence,
        heads: Sequence[SpotRecord],
        frame_no: int,
    ) -> List[PolarFilament]:
        """Associate the heads of one frame with that frame's filaments.

        ``heads`` should be the (tracked) head spots in this frame; each carries a
        ``track_id``.  Returns one :class:`PolarFilament` per filament, with the
        marked tip set when exactly one head sits on exactly one tip (final
        inclusion is decided later by :class:`PolarityClassifier`).
        """
        out: List[PolarFilament] = []
        for label, fil in enumerate(filaments):
            contour = _contour_of(fil)
            if contour is None:
                continue
            tip0, tip1 = _tips(contour)
            cm = getattr(fil, "cm", None)
            cm = (np.asarray(cm, float)[::-1] if cm is not None and len(np.atleast_1d(cm)) == 2
                  else 0.5 * (tip0 + tip1))
            length = float(getattr(fil, "fil_length", 0.0)
                           or getattr(fil, "length", 0.0) or 0.0)

            regions = {"tip0": [], "tip1": [], "middle": []}
            for h in heads:
                reg = self._region_of(contour, h)
                if reg is not None:
                    regions[reg].append(h)

            pf = PolarFilament(frame=frame_no, filament_label=label,
                               cm=cm, length=length)
            pf.head_ids = [h.track_id for h in
                           (regions["tip0"] + regions["tip1"] + regions["middle"])
                           if h.track_id is not None]
            pf._regions = regions          # consumed by PolarityClassifier
            pf._tips = (tip0, tip1)
            out.append(pf)
        return out
