"""Directional result records (FASTplus).

These sit on top of the existing filament data model and the head
:mod:`~fastrack.polarity.spot` records:

* :class:`PolarFilament` -- a detected filament in one frame, together with the
  head(s) found on it and the resulting polarity classification (which tip, if
  any, is the plus-end).
* :class:`DirectionalPath` -- a tracked object (a head track, or a filament path)
  carrying a *signed* velocity time series, where the sign encodes plus-end
  (positive) vs minus-end (negative) directed motion.

Depends only on numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Polarity classification labels (see PolarityClassifier).
PLUS_END = "plus_end"      # exactly one head, on one tip -> unambiguous, included
BOTH_ENDS = "both_ends"    # heads on both tips -> excluded
MIDDLE = "middle"          # head(s) away from the tips -> excluded
NONE = "none"              # no head -> excluded

INCLUDED = {PLUS_END}


@dataclass
class PolarFilament:
    """A filament (one frame) with its associated head(s) and polarity call."""
    frame: int
    filament_label: int
    # filament geometry (image px); ends are the two contour tips
    plus_end_xy: Optional[np.ndarray] = None     # the marked (head) tip
    minus_end_xy: Optional[np.ndarray] = None     # the opposite tip
    cm: Optional[np.ndarray] = None
    length: float = 0.0
    head_ids: List[int] = field(default_factory=list)   # associated head track ids
    classification: str = NONE

    @property
    def is_unambiguous(self) -> bool:
        return self.classification in INCLUDED

    @property
    def polarity_vector(self) -> Optional[np.ndarray]:
        """Unit vector minus-end -> plus-end (the filament's intrinsic axis)."""
        if self.plus_end_xy is None or self.minus_end_xy is None:
            return None
        v = np.asarray(self.plus_end_xy, float) - np.asarray(self.minus_end_xy, float)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    def to_row(self) -> Dict[str, object]:
        pe = self.plus_end_xy if self.plus_end_xy is not None else (np.nan, np.nan)
        return {
            "frame": self.frame,
            "filament_label": self.filament_label,
            "classification": self.classification,
            "n_heads": len(self.head_ids),
            "plus_end_x": float(pe[0]), "plus_end_y": float(pe[1]),
            "length": self.length,
        }


@dataclass
class DirectionalPath:
    """A tracked object with a signed (polarity-aware) velocity series."""
    path_id: int
    source: str = "head"           # "head" or "filament"
    frames: List[int] = field(default_factory=list)
    times_s: List[float] = field(default_factory=list)
    positions: List[np.ndarray] = field(default_factory=list)     # (x, y) px
    signed_velocity_nm_s: List[float] = field(default_factory=list)  # per-step
    plus_end_directed: Optional[bool] = None

    def mean_signed_velocity(self) -> float:
        return float(np.mean(self.signed_velocity_nm_s)) if self.signed_velocity_nm_s else 0.0

    def n_steps(self) -> int:
        return len(self.signed_velocity_nm_s)

    def to_rows(self) -> List[Dict[str, object]]:
        rows = []
        for i, v in enumerate(self.signed_velocity_nm_s):
            # velocity i is between frame i and i+1
            rows.append({
                "path_id": self.path_id,
                "source": self.source,
                "frame": self.frames[i] if i < len(self.frames) else -1,
                "time_s": self.times_s[i] if i < len(self.times_s) else np.nan,
                "signed_velocity_nm_s": v,
            })
        return rows
