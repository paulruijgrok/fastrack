"""Light, serializable records for head detections and head tracks (FASTplus).

Mirrors the role of :mod:`fastrack.datamodel` (``FilamentRecord`` /
``FilamentTable``) for the polarity-label "heads": a :class:`SpotRecord` is one
detected head in one frame, and a :class:`SpotTable` collects them across a
movie and groups them into tracks once a head tracker has run.  Depends only on
numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np


@dataclass
class SpotRecord:
    """A single detected head in a single frame (image coordinates, pixels)."""
    frame: int
    x: float
    y: float
    quality: float = 0.0          # LoG response at the maximum (TrackMate "quality")
    radius: float = 0.0           # estimated radius used for detection (px)
    track_id: Optional[int] = None  # filled in by the head tracker

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def to_row(self) -> Dict[str, object]:
        return {
            "frame": self.frame,
            "track_id": self.track_id if self.track_id is not None else -1,
            "x": self.x, "y": self.y,
            "quality": self.quality, "radius": self.radius,
        }


class SpotTable:
    """An ordered collection of :class:`SpotRecord` spanning many frames."""

    def __init__(self, records: Optional[List[SpotRecord]] = None):
        self.records: List[SpotRecord] = list(records) if records else []

    # -- construction --------------------------------------------------- #
    def add(self, record: SpotRecord) -> None:
        self.records.append(record)

    def extend(self, records) -> None:
        self.records.extend(records)

    # -- access --------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[SpotRecord]:
        return iter(self.records)

    def frames(self) -> List[int]:
        return sorted({r.frame for r in self.records})

    def by_frame(self, frame: int) -> List[SpotRecord]:
        return [r for r in self.records if r.frame == frame]

    def by_track(self, track_id: int) -> List[SpotRecord]:
        return sorted((r for r in self.records if r.track_id == track_id),
                      key=lambda r: r.frame)

    def track_ids(self) -> List[int]:
        return sorted({r.track_id for r in self.records if r.track_id is not None})

    def tracks(self) -> Dict[int, List[SpotRecord]]:
        """Map track_id -> frame-ordered list of spots (tracked spots only)."""
        out: Dict[int, List[SpotRecord]] = {}
        for r in self.records:
            if r.track_id is not None:
                out.setdefault(r.track_id, []).append(r)
        for tid in out:
            out[tid].sort(key=lambda r: r.frame)
        return out

    def to_rows(self) -> List[Dict[str, object]]:
        return [r.to_row() for r in self.records]
