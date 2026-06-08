"""In-memory data model for filament records across frames.

The image-processing classes in :mod:`fastrack.core` (``Frame``, ``Island``,
``Filament``) carry heavy working state -- reduced images, skeletons, links to
neighbouring objects.  This module provides a light, serializable view of the
*results*: one :class:`FilamentRecord` per detected filament, collected into a
:class:`FilamentTable` that spans all frames of a movie.

Persistence backends (:mod:`fastrack.io.stores`) and exporters
(:mod:`fastrack.io.export`) operate on this structure rather than on the live
algorithm objects, which decouples *what is computed* from *how it is stored or
exported*.  The model deliberately depends only on numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class FilamentRecord:
    """A single detected filament in a single frame.

    Geometry is stored as the skeleton ``contour`` (an ``(N, 2)`` integer array
    of pixel coordinates); scalar measurements mirror the quantities the
    detector computes.  ``path_id`` and the link fields are populated once
    frame-to-frame tracking has run.
    """

    frame: int
    label: int
    contour: np.ndarray                      # (N, 2) int pixel coordinates
    length: float = 0.0
    density: float = 0.0
    width: float = 0.0
    area: float = 0.0
    end2end: float = 0.0
    midpoint: Optional[np.ndarray] = None    # (2,) coordinate or None
    cm: Optional[np.ndarray] = None          # centre of mass, (2,) or None

    # Tracking links (filled in after linking); identify partners by (frame,label).
    path_id: Optional[int] = None
    forward_link: Optional[Tuple[int, int]] = None
    reverse_link: Optional[Tuple[int, int]] = None

    @classmethod
    def from_filament(cls, fil) -> "FilamentRecord":
        """Build a record from a live :class:`fastrack.core.filament.Filament`."""
        contour = np.asarray(getattr(fil, "contour", []), dtype=int)
        midpoint = getattr(fil, "midpoint", None)
        cm = getattr(fil, "cm", None)
        return cls(
            frame=int(getattr(fil, "frame_no", 0)),
            label=int(getattr(fil, "label", 0)),
            contour=contour,
            length=float(getattr(fil, "fil_length", 0.0) or 0.0),
            density=float(getattr(fil, "fil_density", 0.0) or 0.0),
            width=float(getattr(fil, "fil_width", 0.0) or 0.0),
            area=float(getattr(fil, "fil_area", 0.0) or 0.0),
            end2end=float(getattr(fil, "end2end", 0.0) or 0.0),
            midpoint=(np.asarray(midpoint) if midpoint is not None and len(np.atleast_1d(midpoint)) else None),
            cm=(np.asarray(cm) if cm is not None and len(np.atleast_1d(cm)) else None),
        )

    def to_row(self) -> Dict[str, object]:
        """Flat, export-friendly dict (scalars only; contour summarized)."""
        mid = self.midpoint if self.midpoint is not None else (np.nan, np.nan)
        return {
            "frame": self.frame,
            "label": self.label,
            "n_points": int(len(self.contour)),
            "length": self.length,
            "density": self.density,
            "width": self.width,
            "area": self.area,
            "end2end": self.end2end,
            "midpoint_x": float(mid[0]),
            "midpoint_y": float(mid[1]),
            "path_id": self.path_id if self.path_id is not None else -1,
        }


class FilamentTable:
    """An ordered collection of :class:`FilamentRecord` spanning many frames."""

    def __init__(self, records: Optional[List[FilamentRecord]] = None):
        self.records: List[FilamentRecord] = list(records) if records else []

    # -- construction --------------------------------------------------- #
    def add(self, record: FilamentRecord) -> None:
        self.records.append(record)

    @classmethod
    def from_frames(cls, frames) -> "FilamentTable":
        """Collect every filament from an iterable of live ``Frame`` objects."""
        table = cls()
        for frame in frames:
            for fil in getattr(frame, "filaments", []):
                table.add(FilamentRecord.from_filament(fil))
        return table

    # -- access --------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[FilamentRecord]:
        return iter(self.records)

    def frames(self) -> List[int]:
        return sorted({r.frame for r in self.records})

    def by_frame(self, frame: int) -> List[FilamentRecord]:
        return [r for r in self.records if r.frame == frame]

    def to_rows(self) -> List[Dict[str, object]]:
        """Tidy, one-row-per-filament representation for CSV/Parquet export."""
        return [r.to_row() for r in self.records]
