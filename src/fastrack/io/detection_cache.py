"""On-disk cache of FASTplus per-frame detections, for fast reruns.

The directional pipeline's dominant cost is per-frame detection (filaments via
the entropy/ridge detector, heads via LoG).  This cache persists those results
so that reruns — e.g. parameter sweeps on association / scoring / fitting — skip
detection (and even the movie load) entirely.

It deliberately **reuses the FASTrack `STORES` machinery** rather than inventing
a new format:

* ``layout="per-movie"`` (modern default) — one ``filXYs<tag>.npz`` per movie
  (the ``per-movie`` store), a zip with one member per frame;
* ``layout="per-frame"`` (legacy) — one ``filXYs<tag>NNN.npy`` per frame (the
  ``npy`` store), matching FASTrack's default ``cache_layout``.

Each frame's payload is the list of records produced by detection
(:class:`~fastrack.datamodel.FilamentRecord` for filaments,
:class:`~fastrack.polarity.spot.SpotRecord` for heads); both are plain dataclasses
and round-trip through ``np.save``/``np.load`` as an object array.

**Invalidation by content:** the ``cache_tag`` embeds a short hash of every
parameter that changes the detection output (detector + its params, channel,
registration, frame subsetting).  Change any of them and the tag changes, so the
old cache is simply not found and detection re-runs — no stale results.
"""
from __future__ import annotations

import hashlib
import json
from typing import List

from .stores import STORES


def params_hash(params: dict, length: int = 10) -> str:
    """Stable short hash of a parameter dict (order-independent)."""
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:length]


class DetectionCache:
    """Per-movie cache of one detection kind ("fil" or "head"), via ``STORES``.

    Addressed by ``directory`` (typically the per-movie output folder) and a
    ``cache_tag`` of the form ``_fp_<kind>_<paramshash>``.
    """

    def __init__(self, directory: str, kind: str, params: dict,
                 layout: str = "per-movie"):
        self.directory = directory
        self.kind = kind
        self.layout = layout
        backend = "per-movie" if layout == "per-movie" else "npy"
        self.store = STORES.create(backend)
        self.tag = "_fp_%s_%s" % (kind, params_hash(params))

    # -- queries -------------------------------------------------------- #
    def cached_frames(self) -> List[int]:
        return self.store.frames(self.directory, self.tag)

    def count(self) -> int:
        return len(self.cached_frames())

    def has_all(self, n_frames: int) -> bool:
        """True iff frames ``0 .. n_frames-1`` are all cached."""
        if n_frames <= 0:
            return False
        present = set(self.cached_frames())
        return all(i in present for i in range(n_frames))

    # -- io ------------------------------------------------------------- #
    def load(self) -> List[list]:
        """Return a list (per frame, ordered) of record lists."""
        return [list(self.store.read(self.directory, self.tag, i))
                for i in sorted(self.cached_frames())]

    def save(self, per_frame_records: List[list], force: bool = True) -> None:
        """Persist ``per_frame_records[i]`` under frame ``i``."""
        self.store.open_write(self.directory, self.tag, force=force)
        try:
            for i, recs in enumerate(per_frame_records):
                self.store.write(self.directory, self.tag, i, list(recs))
        finally:
            self.store.close()
