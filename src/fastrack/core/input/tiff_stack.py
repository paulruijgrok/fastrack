"""Multi-page TIFF stack source -- read a movie straight from one file.

Pages are read lazily with ``tifffile`` (each process opens the file and pulls
only the frames it needs, so workers stay memory-light); if ``tifffile`` isn't
available it falls back to an eager ``cv2.imreadmulti``.  Pixels go through
:func:`to_pipeline_image`, so a stack yields the same images as its pre-split
micro-manager frames.

Per-frame timing is read from embedded ImageJ / micro-manager metadata when
present (``ElapsedTime-ms``); otherwise ``elapsed_times()`` returns ``None`` and
the caller falls back to a uniform ``frame_rate``.
"""
import os
import re

import numpy as np

from .base import FrameSource, register_source, to_pipeline_image


@register_source("tiff_stack")
class TiffStackSource(FrameSource):
    def __init__(self, path, frame_rate=1.0):
        self.path = os.path.abspath(path)
        self.frame_rate = frame_rate
        self._tf = None          # tifffile.TiffFile handle (lazy)
        self._pages = None       # cv2 fallback: list of page arrays
        self._count = None

    # ----- lazy open ----------------------------------------------------- #
    def _open(self):
        if self._count is not None:
            return
        try:
            import tifffile
            self._tf = tifffile.TiffFile(self.path)
            self._count = len(self._tf.pages)
        except ImportError:
            import cv2
            ok, pages = cv2.imreadmulti(self.path, flags=cv2.IMREAD_UNCHANGED)
            self._pages = list(pages) if ok else []
            self._count = len(self._pages)

    # ----- FrameSource API ----------------------------------------------- #
    @property
    def count(self):
        self._open()
        return self._count

    def read(self, frame_no):
        self._open()
        if self._tf is not None:
            page = self._tf.pages[int(frame_no)].asarray()
        else:
            page = self._pages[int(frame_no)]
        return to_pipeline_image(page)

    def elapsed_times(self):
        """Best-effort per-frame seconds from embedded metadata, else None."""
        self._open()
        if self._tf is None:
            return None
        times = []
        try:
            # micro-manager stacks store per-frame JSON in each page's
            # ImageDescription; ImageJ stacks store a block in the file.
            for page in self._tf.pages:
                desc = page.tags.get("ImageDescription")
                val = desc.value if desc is not None else ""
                m = re.search(r'"?ElapsedTime-ms"?\s*[:=]\s*([0-9.]+)', str(val))
                if m:
                    times.append(float(m.group(1)))
        except Exception:
            return None
        if len(times) != self._count or not times:
            return None
        return np.sort(0.001 * np.array(times))

    @property
    def identity(self):
        return self.path

    def descriptor(self):
        return {"kind": "tiff_stack", "path": self.path, "frame_rate": self.frame_rate}

    @classmethod
    def from_descriptor(cls, d):
        return cls(d["path"], frame_rate=d.get("frame_rate", 1.0))
