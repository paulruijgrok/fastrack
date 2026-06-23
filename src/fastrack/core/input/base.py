"""Frame-source seam: how a movie's frames are enumerated and read.

A ``FrameSource`` abstracts *where the pixels come from* so the rest of the
pipeline doesn't care whether a movie is a folder of single-page micro-manager
TIFFs (the original layout) or a single multi-page TIFF stack -- and so new
formats (OME-TIFF, ND2, ...) are added by registering another source.

Each source must be reconstructable in a child process from a small picklable
``descriptor()`` (multiprocessing workers rebuild the source from it), and
exposes a stable ``identity`` used to derive output/cache names.

Pixel convention -- match the original pipeline exactly.  ``Frame.read_frame``
historically did ``cv2.imread(path, IMREAD_GRAYSCALE)`` (which downconverts a
16-bit TIFF to 8-bit by ``>> 8``) followed by ``skimage.img_as_uint`` (which
scales 8-bit -> 16-bit by ``* 257``).  :func:`to_pipeline_image` reproduces that
quantization so a stack and its pre-split frames yield identical results (and the
golden master is preserved).  ``* 257`` equals ``img_as_uint`` for uint8 input,
so this carries no scikit-image import.
"""
from abc import ABC, abstractmethod

import numpy as np

#: kind -> FrameSource subclass (populated by @register_source).
_SOURCE_CLASSES = {}


def register_source(kind):
    def _add(cls):
        cls.kind = kind
        _SOURCE_CLASSES[kind] = cls
        return cls
    return _add


def to_pipeline_image(arr):
    """Quantize a raw page to the pipeline's image, matching the legacy read.

    16-bit -> 8-bit via ``>> 8`` (as cv2 IMREAD_GRAYSCALE does), then 8-bit ->
    16-bit via ``* 257`` (as img_as_uint does).  8-bit input skips the first step.
    """
    a = np.asarray(arr)
    if a.ndim == 3:                       # grayscale data stored with a channel axis
        a = a[..., 0]
    if a.dtype == np.uint16:
        a8 = (a >> 8).astype(np.uint8)
    elif a.dtype == np.uint8:
        a8 = a
    else:                                 # other depths: clip into 8-bit range
        a8 = np.clip(a, 0, 255).astype(np.uint8)
    return (a8.astype(np.uint32) * 257).astype(np.uint16)


class FrameSource(ABC):
    """Enumerates and reads the frames of one movie."""

    kind = "?"

    @property
    @abstractmethod
    def count(self):
        """Number of frames in the movie."""

    @abstractmethod
    def read(self, frame_no):
        """Return frame ``frame_no`` as an ``(H, W)`` uint16 image."""

    def frame_numbers(self):
        """Frame indices to process (default: ``0 .. count-1``)."""
        return list(range(self.count))

    def elapsed_times(self):
        """Per-frame acquisition times in seconds, or ``None`` if unknown.

        When ``None``, the caller falls back to a uniform ``frame_rate``.
        """
        return None

    @property
    @abstractmethod
    def identity(self):
        """A stable path-like string identifying this movie (for output/cache names)."""

    @abstractmethod
    def descriptor(self):
        """A small picklable dict (including ``"kind"``) to rebuild this source."""

    @classmethod
    def from_descriptor(cls, d):  # pragma: no cover - overridden
        raise NotImplementedError


def open_source(descriptor):
    """Rebuild a :class:`FrameSource` from a :meth:`FrameSource.descriptor` dict."""
    d = dict(descriptor)
    return _SOURCE_CLASSES[d["kind"]].from_descriptor(d)


def open_movie(path, input_format="auto", frame_rate=1.0):
    """Return a :class:`FrameSource` for ``path``, or ``None`` if unrecognized.

    ``input_format``: ``"auto"`` (a multi-page ``.tif`` file -> stack; a folder of
    ``img_******NNN__000.tif`` -> micro-manager frames), or force ``"stack"`` /
    ``"frames"``.
    """
    import os
    from .mm_dir import MicroManagerDirSource
    from .tiff_stack import TiffStackSource

    fmt = (input_format or "auto").lower()
    if fmt == "stack":
        return TiffStackSource(path, frame_rate=frame_rate)
    if fmt == "frames":
        return MicroManagerDirSource(path, frame_rate=frame_rate)

    # auto
    if os.path.isfile(path) and path.lower().endswith((".tif", ".tiff")):
        return TiffStackSource(path, frame_rate=frame_rate)
    if os.path.isdir(path) and MicroManagerDirSource.looks_like(path):
        return MicroManagerDirSource(path, frame_rate=frame_rate)
    return None
