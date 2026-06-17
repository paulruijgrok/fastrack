"""Intermediate filament-record persistence interface.

A ``FilamentStore`` saves and restores the per-frame detected filaments (the
``filXYs`` intermediate between detection and tracking) for one movie folder.
Backends differ only in *where the bytes live*, not in their content, so the
choice is invisible to the rest of the pipeline:

* ``"npy"`` -- one ``filXYs<tag>NNN.npy`` file per frame (the original layout).
* ``"per-movie"`` -- a single ``filXYs<tag>.npz`` (zip) per movie, with one
  member per frame, written once and read member-wise.

A store is addressed by ``(directory, cache_tag, frame_no)``; ``cache_tag``
namespaces detectors (``""`` for entropy, ``"_ridge"`` etc.) so different
detectors never collide in the same folder.
"""
from abc import ABC, abstractmethod

from ...registry import Registry

#: Registry of available stores (populated on import).
STORES = Registry("filament store")


class FilamentStore(ABC):
    """Persists and restores a movie's per-frame detected filaments."""

    @abstractmethod
    def has(self, directory, cache_tag, frame_no):
        """Return True if ``frame_no`` is already cached for this movie."""
        raise NotImplementedError

    @abstractmethod
    def write(self, directory, cache_tag, frame_no, filXYs):
        """Persist one frame's ``filXYs``.

        For streaming backends, call :meth:`open_write` first and :meth:`close`
        when done; per-file backends may ignore that lifecycle.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, directory, cache_tag, frame_no):
        """Return the stored ``filXYs`` object array for ``frame_no``."""
        raise NotImplementedError

    @abstractmethod
    def frames(self, directory, cache_tag):
        """Return the sorted list of frame numbers currently stored."""
        raise NotImplementedError

    # --- optional streaming lifecycle (single-writer backends override) --- #
    def open_write(self, directory, cache_tag, force=False):
        """Begin a write session (no-op for per-file backends)."""

    def close(self):
        """End a write session (no-op for per-file backends)."""
