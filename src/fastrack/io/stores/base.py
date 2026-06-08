"""Intermediate filament-record persistence interface.

A ``FilamentStore`` saves and loads the per-frame detected filaments (the
intermediate result between detection and tracking).  The current backend keeps
one ``.npy`` blob per frame, matching the original ``filXYs`` files; future
backends (Parquet, HDF5, SQLite) can store a queryable cross-frame table by
registering under ``STORES`` and selecting by name.
"""
from abc import ABC, abstractmethod

from ...registry import Registry

#: Registry of available stores (populated on import).
STORES = Registry("filament store")


class FilamentStore(ABC):
    """Persists and restores a frame's detected filaments."""

    @abstractmethod
    def save_frame(self, frame):
        """Persist ``frame``'s detected filaments (``frame.filXYs``)."""
        raise NotImplementedError

    @abstractmethod
    def load_frame(self, frame):
        """Restore ``frame.filXYs`` (and filaments) from storage."""
        raise NotImplementedError
