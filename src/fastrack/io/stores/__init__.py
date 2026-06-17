"""Intermediate filament-record stores."""
from . import npy_store  # noqa: F401  (registers NpyFilamentStore as "npy")
from . import npz_store  # noqa: F401  (registers NpzFilamentStore as "per-movie")
from .base import STORES, FilamentStore

__all__ = ["STORES", "FilamentStore"]
