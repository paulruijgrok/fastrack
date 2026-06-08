"""Per-frame ``.npy`` filament store -- the original ``filXYs`` format.

This wraps the existing ``Frame.save_filXYs`` / ``Frame.read_filXYs`` behaviour
so the storage format is selectable without changing the core path.  It is the
default and preserves the exact on-disk files the rest of the pipeline expects.
"""
from .base import STORES, FilamentStore


@STORES.register("npy")
class NpyFilamentStore(FilamentStore):
    def save_frame(self, frame):
        frame.save_filXYs()

    def load_frame(self, frame):
        frame.read_filXYs()
        frame.filXY2filaments()
