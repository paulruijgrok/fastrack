"""Frame-to-frame linking strategies.

Importing this package registers the built-in linkers under ``LINKERS``.
"""
from . import greedy  # noqa: F401  (registers GreedyLinker)
from .base import LINKERS, Linker

__all__ = ["LINKERS", "Linker"]
