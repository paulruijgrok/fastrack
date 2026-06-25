"""Frame-to-frame linking strategies.

Importing this package registers the built-in linkers under ``LINKERS``.
"""
from . import greedy  # noqa: F401  (registers GreedyLinker)
from . import head_tracker  # noqa: F401  (registers KalmanLAPTracker under HEAD_TRACKERS)
from .base import LINKERS, Linker
from .head_tracker import HEAD_TRACKERS

__all__ = ["LINKERS", "Linker", "HEAD_TRACKERS"]
