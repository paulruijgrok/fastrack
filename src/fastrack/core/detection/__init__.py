"""Filament-detection strategies.

Importing this package registers the built-in detectors under ``DETECTORS``.
"""
from . import entropy  # noqa: F401  (registers EntropyWatershedDetector)
from . import ridge  # noqa: F401  (registers RidgeLineDetector; ridge-detector imported lazily)
from . import heads  # noqa: F401  (registers HeadDetector; FASTplus point/head detector)
from .base import DETECTORS, Detector

__all__ = ["DETECTORS", "Detector"]
