"""Filament-detection strategies.

Importing this package registers the built-in detectors under ``DETECTORS``.
"""
from . import entropy  # noqa: F401  (registers EntropyWatershedDetector)
from .base import DETECTORS, Detector

__all__ = ["DETECTORS", "Detector"]
