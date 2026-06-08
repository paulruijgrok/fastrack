"""Filament-detection (segmentation) strategy interface.

A ``Detector`` turns a loaded :class:`~fastrack.core.frame.Frame` into a set of
detected filaments.  The current entropy/watershed approach is one
implementation; a low-SNR detector (denoising front-end, alternative
thresholding, or an ML segmenter) is added by registering a new class under
``DETECTORS`` and selecting it by name in the settings -- ``Frame`` and the
pipeline do not change.
"""
from abc import ABC, abstractmethod

from ...registry import Registry

#: Registry of available detector implementations (populated on import).
DETECTORS = Registry("detector")


class Detector(ABC):
    """Detects filaments in a single frame."""

    @abstractmethod
    def detect(self, frame):
        """Populate ``frame.filaments`` and ``frame.filXYs`` from ``frame.img``."""
        raise NotImplementedError

    def assess_quality(self, frame):
        """Return ``"good"``/``"bad"`` for an acquisition-quality gate.

        Detectors that have no quality notion may use the default.
        """
        return "good"
