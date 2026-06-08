"""Entropy/watershed filament detector -- the original FAST segmentation.

This orchestrates the per-frame primitives that live on
:class:`~fastrack.core.frame.Frame` (low-pass, local-contrast thresholding,
island labelling, Otsu/watershed decomposition, skeletonization).  The exact
sequence and numerics are unchanged from the original
``Motility.read_frame`` -> ``Frame`` pipeline; this class just makes that
sequence the swappable unit, and carries the ``fast_rank`` / ``morph_contrast``
performance options that configure it.
"""
from .base import DETECTORS, Detector


@DETECTORS.register("entropy")
class EntropyWatershedDetector(Detector):
    def __init__(self, fast_rank=True, morph_contrast=False):
        self.fast_rank = fast_rank
        self.morph_contrast = morph_contrast

    def _configure(self, frame):
        frame.fast_rank = self.fast_rank
        frame.morph_contrast = self.morph_contrast

    def detect(self, frame):
        """Run the entropy/watershed detection sequence on ``frame``.

        Equivalent to the original ``low_pass_filter`` -> ``entropy_clusters``
        -> ``filter_islands`` -> ``skeletonize_islands`` -> ``filaments2filXYs``.
        """
        self._configure(frame)
        frame.low_pass_filter()
        frame.entropy_clusters()
        frame.filter_islands()
        frame.skeletonize_islands()
        frame.filaments2filXYs()

    def assess_quality(self, frame):
        self._configure(frame)
        return frame.check_picture_quality()
