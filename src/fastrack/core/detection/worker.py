"""Import-light per-frame detection for process-pool workers.

Worker processes on macOS / Windows are *spawned*: each one starts a fresh
interpreter and imports whatever module holds the task function.  Importing it
from ``fastrack.pipelines`` drags in the whole gliding pipeline (matplotlib,
scikit-image, the plotting code, ...), which cost ~3 s per worker per run on a
Rosetta-emulated Mac -- more than the detection itself for a short preview.

This module needs only numpy, scipy.ndimage (already imported by the ridge
detectors) and the detector registry.  The detectors call
``frame.filXY2filaments()``; workers don't need the Filament objects (the
caller rebuilds them from ``filXYs``), so a minimal stand-in frame is used
instead of :class:`fastrack.core.frame.Frame`.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .base import DETECTORS


class _LightFrame:
    """The attributes the ridge detectors read / write on a frame."""

    def __init__(self, img, frame_no, noise_sd):
        self.img = img
        self.width, self.height = img.shape
        self.frame_no = frame_no
        self.noise_sd = noise_sd
        self.filXYs = []

    def filXY2filaments(self):  # filaments are rebuilt by the caller
        pass


def _contour_signal(img_f, contour):
    """Median of the (1-px smoothed) image along an integer [row, col] contour."""
    r = np.clip(contour[:, 0], 0, img_f.shape[0] - 1)
    c = np.clip(contour[:, 1], 0, img_f.shape[1] - 1)
    return float(np.median(img_f[r, c]))


def detect_filxys(task):
    """Detect one frame: ``(img, frame_no, noise_sd, detector, params)`` ->
    ``(frame_no, filXYs, per-ridge median signal in image units)``."""
    img, no, noise, algo, params = task
    fr = _LightFrame(img, no, noise)
    DETECTORS.create(algo, **params).detect(fr)
    sm = gaussian_filter(img.astype(np.float32), 1.0)
    sig = [_contour_signal(sm, np.asarray(e[0])) for e in fr.filXYs]
    return no, fr.filXYs, sig


def warm_up(algo):
    """Pool initializer: import the detector backend before the first task."""
    from . import ridge  # noqa: F401
    if algo == "ridge-fast":
        import ridge_detector_fast  # noqa: F401
    elif algo == "ridge":
        import ridge_detector  # noqa: F401
