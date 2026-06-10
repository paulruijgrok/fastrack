"""Ridge-detection filament detector (optional).

Wraps the upstream ``ridge-detector`` package (Steger's multi-scale unbiased
curvilinear-structure detector) as a FASTrack :class:`Detector`.  Instead of the
entropy/watershed segmentation, it detects filament centerlines directly and
maps each detected ridge ``Line`` onto FASTrack's ``filXYs`` representation, so
the rest of the pipeline (tracking, statistics, plotting) is unchanged.

The ``ridge-detector`` dependency is **optional** and imported lazily, so the
default FASTrack install does not require it.  Install it with::

    pip install 'fastrack[ridge]'
"""
import numpy as np

from .base import DETECTORS, Detector


def _contours_to_filxys(contours, img):
    """Convert upstream ridge ``Line`` objects into FASTrack ``filXYs`` entries.

    Each entry is ``[contour, width, density, midpoint]`` where ``contour`` is an
    ``(N, 2)`` integer array of ``[row, col]`` pixel coordinates (matching the
    entropy detector's convention), ``width`` is the mean total ridge width,
    ``density`` is the mean image intensity along the centerline, and
    ``midpoint`` is the central contour point.  ``img`` supplies intensities.

    Ridge coordinates are sub-pixel; they are rounded to integer pixels here so
    that every downstream consumer (including movie reconstruction, which indexes
    an image with the contour) works unchanged.  Sub-pixel precision can be
    revisited later.
    """
    height, width = img.shape[:2]
    filxys = []
    for cont in contours:
        n = int(getattr(cont, "num", 0))
        if n < 2:
            continue
        rows = np.clip(np.rint(np.asarray(cont.row, dtype=float)).astype(int), 0, height - 1)
        cols = np.clip(np.rint(np.asarray(cont.col, dtype=float)).astype(int), 0, width - 1)
        contour = np.column_stack([rows, cols])

        wl = getattr(cont, "width_l", None)
        wr = getattr(cont, "width_r", None)
        if wl is not None and wr is not None:
            fil_width = float(np.mean(np.asarray(wl, dtype=float) + np.asarray(wr, dtype=float)))
        else:
            fil_width = 0.0

        density = float(np.mean(img[rows, cols]))
        midpoint = contour[len(contour) // 2 - 1]
        filxys.append([contour, fil_width, density, midpoint])
    return filxys


@DETECTORS.register("ridge")
class RidgeLineDetector(Detector):
    """Detect filaments as ridges via the upstream ``ridge-detector`` package.

    Parameters mirror ``ridge_detector.RidgeDetector``; defaults are tuned for
    bright filaments on a dark background (``dark_line=False``), as in the
    gliding-assay movies.
    """

    def __init__(
        self,
        line_widths=(3,),
        low_contrast=50,
        high_contrast=150,
        min_len=10,
        max_len=0,
        dark_line=False,
        estimate_width=True,
        extend_line=False,
        correct_pos=False,
    ):
        try:
            from ridge_detector import RidgeDetector as _UpstreamRidgeDetector
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "The 'ridge' detector requires the optional 'ridge-detector' "
                "dependency. Install it with:\n    pip install 'fastrack[ridge]'"
            ) from exc

        self._detector_cls = _UpstreamRidgeDetector
        self.params = dict(
            line_widths=list(line_widths),
            low_contrast=low_contrast,
            high_contrast=high_contrast,
            min_len=min_len,
            max_len=max_len,
            dark_line=dark_line,
            estimate_width=estimate_width,
            extend_line=extend_line,
            correct_pos=correct_pos,
        )

    def detect(self, frame):
        """Detect ridge filaments in ``frame.img`` and populate ``frame``.

        A fresh upstream detector is created per frame (it carries per-image
        state), so this is safe under the multiprocessing per-frame workers.
        """
        det = self._detector_cls(**self.params)
        det.detect_lines(frame.img)
        frame.filXYs = _contours_to_filxys(det.contours or [], frame.img)
        frame.filXY2filaments()

    def assess_quality(self, frame):
        # The ridge detector has no acquisition-quality gate of its own.
        return "good"
