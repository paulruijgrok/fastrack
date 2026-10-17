"""Ridge-detection filament detectors (optional).

Wraps Steger's multi-scale unbiased curvilinear-structure detector as a FASTrack
:class:`Detector`.  Instead of the entropy/watershed segmentation, it detects
filament centerlines directly and maps each detected ridge ``Line`` onto
FASTrack's ``filXYs`` representation, so the rest of the pipeline (tracking,
statistics, plotting) is unchanged.

Two registered variants share identical mapping logic and differ only in the
upstream class they instantiate:

* ``"ridge"`` -- the reference ``ridge-detector`` package (lxfhfut/ridge-detector).
* ``"ridge-fast"`` -- ``ridge-detector-fast`` (paulruijgrok/ridge-detector),
  a numerically-identical, ~4x faster drop-in (analytical 2x2 eigendecomposition,
  float32, OpenCV separable filters, numba-compiled contour tracing).

Both dependencies are **optional** and imported lazily, so the default FASTrack
install requires neither.  Install with ``pip install 'fastrack[ridge]'`` or
``pip install 'fastrack[ridge-fast]'``.
"""
import numpy as np

from .base import DETECTORS, Detector


def estimate_frame_noise(img):
    """Fallback pixel-noise estimate (std) for a frame with no known noise.

    Robust spread (MAD) of the residual after a 5x5 median filter.  Valid for
    unclipped images; preprocessed frames clip the background to 0, which hides
    half the noise -- those carry the exact value from the preprocessing model
    in ``frame.noise_sd`` instead.
    """
    from scipy.ndimage import median_filter
    f = np.asarray(img, dtype=np.float32)
    r = f - median_filter(f, size=5)
    return 1.4826 * float(np.median(np.abs(r - np.median(r))))


def contour_snr(contours, img, noise_sd, smooth_sigma=1.0, statistic="median"):
    """Signal-to-noise ratio of each ridge contour.

    The ``statistic`` (default **median**) of the lightly smoothed image along
    the centerline, minus the frame's median (the background level: ~0 for
    preprocessed frames), divided by the pixel noise.  The median makes the
    signal insensitive to isolated spikes (cosmic rays, hot pixels): a spike
    covering less than half of the centerline does not change it.  Noise lines
    traced by the detector sit at SNR ~1-2; real filaments are well above.
    ``statistic="mean"`` is kept for comparison.  Returns a float array (one
    value per contour).
    """
    from scipy.ndimage import gaussian_filter
    if not contours:
        return np.zeros(0)
    f = np.asarray(img, dtype=np.float32)
    sm = gaussian_filter(f, smooth_sigma) if smooth_sigma else f
    bg = float(np.median(f))
    noise = max(float(noise_sd), 1e-9)
    h, w = f.shape[:2]
    out = np.empty(len(contours))
    for k, cont in enumerate(contours):
        rows = np.clip(np.rint(np.asarray(cont.row, dtype=float)).astype(int), 0, h - 1)
        cols = np.clip(np.rint(np.asarray(cont.col, dtype=float)).astype(int), 0, w - 1)
        vals = sm[rows, cols]
        sig = float(np.median(vals)) if statistic == "median" else float(vals.mean())
        out[k] = (sig - bg) / noise if len(rows) else 0.0
    return out


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


class _RidgeAdapterBase(Detector):
    """Shared ridge->filXYs adapter; subclasses pick the upstream detector class.

    Parameters mirror ``ridge_detector.RidgeDetector``; defaults are tuned for
    bright filaments on a dark background (``dark_line=False``), as in the
    gliding-assay movies.  ``min_snr`` (FASTrack's own, default 0 = off) drops
    ridges whose median centerline intensity is below ``min_snr`` x the frame's
    pixel noise
    (see :func:`contour_snr`); it makes the result insensitive to the contrast
    thresholds on low-SNR data, where those thresholds otherwise trace noise.  The two concrete detectors are interchangeable and
    use the same parameters -- only the underlying implementation differs.
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
        min_snr=0.0,
    ):
        self._detector_cls = self._import_detector_cls()
        # Signal-to-noise post-filter (0 = off).  Not an upstream parameter, so
        # it is kept out of ``self.params`` (which is passed to the detector).
        self.min_snr = float(min_snr or 0.0)
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

    @staticmethod
    def _import_detector_cls():  # pragma: no cover - overridden
        raise NotImplementedError

    def detect(self, frame):
        """Detect ridge filaments in ``frame.img`` and populate ``frame``.

        A fresh upstream detector is created per frame (it carries per-image
        state), so this is safe under the multiprocessing per-frame workers.
        """
        det = self._detector_cls(**self.params)
        det.detect_lines(frame.img)
        contours = det.contours or []
        if self.min_snr > 0 and contours:
            noise = getattr(frame, "noise_sd", None)
            if noise is None:
                noise = estimate_frame_noise(frame.img)
            snr = contour_snr(contours, frame.img, noise)
            contours = [c for c, v in zip(contours, snr) if v >= self.min_snr]
        frame.filXYs = _contours_to_filxys(contours, frame.img)
        frame.filXY2filaments()

    def assess_quality(self, frame):
        # The ridge detector has no acquisition-quality gate of its own.
        return "good"


@DETECTORS.register("ridge")
class RidgeLineDetector(_RidgeAdapterBase):
    """Reference ridge detector (``ridge-detector``; lxfhfut/ridge-detector)."""

    @staticmethod
    def _import_detector_cls():
        try:
            from ridge_detector import RidgeDetector
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "The 'ridge' detector requires the optional 'ridge-detector' "
                "dependency. Install it with:\n    pip install 'fastrack[ridge]'"
            ) from exc
        return RidgeDetector


@DETECTORS.register("ridge-fast")
class OptimizedRidgeLineDetector(_RidgeAdapterBase):
    """Speed-optimized ridge detector (``ridge-detector-fast``).

    Numerically identical to ``"ridge"`` (verified: same contour and point
    counts) but ~4x faster end to end.  See paulruijgrok/ridge-detector.
    """

    @staticmethod
    def _import_detector_cls():
        try:
            from ridge_detector_fast import OptimizedRidgeDetector
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "The 'ridge-fast' detector requires the optional "
                "'ridge-detector-fast' dependency. Install it with:\n"
                "    pip install 'fastrack[ridge-fast]'"
            ) from exc
        return OptimizedRidgeDetector
