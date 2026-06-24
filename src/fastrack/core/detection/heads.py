"""Point-source ("head") detection for polarity-labelled movies (FASTplus).

Python equivalent of the TrackMate spot-detection step used in the FASTplus v0.1
ImageJ pipeline:

    Gaussian pre-blur (sigma) -> single-scale Laplacian-of-Gaussian at the
    estimated head radius -> sub-pixel-refined local maxima above a quality
    (LoG-response) threshold.

TrackMate's "LoG detector" is a *single-scale* LoG at the estimated spot radius,
and its reported "quality" is the LoG response value at the maximum; both are
reproduced here (detection sigma = radius / sqrt(2)).

The heavy image ops prefer :mod:`scipy.ndimage` (a core FASTrack dependency) and
fall back to OpenCV / numpy, so detection runs identically with or without
scipy/scikit-image installed.

``HeadDetector`` plugs into the existing :data:`DETECTORS` registry; it does not
touch the filament-detection path.  It populates ``frame.heads`` (a list of
:class:`~fastrack.polarity.spot.SpotRecord`) rather than ``frame.filaments``.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import DETECTORS, Detector

try:
    from scipy import ndimage as _ndi
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


# --------------------------------------------------------------------------- #
# Low-level ops (scipy -> opencv -> numpy)
# --------------------------------------------------------------------------- #
def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.astype(np.float64, copy=False)
    if _HAVE_SCIPY:
        return _ndi.gaussian_filter(img.astype(np.float64), sigma)
    if _HAVE_CV2:
        k = int(2 * round(3 * sigma) + 1)
        return cv2.GaussianBlur(img.astype(np.float64), (k, k), sigma,
                                borderType=cv2.BORDER_REPLICATE)
    return _gaussian_blur_numpy(img.astype(np.float64), sigma)


def _gaussian_blur_numpy(img: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(round(3 * sigma))
    ax = np.arange(-radius, radius + 1)
    k = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    pad = np.pad(img, radius, mode="edge")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, pad)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, tmp)


def _laplacian(img: np.ndarray) -> np.ndarray:
    if _HAVE_SCIPY:
        return _ndi.laplace(img)
    if _HAVE_CV2:
        return cv2.Laplacian(img, cv2.CV_64F)
    out = -4.0 * img
    out[:-1, :] += img[1:, :]
    out[1:, :] += img[:-1, :]
    out[:, :-1] += img[:, 1:]
    out[:, 1:] += img[:, :-1]
    return out


def _maximum_filter(resp: np.ndarray, size: int) -> np.ndarray:
    if _HAVE_SCIPY:
        return _ndi.maximum_filter(resp, size=size, mode="constant", cval=-np.inf)
    if _HAVE_CV2:
        kernel = np.ones((size, size), np.uint8)
        return cv2.dilate(resp, kernel)
    r = size // 2
    pad = np.pad(resp, r, mode="constant", constant_values=-np.inf)
    out = np.full_like(resp, -np.inf)
    for dy in range(size):
        for dx in range(size):
            out = np.maximum(out, pad[dy:dy + resp.shape[0], dx:dx + resp.shape[1]])
    return out


def _subpixel(resp: np.ndarray, r: int, c: int):
    """2-D parabolic refinement of a peak at integer (row, col) -> (x, y)."""
    H, W = resp.shape
    if r <= 0 or r >= H - 1 or c <= 0 or c >= W - 1:
        return float(c), float(r)
    cen = resp[r, c]
    dxx = resp[r, c + 1] - 2 * cen + resp[r, c - 1]
    dyy = resp[r + 1, c] - 2 * cen + resp[r - 1, c]
    ox = -(resp[r, c + 1] - resp[r, c - 1]) / 2.0 / dxx if dxx != 0 else 0.0
    oy = -(resp[r + 1, c] - resp[r - 1, c]) / 2.0 / dyy if dyy != 0 else 0.0
    ox = float(np.clip(ox, -0.5, 0.5))
    oy = float(np.clip(oy, -0.5, 0.5))
    return c + ox, r + oy


# --------------------------------------------------------------------------- #
# Functional API (used by the detector and by tests / prototypes)
# --------------------------------------------------------------------------- #
def log_response(img: np.ndarray, sigma: float) -> np.ndarray:
    """Scale-normalized Laplacian-of-Gaussian; bright point sources -> positive."""
    return -(sigma ** 2) * _laplacian(_gaussian_blur(img, sigma))


def detect_spots(
    img: np.ndarray,
    frame: int = 0,
    *,
    gaussian_sigma: float = 1.5,
    radius: float = 5.0,
    quality_threshold: float = 5.0,
    subpixel: bool = True,
    min_separation: Optional[float] = None,
) -> List["object"]:
    """Detect head spots in one 2-D frame; returns a list of ``SpotRecord``."""
    from ...polarity.spot import SpotRecord  # local import avoids a cycle

    img = np.asarray(img, dtype=np.float64)
    pre = _gaussian_blur(img, gaussian_sigma) if gaussian_sigma > 0 else img
    det_sigma = float(radius) / np.sqrt(2.0)
    resp = log_response(pre, det_sigma)

    md = int(round(min_separation if min_separation is not None else radius))
    md = max(1, md)
    peaks = (resp == _maximum_filter(resp, 2 * md + 1))
    ys, xs = np.nonzero(peaks)

    spots: List[SpotRecord] = []
    for r, c in zip(ys, xs):
        q = float(resp[r, c])
        if q < quality_threshold:
            continue
        x, y = _subpixel(resp, int(r), int(c)) if subpixel else (float(c), float(r))
        spots.append(SpotRecord(frame=int(frame), x=x, y=y, quality=q, radius=float(radius)))
    spots.sort(key=lambda s: s.quality, reverse=True)
    return spots


# --------------------------------------------------------------------------- #
# Detector strategy (registry plug-in)
# --------------------------------------------------------------------------- #
@DETECTORS.register("heads-log")
class HeadDetector(Detector):
    """Single-scale LoG point detector for polarity heads (≈ TrackMate LoG)."""

    def __init__(self, gaussian_sigma=1.5, radius=5.0, quality_threshold=5.0,
                 subpixel=True, min_separation=None):
        self.gaussian_sigma = gaussian_sigma
        self.radius = radius
        self.quality_threshold = quality_threshold
        self.subpixel = subpixel
        self.min_separation = min_separation

    def detect(self, frame):
        """Populate ``frame.heads`` with detected head spots from ``frame.img``."""
        frame.heads = detect_spots(
            frame.img, frame=getattr(frame, "frame_no", 0),
            gaussian_sigma=self.gaussian_sigma, radius=self.radius,
            quality_threshold=self.quality_threshold, subpixel=self.subpixel,
            min_separation=self.min_separation,
        )
        return frame.heads

    def assess_quality(self, frame):
        return "good"


def backend_info() -> str:
    return "scipy=%s opencv=%s" % (_HAVE_SCIPY, _HAVE_CV2)
